""" Utilities for trajectory optimization based planning. """

import numpy as np
import warnings

import pinocchio
from pydrake.autodiffutils import AutoDiffXd, ExtractValue
from pydrake.solvers import MathematicalProgram, Solve
from pyroboplan.trajectory.polynomial import CubicPolynomialTrajectory


class CubicTrajectoryOptimizationOptions:
    """Options for cubic polynomial trajectory optimization."""

    def __init__(
        self,
        num_waypoints=3,
        samples_per_segment=11,
        min_segment_time=0.01,
        max_segment_time=10.0,
        min_vel=-np.inf,
        max_vel=np.inf,
        min_accel=-np.inf,
        max_accel=np.inf,
        min_jerk=-np.inf,
        max_jerk=np.inf,
        check_collisions=False,
        min_collision_dist=0.0,
        collision_influence_dist=0.05,
    ):
        """
        Initializes a set of options for cubic polynomial trajectory optimization.

        Parameters
        ----------
            num_waypoints : int
                The number of waypoints in the trajectory. Must be greater than or equal to 2.
            samples_per_segment : int
                The number of samples to take along each trajectory segment for setting kinematic constraints.
            min_segment_time : float
                The minimum duration of a trajectory segment, in seconds.
            max_segment_time : float
                The maximum duration of a trajectory segment, in seconds.
            min_vel : float, or array-like
                The minimum velocity along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            max_vel : float or array-like
                The maximum velocity along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            min_accel : float, or array-like
                The minimum acceleration along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            max_accel : float or array-like
                The maximum acceleration along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            min_jerk : float, or array-like
                The minimum jerk along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            max_jerk : float or array-like
                The maximum jerk along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            check_collisions: bool
                If true, adds collision constraints to trajectory optimization.
            min_collision_dist : float
                The minimum allowable collision distance, in meters.
            collision_influence_dist : float
                The distance for collision/distance checks, in meters, above which to ignore results.
        """
        if num_waypoints < 2:
            raise ValueError(
                "The number of waypoints must be greater than or equal to 2."
            )
        if min_segment_time <= 0:
            raise ValueError("The minimum segment time must be positive.")

        self.num_waypoints = num_waypoints
        self.samples_per_segment = samples_per_segment
        self.min_segment_time = min_segment_time
        self.max_segment_time = max_segment_time
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.min_jerk = min_jerk
        self.max_jerk = max_jerk
        self.check_collisions = check_collisions
        self.min_collision_dist = min_collision_dist
        self.collision_influence_dist = collision_influence_dist


class CubicTrajectoryOptimization:
    """
    Trajectory optimization based planner.

    This uses the direct collocation approach to optimize over waypoint and collocation point placements that describe a multi-segment cubic polynomial trajectory.

    Some good resources include:
        * Matthew Kelly's tutorials: https://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/index.html
        * Russ Tedrake's manipulation course book: https://underactuated.mit.edu/trajopt.html
    """

    def __init__(
        self, model, collision_model, options=CubicTrajectoryOptimizationOptions()
    ):
        """
        Creates an instance of a cubic trajectory optimization planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.GeometryModel`
                The model to use for collision checking.
            options : `CubicTrajectoryOptimizationOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.options = options

    def _process_limits(self, limits, num_dofs, name):
        """
        Helper function to process kinematics limits options.

          * If the input limits are scalar, reshape them to the number of degrees of freedom.
          * Whether the input limits are scalar, a list, or a numpy array, always return a numpy array.
          * If the input limits are not scalar, but the wrong size, this will raise an Exception.

        Parameters
        ----------
            limits : None, float, or array-like
                The input limits in various formats.
            num_dofs : int
                The number of degrees of freedom.
            name : str
                The name of the input limits, to generate a descriptive error.

        Return
        ------
            numpy.ndarray
                The processed limits, for compatibility with trajectory optimization.
        """
        limits = np.array(limits)
        if len(limits.shape) == 0 or limits.shape[0] == 1:
            limits = limits * np.ones(num_dofs)
        elif limits.shape != (num_dofs,):
            raise ValueError(f"{name} vector must have shape ({num_dofs},)")
        return limits

    def _eval_position(self, x, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory position.

        This directly relates to equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

        Parameters
        ----------
            x : pydrake.autodiffutils.AutoDiffXd
                The waypoint position points.
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The position along the specified segment evaluated at the specified step.
        """
        return (
            x[k, n]
            + x_d[k, n] * (step * h[k])
            + 0.5
            * (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
            * (step * h[k]) ** 2
            / h[k]
            + (2.0 / 3.0)
            * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
            * (step * h[k]) ** 3
            / h[k] ** 2
        )

    def _eval_velocity(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory velocity.

        This is the first derivative of equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

        Parameters
        ----------
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The velocity along the specified segment evaluated at the specified step.
        """
        return (
            x_d[k, n]
            + (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
            * (step * h[k])
            / h[k]
            + 2.0
            * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
            * (step * h[k]) ** 2
            / h[k] ** 2
        )

    def _eval_acceleration(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory acceleration.

        This is the second derivative of equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

        Parameters
        ----------
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The acceleration along the specified segment evaluated at the specified step.
        """
        return (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n]) / h[k] + 4.0 * (
            x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]
        ) * (step * h[k]) / h[k] ** 2

    def _eval_jerk(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory jerk.

        This is the third derivative of equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

        Parameters
        ----------
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The jerk along the specified segment evaluated at the specified step.
        """
        return 8.0 * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]) / h[k] ** 2

    def get_collision_pairs_involving(self, obj_list):
        pairs = []
        collision_ids = set()

        from pyroboplan.core.utils import get_collision_geometry_ids

        for obj in obj_list:
            ids = get_collision_geometry_ids(self.model, self.collision_model, obj)
            collision_ids.update(ids)
        for idx, p in enumerate(self.collision_model.collisionPairs):
            if p.first in collision_ids or p.second in collision_ids:
                pairs.append(idx)

        return pairs

    def _collision_constraint(self, q_val, influence_dist, collision_pairs):
        """
        Helper function to evaluate collision constraint and its gradients.

        These calculations are based off the following resources:
          * https://typeset.io/pdf/a-collision-free-mpc-for-whole-body-dynamic-locomotion-and-1l6itpfk.pdf
          * https://laas.hal.science/hal-04425002
        """
        if q_val.dtype != float:
            q = ExtractValue(q_val)

            # Compute collision and distance checks
            pinocchio.framesForwardKinematics(self.model, self.data, q)
            pinocchio.computeCollisions(
                self.model,
                self.data,
                self.collision_model,
                self.collision_data,
                q,
                False,
            )
            pinocchio.computeDistances(
                self.model, self.data, self.collision_model, self.collision_data, q
            )

            autodiffs = []
            for pairs in collision_pairs:
                # Get the minimum distance in the allowable range
                min_distance_idx = -1
                min_distance = influence_dist
                gradient = np.zeros_like(q)

                for p in pairs:
                    cp = self.collision_model.collisionPairs[p]
                    cr = self.collision_data.collisionResults[p]
                    dr = self.collision_data.distanceResults[p]

                    dist = dr.min_distance

                    if dist <= influence_dist and dist < min_distance:
                        min_distance_idx = p
                        min_distance = dist

                # Find the collision Jacobian for the closest point pair.
                if min_distance_idx >= 0:
                    cr = self.collision_data.collisionResults[min_distance_idx]
                    dr = self.collision_data.distanceResults[min_distance_idx]
                    cp = self.collision_model.collisionPairs[min_distance_idx]

                    if cr.isCollision():
                        # According to the HPP-FCL documentation, the normal always points from object1 to object2.
                        contact = cr.getContact(0)
                        coll_points = [
                            contact.pos,
                            contact.pos + contact.normal * contact.penetration_depth,
                        ]
                    else:
                        coll_points = [dr.getNearestPoint1(), dr.getNearestPoint2()]
                    distance_vec = coll_points[1] - coll_points[0]

                    parent_frame1 = self.collision_model.geometryObjects[
                        cp.first
                    ].parentFrame
                    parent_frame2 = self.collision_model.geometryObjects[
                        cp.second
                    ].parentFrame
                    if parent_frame1 >= self.model.nframes:
                        parent_frame1 = 0
                    Jframe1 = pinocchio.computeFrameJacobian(
                        self.model,
                        self.data,
                        q,
                        parent_frame1,
                        pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    t_frame1_to_point1 = pinocchio.SE3(
                        np.eye(3),
                        coll_points[0] - self.data.oMf[parent_frame1].translation,
                    )
                    Jcoll1 = t_frame1_to_point1.toActionMatrix()[3:, :] @ Jframe1

                    if parent_frame2 >= self.model.nframes:
                        parent_frame2 = 0
                    Jframe2 = pinocchio.computeFrameJacobian(
                        self.model,
                        self.data,
                        q,
                        parent_frame2,
                        pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    t_frame2_to_point2 = pinocchio.SE3(
                        np.eye(3),
                        coll_points[1] - self.data.oMf[parent_frame2].translation,
                    )
                    Jcoll2 = t_frame2_to_point2.toActionMatrix()[3:, :] @ Jframe2

                    # Calculate the gradients.
                    distance_vec = distance_vec / np.linalg.norm(distance_vec)
                    gradient = np.sign(dist) * distance_vec @ (Jcoll2 - Jcoll1)

                autodiffs.append(
                    AutoDiffXd(
                        min_distance,
                        gradient,
                    )
                )

            return np.array(autodiffs)
        else:
            # This case should not be used by optimization, but can be used when testing without autodiff.
            pinocchio.computeDistances(
                self.model, self.data, self.collision_model, self.collision_data, q_val
            )
            return min([dr.min_distance for dr in self.collision_data.distanceResults])

    def plan(self, q_path, init_path=None):
        """
        Plans a trajectory from a start to a goal configuration, or along an entire trajectory.

        If the input list has 2 elements, then this is assumed to be the start and goal configurations.
        The intermediate waypoints will be determined automatically.

        If the input list has more than 2 elements, then these are the actual waypoints that must be achieved.

        Parameters
        ----------
            q_path : list[array-like]
                A list of joint configurations describing the desired motion.
            init_path : list[array-like], optional
                If set, defines the initial guess for the path waypoints.

        Return
        ------
            Optional[pyroboplan.trajectory.polynomial.CubicPolynomialTrajectory]
                The resulting trajectory, or None if optimization failed
        """
        if len(q_path) == 0:
            warnings.warn("Cannot optimize over an empty path.")
            return None
        num_waypoints = self.options.num_waypoints
        num_dofs = len(q_path[0])

        if len(q_path) == num_waypoints:
            fully_specified_path = True
        elif len(q_path) == 2:
            fully_specified_path = False
        else:
            raise ValueError("Path must either be length 2 or equal to num_waypoints.")

        # Preprocess the kinematic limits.
        min_vel = self._process_limits(self.options.min_vel, num_dofs, "min_vel")
        max_vel = self._process_limits(self.options.max_vel, num_dofs, "max_vel")
        min_accel = self._process_limits(self.options.min_accel, num_dofs, "min_accel")
        max_accel = self._process_limits(self.options.max_accel, num_dofs, "max_accel")
        min_jerk = self._process_limits(self.options.min_jerk, num_dofs, "min_jerk")
        max_jerk = self._process_limits(self.options.max_jerk, num_dofs, "max_jerk")

        # Initialize the basic program and its variables.
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(num_waypoints, num_dofs)
        x_d = prog.NewContinuousVariables(num_waypoints, num_dofs)
        xc = prog.NewContinuousVariables(num_waypoints - 1, num_dofs)
        xc_d = prog.NewContinuousVariables(num_waypoints - 1, num_dofs)
        h = prog.NewContinuousVariables(num_waypoints - 1)

        # Initial, final, and intermediate waypoint conditions.
        if fully_specified_path:
            for idx in range(num_waypoints):
                prog.AddBoundingBoxConstraint(q_path[idx], q_path[idx], x[idx, :])
        else:
            prog.AddBoundingBoxConstraint(q_path[0], q_path[0], x[0, :])
            prog.AddBoundingBoxConstraint(q_path[-1], q_path[-1], x[-1, :])
        # Initial and final velocities should always be zero.
        prog.AddBoundingBoxConstraint(0.0, 0.0, x_d[0, :])
        prog.AddBoundingBoxConstraint(0.0, 0.0, x_d[num_waypoints - 1, :])

        # Collision checking at the waypoints and collocation points.
        min_dist = self.options.min_collision_dist
        from functools import partial

        link_list = [
            "panda_hand",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        min_dist_val = [min_dist for _ in link_list]
        max_dist_val = [np.inf for _ in link_list]
        all_pairs = []
        for link in link_list:
            all_pairs.append(self.get_collision_pairs_involving([link]))
        collision_expr = partial(
            self._collision_constraint,
            influence_dist=self.options.collision_influence_dist,
            collision_pairs=all_pairs,
        )
        if self.options.check_collisions:
            for k in range(1, num_waypoints - 1):
                prog.AddConstraint(collision_expr, min_dist_val, max_dist_val, x[k, :])
            for k in range(num_waypoints - 1):
                prog.AddConstraint(collision_expr, min_dist_val, max_dist_val, xc[k, :])

        for n in range(num_dofs):
            # Collocation point constraints.
            # Specifically, this constrains the position and velocities of the collocation points to be
            # expressed in terms of the waypoint positions and velocities, assuming cubic splines.
            # This is described in the "Direct Collocation" section here:
            # https://underactuated.mit.edu/trajopt.html
            for k in range(num_waypoints - 1):
                prog.AddConstraint(
                    xc[k, n]
                    == 0.5 * (x[k, n] + x[k + 1, n])
                    + (h[k] / 8.0) * (x_d[k, n] - x_d[k + 1, n])
                )
                prog.AddConstraint(
                    xc_d[k, n]
                    == -(1.5 / h[k]) * (x[k, n] - x[k + 1, n])
                    - 0.25 * (x_d[k, n] + x_d[k + 1, n])
                )

                # Sample points along each segment to evaluate the position and velocity constraints, since
                # they are cubic and quadratic (respectively) and could overshoot the waypoints and collocation points.
                for step in np.linspace(0.0, 1.0, self.options.samples_per_segment):
                    pos = self._eval_position(x, x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(pos <= self.model.upperPositionLimit[n])
                    prog.AddConstraint(pos >= self.model.lowerPositionLimit[n])

                    vel = self._eval_velocity(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(vel >= min_vel[n])
                    prog.AddConstraint(vel <= max_vel[n])

                # The acceleration and jerk are linear and piecewise constant (respectively),
                # so these can be constrained simply by checking each endpoint.
                for step in [0.0, 1.0]:
                    accel = self._eval_acceleration(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(accel >= min_accel[n])
                    prog.AddConstraint(accel <= max_accel[n])

                    jerk = self._eval_jerk(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(jerk >= min_jerk[n])
                    prog.AddConstraint(jerk <= max_jerk[n])

            # Enforce acceleration continuity between segments.
            # That is, the final acceleration of the kth waypoint must equal the initial acceleration of the (k+1)th waypoint.
            for k in range(num_waypoints - 2):
                prog.AddConstraint(
                    self._eval_acceleration(x_d, xc_d, h, k, n, 1.0)
                    == self._eval_acceleration(x_d, xc_d, h, k + 1, n, 0.0)
                )

        # Cost and bounds on trajectory segment times.
        prog.AddBoundingBoxConstraint(
            self.options.min_segment_time, self.options.max_segment_time, h
        )
        prog.AddQuadraticCost(
            Q=np.eye(num_waypoints - 1),
            b=np.zeros(num_waypoints - 1),
            c=0.0,
            vars=h,
        )

        # Set initial conditions to help search.
        if init_path or fully_specified_path:
            # Set initial guess assuming collocation points are exactly between the waypoints.
            if init_path:
                q_path = init_path
            prog.SetInitialGuess(x, np.array(q_path))
            init_collocation_points = []
            for k in range(num_waypoints - 1):
                init_collocation_points.append(0.5 * (q_path[k] + q_path[k + 1]))
            prog.SetInitialGuess(xc, np.array(init_collocation_points))
        else:
            # Set initial guess assuming linear trajectory from start to end.
            init_points = np.linspace(q_path[0], q_path[-1], 2 * num_waypoints - 1)
            prog.SetInitialGuess(x, init_points[::2])
            prog.SetInitialGuess(xc, init_points[1::2])

        h_init = 0.5 * (self.options.max_segment_time - self.options.min_segment_time)
        prog.SetInitialGuess(h, h_init * np.ones(num_waypoints - 1))
        prog.SetInitialGuess(x_d, np.zeros((num_waypoints, num_dofs)))
        prog.SetInitialGuess(xc_d, np.zeros((num_waypoints - 1, num_dofs)))

        # Solve the program.
        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed.")
            return None

        # Unpack the values.
        h_opt = result.GetSolution(h)
        x_opt = result.GetSolution(x)
        x_d_opt = result.GetSolution(x_d)

        # Generate the cubic trajectory and return it.
        t_opt = [0] + list(np.cumsum(h_opt))
        self.latest_trajectory = CubicPolynomialTrajectory(
            np.array(t_opt),
            np.array(x_opt.T),
            np.array(x_d_opt.T),
        )
        return self.latest_trajectory
