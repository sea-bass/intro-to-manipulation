""" Utilities for differential IK. """

import numpy as np
import pinocchio
import time

from ..core.utils import (
    check_collisions_at_state,
    check_within_limits,
    get_random_state,
)
from ..visualization.meshcat_utils import visualize_frame

VIZ_INITIAL_RENDER_TIME = 0.5
VIZ_SLEEP_TIME = 0.05


class DifferentialIkOptions:
    """Options for differential IK."""

    def __init__(
        self,
        max_iters=200,
        max_retries=10,
        max_translation_error=1e-3,
        max_rotation_error=1e-3,
        damping=1e-3,
        min_step_size=0.1,
        max_step_size=0.5,
    ):
        """
        Initializes a set of differential IK options.

        Parameters
        ----------
            max_iters : int
                Maximum number of iterations per try.
            max_retries : int
                Maximum number of retries with random restarts.
                If set to 0, only the initial state provided will be used.
            max_translation_error : float
                Maximum translation error, in meters, to consider IK solved.
            max_rotation_error : float
                Maximum rotation error, in radians, to consider IK solved.
            damping : float
                Damping value, between 0 and 1, for the Jacobian pseudoinverse.
                Setting this to a nonzero value is using Levenberg-Marquardt.
            min_step_size : float
                Minimum gradient step, between 0 and 1, based on ratio of current distance to target to initial distance to target.
                To use a fixed step size, set both minimum and maximum values to be equal.
            max_step_size : float
                Maximum gradient step, between 0 and 1, based on ratio of current distance to target to initial distance to target.
                To use a fixed step size, set both minimum and maximum values to be equal.
        """
        self.max_iters = max_iters
        self.max_retries = max_retries
        self.max_translation_error = max_translation_error
        self.max_rotation_error = max_rotation_error
        self.damping = damping
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size


class DifferentialIk:
    """Differential IK solver.

    This is a numerical IK solver that uses the manipulator's Jacobian to take first-order steps towards a solution.
    It contains several of the common options such as damped least squares (Levenberg-Marquardt), random restarts, and nullspace projection.

    Some good resources:
      * https://motion.cs.illinois.edu/RoboticSystems/InverseKinematics.html
      * https://homes.cs.washington.edu/~todorov/courses/cseP590/06_JacobianMethods.pdf
      * https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    """

    def __init__(
        self,
        model,
        collision_model=None,
        data=None,
        visualizer=None,
        options=DifferentialIkOptions(),
    ):
        """
        Creates an instance of a DifferentialIk solver.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`, optional
                The model to use for collision checking. If None, no collision checking takes place.
            data : `pinocchio.Data`, optional
                The model data to use for this solver. If None, data is created automatically.
            visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`, optional
                The visualizer to use for this solver.
            options : `DifferentialIkOptions`, optional
                The options to use for solving IK. If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model

        if not data:
            data = model.createData()
        self.data = data

        self.visualizer = visualizer
        self.options = options

    def solve(
        self,
        target_frame,
        target_tform,
        init_state=None,
        nullspace_components=[],
        verbose=False,
    ):
        """
        Solves an IK query.

        Parameters
        ----------
            target_frame : str
                The name of the target frame in the model.
            target_tform : `pinocchio.SE3`
                The desired transformation of the target frame in the model.
            init_state : array-like, optional
                The initial state to solve from. If not specified, a random initial state will be selected.
            nullspace_components : list[function], optional
                An optional list of nullspace components to use when solving.
                These components must take the form `lambda model, q: component(model, q, <other_args>)`.
            verbose : bool, optional
                If True, prints additional information to the console.

        Returns
        -------
            array-like or None
                A list of joint configuration values with the solution, if one was found. Otherwise, returns None.
        """
        target_frame_id = self.model.getFrameId(target_frame)

        # Create a random initial state, if not specified
        if init_state is None:
            init_state = get_random_state(self.model)
        if self.visualizer:
            self.visualizer.displayFrames(True, frame_ids=[target_frame_id])
            visualize_frame(self.visualizer, "ik_target_pose", target_tform)

        # Initialize IK
        solved = False
        n_tries = 0
        q_cur = init_state
        initial_error_norm = None

        if self.visualizer:
            self.visualizer.display(q_cur)
            time.sleep(VIZ_INITIAL_RENDER_TIME)  # Needed to render

        while n_tries <= self.options.max_retries:
            n_iters = 0
            while n_iters < self.options.max_iters:
                # Compute forward kinematics at the current state
                pinocchio.framesForwardKinematics(self.model, self.data, q_cur)
                cur_tform = self.data.oMf[target_frame_id]

                # Check the error using actInv
                error = target_tform.actInv(cur_tform)
                error = -pinocchio.log(error).vector
                if (
                    np.linalg.norm(error[:3]) < self.options.max_translation_error
                    and np.linalg.norm(error[3:]) < self.options.max_rotation_error
                ):
                    # Wrap to the range -/+ pi, and then check joint limits and collision.
                    q_cur = (q_cur + np.pi) % (2 * np.pi) - np.pi
                    if check_within_limits(self.model, q_cur):
                        if self.collision_model is not None:
                            if check_collisions_at_state(
                                self.model, self.collision_model, q_cur
                            ):
                                if verbose:
                                    print(
                                        "Solved and within joint limits, but in collision."
                                    )
                            else:
                                solved = True
                                if verbose:
                                    print(
                                        "Solved, within joint limits, and collision-free!"
                                    )
                        else:
                            solved = True
                            if verbose:
                                print("Solved and within joint limits!")
                    else:
                        if verbose:
                            print("Solved, but outside joint limits.")
                    break

                # Calculate the Jacobian
                J = pinocchio.computeFrameJacobian(
                    self.model,
                    self.data,
                    q_cur,
                    target_frame_id,
                    pinocchio.ReferenceFrame.LOCAL,
                )

                # Solve for the gradient using damping and nullspace components,
                # as specified
                jjt = J.dot(J.T) + self.options.damping**2 * np.eye(6)

                # Compute the gradient descent step size
                error_norm = np.linalg.norm(error)
                if not initial_error_norm:
                    initial_error_norm = error_norm
                alpha = self.options.min_step_size + (
                    1.0 - error_norm / initial_error_norm
                ) * (self.options.max_step_size - self.options.min_step_size)

                # Gradient descent step
                if not nullspace_components:
                    q_cur += alpha * J.T @ np.linalg.solve(jjt, error)
                else:
                    nullspace_term = sum(
                        [comp(self.model, q_cur) for comp in nullspace_components]
                    )
                    q_cur += alpha * (
                        J.T @ (np.linalg.solve(jjt, error - J @ (nullspace_term)))
                        + nullspace_term
                    )

                n_iters += 1

                if self.visualizer:
                    self.visualizer.display(q_cur)
                    time.sleep(VIZ_SLEEP_TIME)

            # Check results at the end of this try
            if solved:
                if verbose:
                    print(f"Solved in {n_tries+1} tries.")
                break
            else:
                q_cur = get_random_state(self.model)
                n_tries += 1
                if verbose:
                    print(f"Retry {n_tries}")

        # Check final results
        if solved:
            return q_cur
        else:
            return None
