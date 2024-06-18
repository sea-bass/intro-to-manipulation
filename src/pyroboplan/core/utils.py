""" Core utilities for robot modeling. """

import copy
import numpy as np
import pinocchio

from ..models.utils import create_ellipsoid_mesh, add_ellipsoid_model


def check_collisions_at_state(
    model, collision_model, q, data=None, collision_data=None
):
    """
    Checks whether a specified joint configuration is collision-free.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        q : array-like
            The joint configuration of the model.
        data : `pinocchio.Data`, optional
            The model data to use for collision checking. If None, data is created automatically.
        collision_data : `pinocchio.GeometryData`, optional
            The collision_model data to use for collision checking. If None, data is created automatically.

    Returns
    -------
        bool
            True is there are any collisions, otherwise False.
    """
    if not data:
        data = model.createData()
    if not collision_data:
        collision_data = collision_model.createData()
    stop_at_first_collision = True  # For faster computation

    pinocchio.computeCollisions(
        model, data, collision_model, collision_data, q, stop_at_first_collision
    )
    return np.any([cr.isCollision() for cr in collision_data.collisionResults])


def check_collisions_along_path(
    model, collision_model, q_path, data=None, collision_data=None
):
    """
    Checks whether a path consisting of multiple joint configurations is collision-free.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        q_path : list[array-like]
            A list of joint configurations describing the path.
        data : `pinocchio.Data`, optional
            The model data to use for collision checking. If None, data is created automatically.
        collision_data : `pinocchio.GeometryData`, optional
            The collision_model data to use for collision checking. If None, data is created automatically.

    Returns
    -------
        bool
            True is there are any collisions, otherwise False.
    """
    if not data:
        data = model.createData()
    if not collision_data:
        collision_data = collision_model.createData()
    stop_at_first_collision = True  # For faster computation

    for q in q_path:
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, q, stop_at_first_collision
        )
        if np.any([cr.isCollision() for cr in collision_data.collisionResults]):
            return True

    return False


def configuration_distance(q_start, q_end):
    """
    Returns the distance between two joint configurations.

    Parameters
    ----------
        q_start : array-like
            The start joint configuration.
        q_end : array-like
            The end joint configuration.

    Returns
    -------
        float
            The distance between the two joint configurations.
    """
    return np.linalg.norm(q_end - q_start)


def get_path_length(q_path):
    """
    Returns the configuration distance of a path.

    Parameters
    ----------
        q_path : list[array-like]
            A list of joint configurations describing a path.

    Returns
    -------
        float
            The total configuration distance of the entire path.
    """
    total_distance = 0.0
    for idx in range(1, len(q_path)):
        total_distance += configuration_distance(q_path[idx - 1], q_path[idx])
    return total_distance


def get_random_state(model, padding=0.0):
    """
    Returns a random state that is within the model's joint limits.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        padding : float or array-like, optional
            The padding to use around the sampled joint limits.

    Returns
    -------
        array-like
            A set of randomly generated joint variables.
    """
    return np.random.uniform(
        model.lowerPositionLimit + padding, model.upperPositionLimit - padding
    )


def get_random_state_inside_ellipsoid(model, target_frame, q_start, start, goal, ik, c_best = None, max_tries=1000):
    """
    Returns a random state that is within the generated model's ellipsoid.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random transform.
        target_frame : str
            The name of the frame for which to generate a random transform inside the ellipsoid.
        q_start : array-like
            The current starting joint configuration.
        start : list
            The starting cartesian position of the joints.
        goal : list
            The ending cartesian position of the joints.
        ik : `pinocchio.IK`
            The inverse kinematics solver to use for generating the random state.
        c_best : float, optional
            The best cost found so far.
        max_tries : int, optional
            The maximum number of tries to generate a random state inside the ellipsoid.
    Returns
    -------
        array-like
            A set of randomly generated joint variables.
    """
    c_min = configuration_distance(start, goal)
    if c_best == np.inf:
        c_best = c_min*2.5
    normalized_direction_vector = (goal - start)/np.linalg.norm(goal - start)
    # Compute rotation matrix
    R = np.zeros((3, 3))
    R[:, 0] = normalized_direction_vector
    R[:, 1] = np.cross(normalized_direction_vector, [1, 0, 0])
    if np.linalg.norm(R[:, 1]) == 0:
        R[:, 1] = np.cross(normalized_direction_vector, [0, 1, 0])
    R[:, 1] /= np.linalg.norm(R[:, 1])
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    # Compute diagonal matrix
    L = np.diag([c_best / 2, c_best / 2, np.sqrt(c_best**2 - c_min**2) / 2])
    # Ellipsoid visualization
    ellipsoid_center = [(goal[0] + start[0]) / 2, (goal[1] + start[1]) / 2, (goal[2] + start[2]) / 2]
    x_radius = L[0, 0]
    y_radius = L[1, 1]
    z_radius = L[2, 2]
    ellipsoid_mesh = create_ellipsoid_mesh(x_radius, y_radius, z_radius)
    add_ellipsoid_model(model, ellipsoid_mesh, position=ellipsoid_center, rotation_matrix=R)
    q_sample = None
    tries = 0
    while q_sample is None:
        if tries == max_tries:
            return None
        # Cast a sample from a unit ball
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        unit_ball = np.array([x, y, z])
        if np.square(x) + np.square(y) + np.square(z) < 1:
            #Map ball sample to the ellipsoid
            sampled_point = np.dot(R, np.dot(L, unit_ball)) + ellipsoid_center
            # Use inverse kinematics to find a configuration that coincides with the sampled point
            target_tform = pinocchio.SE3.Identity()
            target_tform.translation = sampled_point
            q_sample = ik.solve(
                target_frame,
                target_tform,
                init_state=q_start,
                nullspace_components=[],
                verbose=False,
            )
        if q_sample is None:
            tries += 1
    print('Sampled random state in ellipsoid:', q_sample)
    return q_sample


def get_random_collision_free_state(model, collision_model, padding=0.0, max_tries=100):
    """
    Returns a random state that is within the model's joint limits and is collision-free according to the collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        padding : float or array-like, optional
            The padding to use around the sampled joint limits.
        max_tries : int, optional
            The maximum number of tries for sampling a collision-free state.

    Returns
    -------
        array-like
            A set of randomly generated collision-free joint variables, or None if one cannot be found.
    """
    num_tries = 0
    while num_tries < max_tries:
        state = get_random_state(model, padding=padding)
        if not check_collisions_at_state(model, collision_model, state):
            return state
        num_tries += 1

    print(f"Could not generate collision-free state after {max_tries} tries.")
    return None


def get_random_transform(model, target_frame, joint_padding=0.0):
    """
    Returns a random transform for a target frame that is within the model's joint limits.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random transform.
        target_frame : str
            The name of the frame for which to generate a random transform.
        joint_padding : float or array-like, optional
            The padding to use around the sampled joint limits.

    Returns
    -------
        `pinocchio.SE3`
            A randomly generated transform for the specified target frame.
    """
    q_target = get_random_state(model, joint_padding)
    data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_target)
    return data.oMf[target_frame_id]


def get_random_collision_free_transform(
    model, collision_model, target_frame, joint_padding=0.0, max_tries=100
):
    """
    Returns a random transform for a target frame that is within the model's joint limits and is collision-free.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random transform.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        target_frame : str
            The name of the frame for which to generate a random transform.
        joint_padding : float or array-like, optional
            The padding to use around the sampled joint limits.
        max_tries : int, optional
            The maximum number of tries for sampling a collision-free state.

    Returns
    -------
        `pinocchio.SE3`
            A randomly generated transform for the specified target frame.
    """
    q_target = get_random_collision_free_state(
        model, collision_model, joint_padding, max_tries
    )
    data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_target)
    return data.oMf[target_frame_id]


def check_within_limits(model, q):
    """
    Checks whether a particular joint configuration is within the model's joint limits.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        q : array-like
            The joint configuration for the model.

    Returns
    -------
        bool
            True if the configuration is within joint limits, otherwise False.
    """
    return np.all(q >= model.lowerPositionLimit) and np.all(
        q <= model.upperPositionLimit
    )


def extract_cartesian_pose(model, target_frame, q, data=None):
    """
    Extracts the Cartesian pose of a specified model frame given a joint configuration.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to perform forward kinematics.
        target_frame : str
            The name of the target frame.
        q : array-like
            The joint configuration values describing the robot state.
        data : `pinocchio.Data`, optional
            The model data to use. If not set, one will be created.

    Returns
    -------
        `pinocchio.SE3`
            The transform describing the Cartesian pose of the specified frame at the provided joint configuration.
    """
    if data is None:
        data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q)
    return copy.deepcopy(data.oMf[target_frame_id])


def extract_cartesian_poses(model, target_frame, q_vec, data=None):
    """
    Extracts the Cartesian poses of a specified model frame given a list of joint configurations.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to perform forward kinematics.
        target_frame : str
            The name of the target frame.
        q_vec : array-like
            A list of joint configuration values describing the path.
        data : `pinocchio.Data`, optional
            The model data to use. If not set, one will be created.

    Returns
    -------
        list[`pinocchio.SE3`]
            A list of transforms describing the Cartesian poses of the specified frame at the provided joint configurations.
    """
    if data is None:
        data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    tforms = []
    for q in q_vec:
        pinocchio.framesForwardKinematics(model, data, q)
        tforms.append(copy.deepcopy(data.oMf[target_frame_id]))
    return tforms


def get_translation_path_length(model, target_frame, q_vec):
    """
    Calculates the translation distance of a path in the given frame.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to perform forward kinematics.
        target_frame : str
            The name of the target frame.
        q_vec : array-like
            A list of joint configuration values describing the path.
    
    Returns
    -------
        float
            The total translation distance traveled by the path in the given frame.
    """
    translation_path = []
    for q in q_vec:
        translation_path.append(extract_cartesian_pose(model, target_frame, q).translation)
    translation_distance = 0
    for i in range(1, len(q_vec)):
        translation_distance += np.linalg.norm(translation_path[i] - translation_path[i-1])
    return translation_distance


def get_collision_geometry_ids(model, collision_model, body):
    """
    Gets a list of collision geometry model IDs for a specified body name.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for getting frame IDs.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        body : str
            The name of the body.
            This can be directly the name of a geometry in the collision model,
            or it can be the name of the frame in the main model.

    Return
    ------
        list[int]
            A list of collision geometry IDs corresponding to the body name.
    """
    body_collision_geom_ids = []

    # First, check if this is directly a collision geometry.
    body_collision_geom_id = collision_model.getGeometryId(body)
    if body_collision_geom_id < collision_model.ngeoms:
        body_collision_geom_ids.append(body_collision_geom_id)
    else:
        # Otherwise, look for the frame name in the model and return its associated collision objects.
        body_frame_id = model.getFrameId(body)
        if body_frame_id < model.nframes:
            for id, obj in enumerate(collision_model.geometryObjects):
                if obj.parentFrame == body_frame_id:
                    body_collision_geom_ids.append(id)

    return body_collision_geom_ids


def set_collisions(model, collision_model, body1, body2, enable):
    """
    Sets collision checking between two bodies by searching for their corresponding geometry objects in the collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for getting frame IDs.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        body1 : str
            The name of the first body.
        body2 : str
            The name of the second body.
        enable : bool
            If True, enables collisions. If False, disables collisions.
    """
    body1_collision_ids = get_collision_geometry_ids(model, collision_model, body1)
    body2_collision_ids = get_collision_geometry_ids(model, collision_model, body2)

    for id1 in body1_collision_ids:
        for id2 in body2_collision_ids:
            pair = pinocchio.CollisionPair(id1, id2)
            if enable:
                collision_model.addCollisionPair(pair)
            else:
                collision_model.removeCollisionPair(pair)


def calculate_collision_vector_and_jacobians(
    model, collision_model, data, collision_data, pair_idx, q
):
    """
    Given collision and distance results from collision model data, computes the collision vector
    and collision Jacobians at both collision points.

    This is useful for algorithms that perform collision avoidance, such as IK and trajectory optimization.

    Note that forward kinematics, collision, and distance checks must be evaluated first to populate the
    `data` and `collision_data` variables.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for Jacobian computation.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        data : `pinocchio.Data`
            The model data to use for Jacobian computation.
        collision_data : `pinocchio.GeometryData`
            The collision_model data to use for collision checking.
        pair_idx : int
            The index of the collision pair from which to extract information.
        q : array-like
            The joint configuration of the model.

    Return
    ------
        tuple(array-like, array-like, array-like)
            A tuple containing collision distance vector from frame1 to frame2,
            and the collision Jacobians at frame1 and frame2.
    """
    cp = collision_model.collisionPairs[pair_idx]
    cr = collision_data.collisionResults[pair_idx]
    dr = collision_data.distanceResults[pair_idx]

    if cr.isCollision():
        # According to the HPP-FCL documentation, the normal always points from object1 to object2.
        # TODO: Once HPP-FCL Python bindings expose the getNearestPointN() functions, use those instead.
        contact = cr.getContact(0)
        coll_points = [
            contact.pos,
            contact.pos - contact.normal * contact.penetration_depth,
        ]
    else:
        coll_points = [dr.getNearestPoint1(), dr.getNearestPoint2()]

    distance_vec = coll_points[1] - coll_points[0]

    # Calculate the Jacobians at the parent frames of both collision points.
    parent_frame1 = collision_model.geometryObjects[cp.first].parentFrame
    if parent_frame1 >= model.nframes:
        parent_frame1 = 0
    Jframe1 = pinocchio.computeFrameJacobian(
        model, data, q, parent_frame1, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    t_frame1_to_point1 = pinocchio.SE3(
        np.eye(3), coll_points[0] - data.oMf[parent_frame1].translation
    )
    Jcoll1 = t_frame1_to_point1.toActionMatrix()[3:, :] @ Jframe1

    parent_frame2 = collision_model.geometryObjects[cp.second].parentFrame
    if parent_frame2 >= model.nframes:
        parent_frame2 = 0
    Jframe2 = pinocchio.computeFrameJacobian(
        model, data, q, parent_frame2, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    t_frame2_to_point2 = pinocchio.SE3(
        np.eye(3), coll_points[1] - data.oMf[parent_frame2].translation
    )
    Jcoll2 = t_frame2_to_point2.toActionMatrix()[3:, :] @ Jframe2

    return distance_vec, Jcoll1, Jcoll2
