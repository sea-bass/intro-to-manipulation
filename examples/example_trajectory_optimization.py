"""
This example shows PyRoboPlan capabilities for path planning using
trajectory optimization.
"""

import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import time

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.trajectory_optimization import CubicTrajectoryOptimization
from pyroboplan.visualization.meshcat_utils import visualize_frames


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    # add_object_collisions(model, collision_model, visual_model)

    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Define the start and end configurations
    q_start = get_random_collision_free_state(model, collision_model)
    q_end = get_random_collision_free_state(model, collision_model)
    viz.display(q_start)
    time.sleep(1.0)

    # Search for a trajectory
    dt = 0.025
    planner = CubicTrajectoryOptimization(model, collision_model)
    traj = planner.plan(q_start, q_end)

    if traj is not None:
        t_vec, q_vec, qd_vec, qdd_vec = traj.generate(dt)

        # Display the trajectory and points along the path.
        plt.ion()
        traj.visualize(dt=dt, joint_names=model.names[1:])
        time.sleep(0.5)

        tforms = extract_cartesian_poses(model, "panda_hand", q_vec.T)
        viz.display(q_start)
        visualize_frames(viz, "waypoints", tforms, line_length=0.075, line_width=2)
        time.sleep(1.0)

        # Animate the generated trajectory
        input("Press 'Enter' to animate the path.")
        for idx in range(q_vec.shape[1]):
            viz.display(q_vec[:, idx])
            time.sleep(dt)