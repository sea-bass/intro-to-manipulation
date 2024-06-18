"""
This example shows PyRoboPlan capabilities for path planning using
Informed Rapidly-Exploring Random Tree (Informed RRT) algorithm.
"""

from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time

from pyroboplan.core.utils import (
    extract_cartesian_poses,
    get_random_collision_free_state,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)

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

    # Search for a path
    options = RRTPlannerOptions(
        max_step_size=0.05,
        max_connection_dist=np.inf,
        rrt_connect=False,
        bidirectional_rrt=False,
        rrt_star=False,
        informed_rrt_star=True,
        max_rewire_dist=np.inf,
        max_planning_time=45,
        fast_return=False,
        goal_biasing_probability=0.15,
        visual_model=viz,
    )

    planner = RRTPlanner(model, collision_model, options=options)
    path = planner.plan(q_start, q_end)
    planner.visualize(viz, "panda_hand", show_tree=True)

    # Animate the path
    if path:
        # Optionally shortcut the path
        do_shortcutting = False
        if do_shortcutting:
            path = shortcut_path(model, collision_model, path)

        discretized_path = discretize_joint_space_path(path, options.max_step_size)

        if do_shortcutting:
            target_tforms = extract_cartesian_poses(
                model, "panda_hand", discretized_path
            )
            visualize_frames(
                viz, "shortened_path", target_tforms, line_length=0.05, line_width=1.5
            )

        input("Press 'Enter' to animate the path.")
        for q in discretized_path:
            viz.display(q)
            time.sleep(0.05)
