""" Utilities for manipulation-specific Rapidly-Exploring Random Trees (RRTs). """

import numpy as np
import time

from ..core.utils import (
    check_collisions_at_state,
    check_collisions_along_path,
    configuration_distance,
    extract_cartesian_pose,
    extract_cartesian_poses,
    get_random_state,
    get_random_state_inside_ellipsoid,
    get_translation_path_length
)
from ..ik.differential_ik import DifferentialIk, DifferentialIkOptions
from ..visualization.meshcat_utils import visualize_frames, visualize_paths

from .graph import Node, Graph
from .utils import (
    discretize_joint_space_path,
    extend_robot_state,
    has_collision_free_path,
    retrace_path,
)


class RRTPlannerOptions:
    """Options for Rapidly-exploring Random Tree (RRT) planning."""

    def __init__(
        self,
        max_step_size=0.05,
        max_connection_dist=0.2,
        rrt_connect=False,
        bidirectional_rrt=False,
        rrt_star=False,
        informed_rrt_star=False,
        max_rewire_dist=np.inf,
        max_planning_time=10.0,
        rng_seed=None,
        fast_return=True,
        goal_biasing_probability=0.0,
        target_frame = "panda_hand",
        visual_model=None
    ):
        """
        Initializes a set of RRT planner options.

        Parameters
        ----------
            max_step_size : float
                Maximum joint configuration step size for collision checking along path segments.
            max_connection_dist : float
                Maximum angular distance, in radians, for connecting nodes.
            rrt_connect : bool
                If true, uses bidirectional RRTs from both start and goal nodes.
                Otherwise, only grows a tree from the start node.
            bidirectional_rrt : bool
                If true, enables the RRTConnect algorithm, which continues extending
                nodes towards a random node until an invalid state is reached.
            rrt_star : bool
                If true, enables the RRT* algorithm to shortcut node connections during planning.
                This in turn will use the `max_rewire_dist` parameter.
            informed_rrt_star : bool
                If true, enables the Informed RRT* algorithm, which uses an ellipsoidal heuristic to 
                bias sampling from the start node towards the goal node.
            max_rewire_dist : float
                Maximum angular distance, in radians, to consider rewiring nodes for RRT*.
                If set to `np.inf`, all nodes in the trees will be considered for rewiring.
            max_planning_time : float
                Maximum planning time, in seconds.
            rng_seed : int, optional
                Sets the seed for random number generation. Use to generate deterministic results.
            fast_return : bool
                If True, return as soon as a solution is found. Otherwise continuing building the tree
                until max_planning_time is reached.
            goal_biasing_probability : float
                Probability of sampling the goal configuration itself, which can help planning converge.
            target_frame : str
                The name of the target frame in the model.

        """
        self.max_step_size = max_step_size
        self.max_connection_dist = max_connection_dist
        self.rrt_connect = rrt_connect
        self.bidirectional_rrt = bidirectional_rrt
        self.rrt_star = rrt_star
        self.informed_rrt_star = informed_rrt_star
        self.max_rewire_dist = max_rewire_dist
        self.max_planning_time = max_planning_time
        self.rng_seed = rng_seed
        self.fast_return = fast_return
        self.goal_biasing_probability = goal_biasing_probability
        self.target_frame = target_frame
        self.visual_model = visual_model



class RRTPlanner:
    """Rapidly-expanding Random Tree (RRT) planner.

    This is a sampling-based motion planner that finds collision-free paths from a start to a goal configuration.

    Some good resources:
      * Original RRT paper: https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf
      * RRTConnect paper: https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
      * RRT* and PRM* paper: https://arxiv.org/abs/1105.1186
      * Informed RRT*: https://www.ri.cmu.edu/pub_files/2014/9/TR-2013-JDG003.pdf
    """

    def __init__(self, model, collision_model, options=RRTPlannerOptions()):
        """
        Creates an instance of an RRT planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
            options : `RRTPlannerOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.data = model.createData()
        self.collision_model = collision_model
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.options = options
        self.reset()

    def reset(self):
        """Resets all the planning data structures."""
        self.latest_path = None
        self.start_tree = Graph()
        self.goal_tree = Graph()
        np.random.seed(self.options.rng_seed)

    def plan(self, q_start, q_goal):
        """
        Plans a path from a start to a goal configuration.

        Parameters
        ----------
            q_start : array-like
                The starting robot configuration.
            q_goal : array-like
                The goal robot configuration.
        """
        self.reset()
        t_start = time.time()

        start_node = Node(q_start, parent=None, cost=0.0)
        self.start_tree.add_node(start_node)
        goal_node = Node(q_goal, parent=None, cost=0.0)
        self.goal_tree.add_node(goal_node)

        goal_found = False
        latest_start_tree_node = start_node
        latest_goal_tree_node = goal_node

        # Check start and end pose collisions.
        if check_collisions_at_state(
            self.model, self.collision_model, q_start, self.data, self.collision_data
        ):
            print("Start configuration in collision.")
            return None
        if check_collisions_at_state(
            self.model, self.collision_model, q_goal, self.data, self.collision_data
        ):
            print("Goal configuration in collision.")
            return None

        # Check direct connection to goal.
        path_to_goal = discretize_joint_space_path(
            [q_start, q_goal], self.options.max_step_size
        )
        if not check_collisions_along_path(
            self.model, self.collision_model, path_to_goal
        ):
            print("Start and goal can be directly connected!")
            goal_found = True
            self.latest_path = self.extract_path_from_trees(
                latest_start_tree_node, latest_goal_tree_node
            )
            return self.latest_path

        # Initialize inverse kinematics solver and start and goal node translations for informed rrt*
        if self.options.informed_rrt_star:
            ik = DifferentialIk(
                        model=self.model,
                        data=self.data,
                        collision_model=self.collision_model,
                        options=DifferentialIkOptions(
                            max_iters=300,
                            max_retries=20,
                            max_translation_error=0.05,
                            max_rotation_error=2*np.pi
                        )
            )
            start_trans = extract_cartesian_pose(self.model, self.options.target_frame, q_start).translation
            goal_trans = extract_cartesian_pose(self.model, self.options.target_frame, q_goal).translation
            best_trans_path_length = np.inf

        start_tree_phase = True
        while True:
            # Only return on success if specified in the options.
            if goal_found and self.options.fast_return:
                break

            # Check for timeouts.
            if time.time() - t_start > self.options.max_planning_time:
                message = "succeeded" if goal_found else "timed out"
                print(
                    f"Planning {message} after {self.options.max_planning_time} seconds."
                )
                break

            # Choose variables based on whether we're growing the start or goal tree.
            tree = self.start_tree if start_tree_phase else self.goal_tree
            other_tree = self.goal_tree if start_tree_phase else self.start_tree

            # Sample a new configuration.
            if np.random.random() < self.options.goal_biasing_probability:
                q_sample = q_goal if start_tree_phase else q_start
            else:
                if self.options.informed_rrt_star and goal_found:
                    q_sample = get_random_state_inside_ellipsoid(
                        self.options.visual_model,
                        self.options.target_frame,
                        q_start,
                        start_trans,
                        goal_trans,
                        ik,
                        c_best=best_trans_path_length,
                    )
                    if q_sample is None:
                        print('Failed to find a valid state in the current ellipsoid.')
                        continue
                else:
                    q_sample = get_random_state(self.model)

            # Run the extend or connect operation to connect the tree to the new node.
            nearest_node = tree.get_nearest_node(q_sample)
            q_new = self.extend_or_connect(nearest_node, q_sample)

            # Only if extend/connect succeeded, add the new node to the tree.
            if q_new is not None:
                new_node = self.add_node_to_tree(tree, q_new, nearest_node)
                if start_tree_phase:
                    latest_start_tree_node = new_node
                else:
                    latest_goal_tree_node = new_node

                # Check if latest node connects directly to the other tree.
                # If so, add it to the tree and mark planning as complete.
                nearest_node_in_other_tree = other_tree.get_nearest_node(new_node.q)
                path_to_other_tree = discretize_joint_space_path(
                    [new_node.q, nearest_node_in_other_tree.q],
                    self.options.max_step_size,
                )
                if not check_collisions_along_path(
                    self.model, self.collision_model, path_to_other_tree
                ):
                    new_node = self.add_node_to_tree(
                        tree, nearest_node_in_other_tree.q, new_node
                    )
                    if start_tree_phase:
                        latest_start_tree_node = new_node
                        latest_goal_tree_node = nearest_node_in_other_tree
                    else:
                        latest_start_tree_node = nearest_node_in_other_tree
                        latest_goal_tree_node = new_node
                    goal_found = True

                # Switch to the other tree next iteration, if bidirectional mode is enabled.
                if self.options.bidirectional_rrt:
                    start_tree_phase = not start_tree_phase

            # Back out the path by traversing the parents from the goal.
            self.latest_path = []
            if goal_found:
                self.latest_path = self.extract_path_from_trees(
                    latest_start_tree_node, latest_goal_tree_node
                )
                if self.options.informed_rrt_star:
                    latest_trans_path_length = get_translation_path_length(self.model, self.options.target_frame, self.latest_path)
                    if latest_trans_path_length < best_trans_path_length:
                        best_trans_path_length = latest_trans_path_length
        return self.latest_path

    def extend_or_connect(self, parent_node, q_sample):
        """
        Extends a tree towards a sampled node with steps no larger than the maximum connection distance.

        Parameters
        ----------
            parent_node : `pyroboplan.planning.graph.Node`
                The node from which to start extending or connecting towards the sample.
            q_sample : array-like
                The robot configuration sample to extend or connect towards.
        """
        # If they are the same node there's nothing to do.
        if np.array_equal(parent_node.q, q_sample):
            return None

        q_out = None
        q_cur = parent_node.q
        while True:
            # Compute the next incremental robot configuration.
            q_extend = extend_robot_state(
                q_cur,
                q_sample,
                self.options.max_connection_dist,
            )

            # If we can connect then it is a valid state
            if has_collision_free_path(
                q_cur,
                q_extend,
                self.options.max_step_size,
                self.model,
                self.collision_model,
            ):
                q_out = q_cur = q_extend
                # If we have reached the sampled state then we are done.
                if np.array_equal(q_cur, q_sample):
                    break
            else:
                break

            # If RRTConnect is disabled, only one iteration is needed.
            if not self.options.rrt_connect:
                break

        return q_out

    def extract_path_from_trees(self, start_tree_final_node, goal_tree_final_node):
        """
        Extracts the final path from the RRT trees.

        Parameters
        ----------
            start_tree_final_node : `pyroboplan.planning.graph.Node`
                The last node of the start tree.
            goal_tree_final_node : `pyroboplan.planning.graph.Node`, optional
                The last node of the goal tree.
                If None, this means the goal tree is ignored.

        Return
        ------
            list[array-like]
                A list of robot configurations describing the path waypoints in order.
        """
        path = retrace_path(start_tree_final_node)

        # If using a goal tree, then the final node should be at index 0, so we must reverse.
        if goal_tree_final_node:
            goal_tree_path = retrace_path(goal_tree_final_node)
            goal_tree_path.reverse()
            path += goal_tree_path

        # Convert to robot configuration states
        return [n.q for n in path]

    def add_node_to_tree(self, tree, q_new, parent_node):
        """
        Add a new node to the tree. If the RRT* algorithm is enabled, will also rewire.

        Parameters
        ----------
            tree : `pyroboplan.planning.graph.Graph`
                The tree to which to add the new node.
            q_new : array-like
                The robot configuration from which to create a new tree node.
            parent_node : `pyroboplan.planning.graph.Node`
                The parent node to connect the new node to.

        Returns
        -------
            `pyroboplan.planning.graph.Node`
                The new node that was added to the tree.
        """
        # Add the new node to the tree
        new_node = Node(q_new, parent=parent_node)
        tree.add_node(new_node)
        edge = tree.add_edge(parent_node, new_node)
        new_node.cost = parent_node.cost + edge.cost

        # If RRT* is enabled, rewire that node in the tree.
        if self.options.rrt_star or self.options.informed_rrt_star:
            min_cost = new_node.cost
            for other_node in tree.nodes:
                # Do not consider trivial nodes.
                if other_node == new_node or other_node == parent_node:
                    continue
                # Do not consider nodes farther than the configured rewire distance.
                new_distance = configuration_distance(other_node.q, q_new)
                if new_distance > self.options.max_rewire_dist:
                    continue
                # Rewire if this new connections would be of lower cost and is collision free.
                new_cost = other_node.cost + new_distance
                if new_cost < min_cost:
                    new_path = discretize_joint_space_path(
                        [q_new, other_node.q], self.options.max_step_size
                    )
                    if not check_collisions_along_path(
                        self.model, self.collision_model, new_path
                    ):
                        new_node.parent = other_node
                        new_node.cost = new_cost
                        tree.remove_edge(parent_node, new_node)
                        edge = tree.add_edge(other_node, new_node)
                        min_cost = new_cost
        return new_node

    def visualize(
        self,
        visualizer,
        frame_name,
        path_name="planned_path",
        tree_name="rrt",
        show_path=True,
        show_tree=False,
    ):
        """
        Visualizes the RRT path.

        Parameters
        ----------
            visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`, optional
                The visualizer to use for this solver.
            frame_name : str
                The name of the frame to use when visualizing paths in Cartesian space.
            path_name : str, optional
                The name of the MeshCat component for the path.
            tree_name : str, optional
                The name of the MeshCat component for the tree.
            show_path : bool, optional
                If true, shows the final path from start to goal.
            show_tree : bool, optional
                If true, shows the entire sampled tree.
        """
        visualizer.viewer[path_name].delete()
        if show_path:
            q_path = discretize_joint_space_path(
                self.latest_path, self.options.max_step_size
            )

            target_tforms = extract_cartesian_poses(self.model, frame_name, q_path)
            visualize_frames(
                visualizer, path_name, target_tforms, line_length=0.05, line_width=1.5
            )

        if show_tree:
            start_path_tforms = []
            for edge in self.start_tree.edges:
                q_path = discretize_joint_space_path(
                    [edge.nodeA.q, edge.nodeB.q], self.options.max_step_size
                )
                start_path_tforms.append(
                    extract_cartesian_poses(self.model, frame_name, q_path)
                )
            visualize_paths(
                visualizer,
                f"{tree_name}_start/edges",
                start_path_tforms,
                line_width=0.5,
                line_color=[0.9, 0.0, 0.9],
            )

            goal_path_tforms = []
            for edge in self.goal_tree.edges:
                q_path = discretize_joint_space_path(
                    [edge.nodeA.q, edge.nodeB.q], self.options.max_step_size
                )
                goal_path_tforms.append(
                    extract_cartesian_poses(self.model, frame_name, q_path)
                )
            visualize_paths(
                visualizer,
                f"{tree_name}_goal/edges",
                goal_path_tforms,
                line_width=0.5,
                line_color=[0.0, 0.9, 0.9],
            )
