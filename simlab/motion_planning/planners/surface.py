"""Fixed-height SE(2) planning through SimLab's existing OMPL/world adapters."""

import math

import numpy as np
from ompl import base as ob, geometric as og
from scipy.spatial.transform import Rotation as R

from simlab.motion_planning.result import MotionPlanKind, MotionPlanResult


def plan_surface_path(
    planner,
    *,
    start_xyz,
    start_quat_wxyz,
    goal_xyz,
    goal_quat_wxyz,
    time_limit,
    planner_type,
    dynamic_obstacle_prediction_speed=0.0,
    spacing_m=0.20,
    max_points=500,
):
    """Search in x/y/yaw; every collision query and output uses the start height.

    ``planner`` is the existing OmplPlanner, providing its world, bounds,
    safety margin, objective, and dynamic-obstacle timing convention.
    Returned arrays follow the same XYZ/quaternion contract as Dory paths.
    """
    start_xyz = np.asarray(start_xyz, dtype=float).reshape(3)
    goal_xyz = np.asarray(goal_xyz, dtype=float).reshape(3)
    start_quat = np.asarray(start_quat_wxyz, dtype=float).reshape(4)
    goal_quat = np.asarray(goal_quat_wxyz, dtype=float).reshape(4)
    if not all(np.all(np.isfinite(a)) for a in (start_xyz, goal_xyz, start_quat, goal_quat)):
        raise ValueError("Surface planning inputs must be finite.")
    if not math.isfinite(time_limit) or time_limit <= 0.0:
        raise ValueError("Surface planning time_limit must be finite and positive.")
    speed = float(dynamic_obstacle_prediction_speed)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("Dynamic-obstacle prediction speed must be finite and nonnegative.")
    if spacing_m <= 0.0 or max_points < 2:
        raise ValueError("Surface path spacing/budget must be positive and allow two states.")

    height = float(start_xyz[2])
    if not np.isclose(goal_xyz[2], height, atol=1e-8, rtol=0.0):
        raise ValueError("Surface goal must use the start height.")
    x_min, x_max, y_min, y_max, z_min, z_max = planner.env_bounds
    if not z_min - 1e-8 <= height <= z_max + 1e-8:
        raise ValueError("Surface height is outside the environment bounds.")
    start_yaw = R.from_quat(start_quat[[1, 2, 3, 0]]).as_euler("xyz")[2]
    goal_yaw = R.from_quat(goal_quat[[1, 2, 3, 0]]).as_euler("xyz")[2]
    world = planner._resolve_planner_world(planner.rclpy_node)

    def valid(state):
        x, y = state.getX(), state.getY()
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False
        xyz = np.array([x, y, height], dtype=float)
        is_start = np.linalg.norm(xyz - start_xyz) <= 1e-6
        margin = 0.0 if is_start else max(0.0, float(planner.safety_margin))
        offset = 0.0 if is_start else planner._dynamic_obstacle_time_offset(xyz, start_xyz, speed)
        return world.is_state_valid_xyz(xyz, safety_margin=margin, t_offset=offset)

    def failed(message):
        return MotionPlanResult(is_success=False, kind=MotionPlanKind.PATH, message=message)

    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, x_min)
    bounds.setHigh(0, x_max)
    bounds.setLow(1, y_min)
    bounds.setHigh(1, y_max)
    space.setBounds(bounds)
    setup = og.SimpleSetup(space)
    setup.setStateValidityChecker(ob.StateValidityCheckerFn(valid))
    start, goal = ob.State(space), ob.State(space)
    for state, xyz, yaw in ((start, start_xyz, start_yaw), (goal, goal_xyz, goal_yaw)):
        state().setX(float(xyz[0]))
        state().setY(float(xyz[1]))
        state().setYaw(float(yaw))
    if not valid(start()) or not valid(goal()):
        return failed("Surface start or goal is outside bounds or in collision.")
    setup.setStartAndGoalStates(start, goal)
    si = setup.getSpaceInformation()
    resolution = min(0.01, 0.01 / max(space.getMaximumExtent(), 1e-12))
    space.setLongestValidSegmentFraction(resolution)
    si.setStateValidityCheckingResolution(resolution)
    objective = planner._get_path_length_objective(si)
    setup.setOptimizationObjective(objective)
    algorithm = {"Bitstar": og.BITstar, "RRTstar": og.RRTstar, "RRTConnect": og.RRTConnect}.get(planner_type)
    if algorithm is None:
        return failed(f"Unsupported surface planner: {planner_type}")
    setup.setPlanner(algorithm(si))
    if not setup.solve(float(time_limit)) or not setup.haveExactSolutionPath():
        return failed("Surface planner did not find an exact path.")
    try:
        setup.simplifySolution(max(0.02, min(0.15, 0.15 * float(time_limit))))
    except TypeError:
        setup.simplifySolution()
    except Exception as exc:
        planner.rclpy_node.get_logger().warn(f"Surface path simplification skipped: {exc}")

    path = setup.getSolutionPath()
    if path.getStateCount() > max_points:
        return failed("Surface path exceeds the trajectory waypoint budget.")
    count = max(path.getStateCount(), min(max_points, max(2, math.ceil(path.length() / spacing_m) + 1)))
    if not path.check():
        return failed("Surface path failed its final collision check.")
    # Native interpolation retains the checked corners and geometric path.
    path.interpolate(int(count))
    states = [path.getState(i) for i in range(path.getStateCount())]
    xyz = np.array([[s.getX(), s.getY(), height] for s in states], dtype=float)
    yaw = np.array([s.getYaw() for s in states], dtype=float)
    quat = np.column_stack((np.cos(yaw / 2.0), np.zeros(yaw.size), np.zeros(yaw.size), np.sin(yaw / 2.0)))
    if np.max(np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1)) <= 1e-9:
        # A yaw-only goal needs both endpoint orientations, without duplicate
        # intermediate Cartesian waypoints in the trajectory generator.
        xyz, quat = xyz[[0, -1]], quat[[0, -1]]
    return MotionPlanResult(
        is_success=True,
        kind=MotionPlanKind.PATH,
        xyz=xyz,
        quat_wxyz=quat,
        path_length_cost=float(path.cost(objective).value()),
        geom_length=float(path.length()),
        message=f"Surface planner found {xyz.shape[0]} waypoints at z={height:.3f} m.",
    )
