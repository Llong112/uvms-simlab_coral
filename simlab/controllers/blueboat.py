"""Planar surge/heading feedback through SimLab's existing controller interface.

State and references use the internal map-NED/body conventions of Robot.
The output is [Fx, Fy, Fz, Tx, Ty, Tz]; only Fx and Tz are commanded.
Gains and limits are provisional demonstration settings, not calibrated data.
The C++ allocator remains responsible for individual thruster limits.
"""

import casadi as ca
import numpy as np

from simlab.controllers.base import ControllerTemplate


class BlueBoatSurfaceController(ControllerTemplate):
    name = "BlueBoat surge and heading"
    registry_name = "Surface"
    supported_robot_types = ("blueboat",)

    def __init__(self, node, arm_dof=0, robot_prefix=""):
        super().__init__(node, arm_dof, robot_prefix)
        defaults = {
            "position_gain": 0.5,
            "speed_gain": 20.0,
            "heading_gain": 4.0,
            "yaw_damping": 2.0,
            "max_speed": 0.4,
            "max_force": 20.0,
            "max_yaw_moment": 5.9,
            "position_tolerance": 0.1,
        }
        values = []
        for name, default in defaults.items():
            key = f"{self.robot_prefix}surface.{name}"
            if not self.node.has_parameter(key):
                self.node.declare_parameter(key, default)
            value = float(self.node.get_parameter(key).value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be finite and positive")
            values.append(value)

        self.gains = ca.DM(values)
        self.control_function = self._build_control_function()

    @staticmethod
    def _build_control_function():
        state = ca.SX.sym("state", 12)
        target_pose = ca.SX.sym("target_pose", 6)
        target_velocity = ca.SX.sym("target_body_velocity", 6)
        gains = ca.SX.sym("gains", 8)
        kp, ku, kpsi, kr = gains[0], gains[1], gains[2], gains[3]
        max_speed, max_force, max_yaw, tolerance = gains[4:8].elements()

        error_xy = target_pose[:2] - state[:2]
        distance = ca.sqrt(ca.dot(error_xy, error_xy))
        yaw_ref = target_pose[5]
        c, s = ca.cos(yaw_ref), ca.sin(yaw_ref)

        # Robot supplies desired velocities in the desired body's frame.
        # Rotate that reference into map-NED before adding position feedback.
        reference_xy = ca.vertcat(
            c * target_velocity[0] - s * target_velocity[1],
            s * target_velocity[0] + c * target_velocity[1],
        )
        course_xy = reference_xy + kp * error_xy
        course_speed = ca.sqrt(ca.dot(course_xy, course_xy))
        reference_speed = ca.sqrt(ca.dot(reference_xy, reference_xy))

        settled_reference = ca.logic_and(distance <= tolerance, reference_speed <= 1e-6)
        desired_heading = ca.if_else(
            settled_reference, yaw_ref, ca.atan2(course_xy[1], course_xy[0])
        )
        heading_error = ca.atan2(
            ca.sin(desired_heading - state[5]),
            ca.cos(desired_heading - state[5]),
        )

        # Turn toward the course before requesting substantial forward speed.
        speed_ref = ca.if_else(
            settled_reference,
            0.0,
            ca.fmin(course_speed, max_speed) * ca.fmax(0.0, ca.cos(heading_error)),
        )
        surge_force = ku * (speed_ref - state[6])
        yaw_moment = kpsi * heading_error + kr * (target_velocity[5] - state[11])

        wrench = ca.SX.zeros(6, 1)
        wrench[0] = ca.fmin(max_force, ca.fmax(-max_force, surge_force))
        wrench[5] = ca.fmin(max_yaw, ca.fmax(-max_yaw, yaw_moment))
        return ca.Function(
            "BlueBoatSurfaceControl", [state, target_pose, target_velocity, gains], [wrench]
        )

    def vehicle_controller(self, state, target_pos, target_vel, target_acc, dt):
        state = self.vector(state, 12, "state")
        target_pos = self.vector(target_pos, 6, "target_pos")
        target_vel = self.vector(target_vel, 6, "target_vel")
        target_acc = self.vector(target_acc, 6, "target_acc")
        if (
            not np.isfinite(dt)
            or dt <= 0.0
            or not all(np.all(np.isfinite(a)) for a in (state, target_pos, target_vel, target_acc))
        ):
            return np.zeros(6, dtype=float)

        command = self.control_function(state, target_pos, target_vel, self.gains)
        command = np.asarray(command, dtype=float).reshape(6)
        return command if np.all(np.isfinite(command)) else np.zeros(6, dtype=float)

    def arm_controller(self, q, q_dot, q_ref, dq_ref, ddq_ref, dt):
        # Compatibility with the existing five-effort transport. BlueBoat's
        # vehicle-only control path will skip arm control and publication.
        return np.zeros(5, dtype=float)
