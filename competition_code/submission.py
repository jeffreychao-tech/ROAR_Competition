"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""
from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi
def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx
class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # Initialize PID speed controller variables
        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = float(np.linalg.norm(vehicle_velocity))
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # Dynamic look-ahead index scaling with speed (looks 4 to 16 waypoints ahead)
        lookahead_idx = int(np.clip(4 + vehicle_velocity_norm * 0.35, 4, 16))
        waypoint_to_follow = self.maneuverable_waypoints[
            (self.current_waypoint_idx + lookahead_idx) % len(self.maneuverable_waypoints)
        ]

        # Calculate vector and angle to look-ahead target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Cross-track error (distance to the closest track segment)
        closest_wp = self.maneuverable_waypoints[self.current_waypoint_idx]
        c_error_vec = (closest_wp.location - vehicle_location)[:2]
        cross_track_error = np.linalg.norm(c_error_vec)

        # Enhanced Steering Controller (Heading error + Speed-dependent cross-track correction)
        k_cte = 0.15
        k_speed = max(vehicle_velocity_norm, 1.0)
        steer_heading = delta_heading / (np.pi / 4)
        steer_cte = np.arctan2(k_cte * cross_track_error, k_speed)
        
        steer_control = -(steer_heading + steer_cte)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Target Speed Adjustment: Slow down slightly in sharp turns
        turn_severity = abs(delta_heading)
        base_target_speed = 35.0  # m/s
        target_speed = max(15.0, base_target_speed - turn_severity * 15.0)

        # Speed PID Controller Parameters
        Kp, Ki, Kd = 0.12, 0.001, 0.02
        dt = 0.05  # Approximate frame time step

        speed_error = target_speed - vehicle_velocity_norm
        self.speed_error_integral += speed_error * dt
        self.speed_error_integral = np.clip(self.speed_error_integral, -10.0, 10.0)  # Anti-windup
        speed_error_derivative = (speed_error - self.prev_speed_error) / dt
        self.prev_speed_error = speed_error

        pid_output = (Kp * speed_error) + (Ki * self.speed_error_integral) + (Kd * speed_error_derivative)

        # Throttle / Brake calculation
        if pid_output >= 0:
            throttle = np.clip(pid_output, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-pid_output, 0.0, 1.0)

        control = {
            "throttle": float(throttle),
            "steer": float(steer_control),
            "brake": float(brake),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control