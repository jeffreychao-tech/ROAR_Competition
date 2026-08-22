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
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0

        num_wp = len(self.maneuverable_waypoints)
        self.target_speeds = np.zeros(num_wp)
        
        max_lat_accel = 12.5
        max_straight_speed = 60.0
        max_decel = 9.5

        for i in range(num_wp):
            wp_prev = self.maneuverable_waypoints[(i - 1) % num_wp].location[:2]
            wp_curr = self.maneuverable_waypoints[i].location[:2]
            wp_next = self.maneuverable_waypoints[(i + 1) % num_wp].location[:2]

            a = np.linalg.norm(wp_next - wp_curr)
            b = np.linalg.norm(wp_curr - wp_prev)
            c = np.linalg.norm(wp_next - wp_prev)
            
            area = 0.5 * abs((wp_curr[0] - wp_prev[0]) * (wp_next[1] - wp_prev[1]) - 
                            (wp_curr[1] - wp_prev[1]) * (wp_next[0] - wp_prev[0]))
            
            curvature = (4.0 * area) / (a * b * c + 1e-6)
            
            if curvature > 1e-4:
                corner_limit = np.sqrt(max_lat_accel / curvature)
            else:
                corner_limit = max_straight_speed
                
            self.target_speeds[i] = min(max_straight_speed, corner_limit)

        for i in range(num_wp - 1, -1, -1):
            curr_idx = i % num_wp
            next_idx = (i + 1) % num_wp

            wp_curr = self.maneuverable_waypoints[curr_idx].location[:2]
            wp_next = self.maneuverable_waypoints[next_idx].location[:2]
            dist = np.linalg.norm(wp_next - wp_curr)

            allowed_speed = np.sqrt(self.target_speeds[next_idx]**2 + 2 * max_decel * dist)
            self.target_speeds[curr_idx] = min(self.target_speeds[curr_idx], allowed_speed)


    async def step(
        self
    ) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = float(np.linalg.norm(vehicle_velocity))
        
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        target_speed = float(self.target_speeds[self.current_waypoint_idx])

        lookahead_dist = np.clip(0.3 * vehicle_velocity_norm + 3.0, 3.0, 20.0)
        
        accumulated_dist = 0.0
        target_wp_idx = self.current_waypoint_idx
        num_wp = len(self.maneuverable_waypoints)
        
        while accumulated_dist < lookahead_dist:
            next_idx = (target_wp_idx + 1) % num_wp
            p1 = self.maneuverable_waypoints[target_wp_idx].location[:2]
            p2 = self.maneuverable_waypoints[next_idx].location[:2]
            accumulated_dist += np.linalg.norm(p2 - p1)
            target_wp_idx = next_idx

        target_wp = self.maneuverable_waypoints[target_wp_idx]

        dx = target_wp.location[0] - vehicle_location[0]
        dy = target_wp.location[1] - vehicle_location[1]
        yaw = vehicle_rotation[2]

        local_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

        l2 = local_x**2 + local_y**2
        steer_angle = np.arctan2(2.0 * local_y, l2)
        steer_control = np.clip(-steer_angle / (np.pi / 4), -1.0, 1.0)

        Kp, Ki, Kd = 0.35, 0.005, 0.05
        dt = 0.05

        speed_error = target_speed - vehicle_velocity_norm
        self.speed_error_integral = np.clip(self.speed_error_integral + speed_error * dt, -5.0, 5.0)
        speed_derivative = (speed_error - self.prev_speed_error) / dt
        self.prev_speed_error = speed_error

        pid_output = (Kp * speed_error) + (Ki * self.speed_error_integral) + (Kd * speed_derivative)

        if pid_output >= 0:
            throttle = np.clip(pid_output + 0.1, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-pid_output, 0.0, 1.0)

        speed_kmh = vehicle_velocity_norm * 3.6
        if speed_kmh < 30:
            gear = 1
        elif speed_kmh < 60:
            gear = 2
        elif speed_kmh < 100:
            gear = 3
        elif speed_kmh < 140:
            gear = 4
        else:
            gear = 5

        control = {
            "throttle": float(throttle),
            "steer": float(steer_control),
            "brake": float(brake),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": gear
        }
        await self.vehicle.apply_action(control)
        return control