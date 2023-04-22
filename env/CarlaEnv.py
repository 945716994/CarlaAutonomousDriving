import random
import time

from sensors import *
import numpy as np
import carla
# import gymnasium as gym


class CarlaEnv:
    def __init__(self, host, port, param=None):
        # connection simulator
        self.client = carla.Client(host, port)
        # Carla env model(world model), can control most things in env, like weather,vehicles,etc
        self.world = self.client.get_world()
        # blueprint:contains all information about env, like vehicle, human, etc
        self.blueprint_library = self.world.get_blueprint_library()
        # env setting
        self.setting = self.world.get_settings()
        self.town = param['town']
        # map
        self.map = self.world.get_map()
        # recorder: recorder everything to a log file, and replay it if we want
        self.recorder = self.client.start_recorder(param['log_path'])
        # checkPoint
        self.checkpoint_frequency = param['checkpoint_freq']
        # spawn points: can generate vehicle location
        self.spawn_points = self.map.get_spawn_points()
        self.vehicle = None
        self.continuous_action_space = param['continuous_action']
        self.display_on = param['VISUAL_DISPLAY']
        # whether rollback
        self.fresh_start = True
        self.param = param
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0

        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lan_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()

        self.walker_list = list()

    def step(self, action):
        try:
            self.timesteps += 1
            self.fresh_start = False

            # velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            # Todo:why multi 3.6?
            self.velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

            # Action
            if self.continuous_action_space:
                # 是增量
                steer = float(action[0])
                steer = max(min(steer, 1), -1)
                throttle = float(action[1])
                throttle = max(min(throttle, 1), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1,
                                                                throttle=self.throttle * 0.9 + throttle * 0.1))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(
                        carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1))
                self.previous_steer = steer
                self.throttle = 1.0

            # Traffic Light state change light state to green
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)
            self.collision_history = self.collision_obj.collision_data

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # location of the car
            self.location = self.vehicle.get_location()

            # keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                             self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            # 当前经过的航点
            self.current_waypoint_index = waypoint_index
            # calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),
                                                              self.vector(self.next_waypoint.transform.location),
                                                              self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # get angle difference between closest waypoint and vehicle forward vector
            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd)

            # update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (
                                                             self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            # rewards are given below
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                done = True
                reward = -10
            elif self.velocity > self.max_speed:
                done = True
                reward = -10

            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continuous_action_space:
                    if self.velocity < self.min_speed:
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor
                    elif self.velocity > self.target_speed:
                        reward = (1.0 - (self.velocity - self.target_speed) / (self.max_speed - self.min_speed))
                    else:
                        reward = 1.0 * centering_factor * angle_factor
                else:
                    reward = 1.0 * centering_factor * angle_factor

            if self.timesteps >= 7500:
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance // 2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.0001)

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity / self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity,
                                            normalized_distance_from_center, normalized_angle])
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                for sensor in self.sensor_list:
                    sensor.destroy()

                self.remove_sensors()

                for actor in self.actor_list:
                    actor.destroy()

            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered,
                                                                         self.center_lane_deviation]

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def reset(self):
        try:
            # destory last episode information
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()

            # vehicle blueprint
            vehicle_bp = self.get_vehicle(self.param['vehicle_name'])

            # random pick a spawn point
            transform = random.choice(self.map.get_spawn_points())
            self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)

            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            # make sure camera complete initialization
            while (len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            self.timesteps = 0
            # yaw: Z-axis rotation angle.
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = self.param['target_speed']
            self.max_speed = self.param['max_speed']
            self.min_speed = self.param['min_speed']
            self.max_distance_from_center = 3
            # 油门
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.distance_covered = float(0.0)

            if self.fresh_start:
                self.current_waypoint_index = 0
                # waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True,
                                                      lane_type=carla.LaneType.Driving)
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.navigation_obs = np.array(
                [self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])

            time.sleep(0.5)

            self.collision_history.clear()
            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]

        except:
            print("reset has exception!")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def render(self):
        pass

    def close(self):
        pass

    def create_pedestrians(self, numbers):
        '''
        Creating and Spawning Pedestrians in our world
        '''
        try:
            # getting the available spawn points in our world.
            walker_spawn_points = []
            for i in range(numbers):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)
            # we spawn the walker actor and ai controller also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
                # walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute('speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            # start the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])

    def set_other_vehicles(self, numbers):
        """Creating and Spawning other vehciles in our world"""
        try:
            for _ in range(0, numbers):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicle have been generated in autopilot mode.")
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def change_town(self, new_town):
        """setting for changing the town on the server"""
        self.world = self.client.load_world(new_town)

    def get_world(self) -> object:
        return self.world

    def get_blueprint_library(self):
        """ Getter for fetching blueprint library of the simulator."""
        return self.world.get_blueprint_library()

    def angle_diff(self, v0, v1):
        '''
        计算两个向量（张量）的角度差
        :param v0:
        :param v1:
        :return:
        '''
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        '''
        return a vector form about location, rotation,vector3D
        :param v:
        '''
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def get_vehicle(self, vehicle_name):
        '''
        pick a random color for the vehicle everytime
        :param vehicle_name: vehicle type
        '''
        vehicle_bp = self.blueprint_library.filter(vehicle_name)[0]
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)
        return vehicle_bp

    def set_vehicle(self, vehicle_bp, spawn_point):
        '''
        According to the spawn points set the vehicle in the env
        :param vehicle_bp:
        :param spawn_points:
        '''
        result = None
        while result is None:
            result = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        self.vehicle = result

    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None


if __name__ == '__main__':
    param = {
        'town':'Town01',
        'log_path':'%6d.png',
        'checkpoint_freq':100,
        'continuous_action':True,
        'VISUAL_DISPLAY':True,
        'vehicle_name': 'model3',
        'target_speed': 22,
        'max_speed':25,
        'min_speed':15
    }
    env = CarlaEnv('localhost', 2000, param)
    obs = env.reset()
    env.step()