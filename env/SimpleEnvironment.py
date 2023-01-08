from gym import Env
from gym import logger
from gym.spaces import Box
import numpy as np
from typing import Optional
from gym.error import DependencyNotInstalled

# Assume the robot cannot go backwards and does not stop moving (have max speed of 0.2 in assignments)
# Was using 0.8
MAX_SPEED = 2.0
# TODO: Change from 0.05
MIN_SPEED = 0.0
MAX_W = np.pi /4 # TODO: USE?

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Permissable distance from goal
GOAL_DISTANCE = 0.01
GOAL_ANGLE = 0.01
MAX_SENSOR_DISTANCE = 0.12
# When to start factoring in angle difference to the reward
# Checked with 3.0 and 1.0 and 2.0 worked best
GOAL_REWARD_DISTANCE = 0.2
ROBOT_RADIUS = 0.105 / 2

# Robot and obstacle initilization constants
NUM_OBSTACLES = 0
INIT_DISTANCE_FROM_GOAL = 0.7

class SimpleRobotEnviroment(Env):

    # For rendering
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # define your environment
        # action space, observation space

        # Set the grid size that we're operating in, use continuous gridspace
        self.grid_size = 2.0

        # List of obstacles in the environment
        self.obstacles = []
        for i in range(NUM_OBSTACLES):
            # Set now and randomly initialise later in code
            self.obstacles.append(Obstacle(0.6, 0.85, 0.1))

        # TODO: randomly initiliase, ensure it does not clash with obstacles and is reachable (i.e. the robot is still fully in the grid if it reaches the space)
        # Goal position, set now and will randomly initiliase and ensure that it does not clash with any obstacles
        self.goal_position = np.array([1.0,1.0,np.pi])

        # Robot start position, set now and randomly initiliase and ensure that it does not clash with any obstacles
        self.num_sensors = 7
        self.sensor_angles = np.linspace(-np.pi/2, np.pi/2, self.num_sensors)
        # Robot position needs to be a float
        self.robot = SimpleRobot(np.array([0.7,0.7,0.0]), ROBOT_RADIUS)

        # Randomly initialise goal, robot and obstacle positions
        self.reset_positions()

        # Define observation space, [current position, goal position, sensor readings (check how many from assignment/Prorok code) (paper uses 30)]
        observation_shape = self.num_sensors + 6
        obs_min = np.full((observation_shape,), 0.0)
        # min x position robot, min y position robot, min yaw robot, need to account for observing values when we have moved outside the grid
        obs_min[0], obs_min[1], obs_min[2] = 0.0, 0.0, -np.pi
        # min x position goal, min y position goal, min yaw goal
        obs_min[3], obs_min[4], obs_min[5] = 0.0 + self.robot.radius, 0.0 + self.robot.radius, -np.pi

        obs_max = np.full((observation_shape,), MAX_SENSOR_DISTANCE)
        # max x position robot, max y position robot, max yaw robot, need to account for observing values when we move outside the grid
        obs_max[0], obs_max[1], obs_max[2] = self.grid_size, self.grid_size, np.pi
        # max x position goal, max y position goal, max yaw goal
        obs_max[3], obs_max[4], obs_max[5] = self.grid_size - self.robot.radius, self.grid_size - self.robot.radius, np.pi
        self.observation_space = Box(low = obs_min, high = obs_max, shape=(observation_shape,), dtype = np.float32)

        # Define action spac, [forward velocity, angular velocity] 
        # CHANGE: MAX_W?
        self.action_space = Box(low=np.array([MIN_SPEED, -np.pi]), high=np.array([MAX_SPEED, np.pi]), shape=(2,), dtype=np.float32)

        # For rendering
        self.path = [np.copy(self.robot.pose[:2])]
        self.steps_beyond_terminated = None
        self.render_mode = render_mode

        self.scaled_offset = 30
        self.screen_width = 600 + self.scaled_offset
        self.screen_height = 600 + self.scaled_offset
        self.screen = None
        self.clock = None
        self.scale = (self.screen_width - self.scaled_offset) / self.grid_size
        # divide by two to spread spacing amongst all sides
        self.offset = (self.scaled_offset / self.scale)/2
  
    def step(self, action):
        if self.steps_beyond_terminated is not None:
            logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )

        # Take action 
        # action = [u,w]
        u, w = action[0], action[1]
        self.robot.update_pose(u,w,self.grid_size)
        self.path.append(np.copy(self.robot.pose[:2]))

        # Compute observations
        # [robot position x, robot position y, robot position YAW, goal position x, goal position y, goal position YAW, sensor readings ...]
        observation = self.observation()

        # Compute reward
        robot_x_y = self.robot.pose[:2]
        goal_position_x_y = self.goal_position[:2]
        # Distance to goal
        distance = np.linalg.norm(goal_position_x_y - robot_x_y)
        # Difference between current angle and goal angle, smallest angle either anti-clockwise or clockwise
        angle_diff = min(np.abs(self.goal_position[YAW] - self.robot.pose[YAW]), 2*np.pi - np.abs(self.goal_position[YAW] - self.robot.pose[YAW]))
        # Scale the outputs so the angle difference doesn't overwhelm the reward function, distance and angle contribute the same amount to the reward function
        max_distance = np.linalg.norm(np.array([self.grid_size, self.grid_size]))
        # print("Dist ", distance, " angle_diff: ", angle_diff)
        # reward = - (distance/(max_distance) + angle_diff/(2*np.pi))*self.grid_size
        normalized_dist = distance/max_distance
        # reward_proportion = np.tanh(normalized_dist)/GOAL_REWARD_DISTANCE
        reward_proportion = 0.5
        # reward = - normalized_dist if distance >= GOAL_REWARD_DISTANCE else (-0.9 * (normalized_dist*reward_proportion + (angle_diff/(np.pi))*(1-reward_proportion)))
        norm_goal_reward_distance = GOAL_REWARD_DISTANCE/max_distance
        # reward = - normalized_dist if distance > GOAL_REWARD_DISTANCE else \
        #     (- (normalized_dist*reward_proportion + (angle_diff/(np.pi))*norm_goal_reward_distance*(1-reward_proportion)))
        reward = - normalized_dist
        # print("Angle diff ", angle_diff, " angle robot ", self.goal_position[YAW], " angle goal ", self.robot.pose[YAW])
        # reward = -distance if distance >= GOAL_REWARD_DISTANCE else (-distance+np.pi-angle_diff)
        # reward = -distance-(angle_diff/np.pi)
        # if reward > 10:
        #     print("Angle diff: ", angle_diff)
        #     print("Distance: ", distance)

        # Record dictionary
        info_dict = {}
        info_dict["Success"] = 0
        info_dict["Crash"] = 0

        # Compute done
        done = False
        # Large negative reward to ensure agent doesn't just learn to crash into the wall
        collision_reward = -1200
        # if robot is outside the grid (collides with a wall)
        if (self.robot.pose[X] + self.robot.radius >= self.grid_size) or (self.robot.pose[Y] + self.robot.radius >= self.grid_size) \
            or (self.robot.pose[X] <= 0.0 + self.robot.radius) or (self.robot.pose[Y] <= 0.0 + self.robot.radius):
            done = True
            # TODO: 50
            reward += collision_reward
            info_dict["Crash"] = 1
        else:
            # if the robot collides with an object
            collision = np.any(np.array([o.collide(self.robot) for o in self.obstacles]))
            if collision:
                done = True
                # TODO 50
                reward += collision_reward
                info_dict["Crash"] = 1
            # if the robot reaches the goal (is within some distance of the goal position)
            # elif distance <= GOAL_DISTANCE and angle_diff <= GOAL_ANGLE:
            elif distance <= GOAL_DISTANCE:
                done = True
                reward += 1400
                info_dict["Success"] = 1
        # Allow us to throw warning and stop unexpected behaviour
        if done:
            self.steps_beyond_terminated = 0


        # Render the environment
        if self.render_mode == "human":
            self.render()

        # if reward > 10:
        #     print("Big reward: ", reward)
        return np.array(observation), reward, done, info_dict

    def reset(self):
        # reset your environment

        # Reset goal, robot and obstacle positions
        self.reset_positions()
        # self.robot.set_pose(np.array([0.7,0.7,0.0]))

        # Random bits and bobs
        self.steps_beyond_terminated = None
        self.screen = None
        self.clock = None
        self.path = [np.copy(self.robot.pose[:2])]

        # print("ROBOT: ", self.robot.pose)
        # print("GOAL: ", self.goal_position)
        return self.observation()

    def render(self):
        # render your environment (can be a visualisation/print)
        # Use rendering format of example Gym environments
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # define function to get point on circle on line to visualize angles
        def get_point_on_circle(x,y,r,angle):
            x_circle = x + r*np.cos(angle)
            y_circle = y + r*np.sin(angle)
            return x_circle, y_circle

        # draw the box
        max_size_scale_offset = int((self.grid_size + self.offset)*self.scale)
        min_size_scale_offset = int(0 + self.offset*self.scale) 
        gfxdraw.hline(self.surf, min_size_scale_offset, max_size_scale_offset, min_size_scale_offset, (0,0,0))
        gfxdraw.hline(self.surf, min_size_scale_offset, max_size_scale_offset, max_size_scale_offset, (0,0,0))
        gfxdraw.vline(self.surf, min_size_scale_offset, min_size_scale_offset, max_size_scale_offset, (0,0,0))
        gfxdraw.vline(self.surf, max_size_scale_offset, min_size_scale_offset, max_size_scale_offset, (0,0,0))
        

        # draw goal
        goal_x = int((self.goal_position[X]+ self.offset)*self.scale)
        goal_y = int((self.goal_position[Y] + self.offset)*self.scale)
        goal_r = int(self.robot.radius*self.scale)
        # may need to cast value to int
        gfxdraw.aacircle(self.surf, goal_x, goal_y, goal_r, (52, 175, 61),)
        # plot orientation at goal
        x_goal_circle, y_goal_circle = get_point_on_circle(self.goal_position[X], self.goal_position[Y], self.robot.radius, self.goal_position[YAW])
        x_goal_circle = int((x_goal_circle + self.offset)*self.scale)
        y_goal_circle = int((y_goal_circle + self.offset)*self.scale)
        gfxdraw.line(self.surf, goal_x, goal_y, x_goal_circle, y_goal_circle, (0,0,0))

        # draw robot
        robot_x = int((self.robot.pose[X] + self.offset)*self.scale)
        robot_y = int((self.robot.pose[Y] + self.offset)*self.scale)
        robot_r = int(self.robot.radius*self.scale)
        # may need to cast value to int
        gfxdraw.aacircle(self.surf, robot_x, robot_y, robot_r, (159, 197, 236),)
        gfxdraw.filled_circle(self.surf, robot_x, robot_y, robot_r, (159, 197, 236),)
        # plot orientation of robot
        x_robot_circle, y_robot_circle = get_point_on_circle(self.robot.pose[X], self.robot.pose[Y], self.robot.radius, self.robot.pose[YAW])
        x_robot_circle = int((x_robot_circle + self.offset)*self.scale)
        y_robot_circle = int((y_robot_circle + self.offset)*self.scale)
        gfxdraw.line(self.surf, robot_x, robot_y, x_robot_circle, y_robot_circle, (0,0,0))

        #draw obstacles
        for o in self.obstacles:
            o_x = int((o.x_coord + self.offset)*self.scale)
            o_y = int((o.y_coord + self.offset)*self.scale)
            o_r = int(o.radius*self.scale)
            gfxdraw.aacircle(self.surf, o_x, o_y, o_r, (0, 102, 204),)
            gfxdraw.filled_circle(self.surf, o_x, o_y, o_r, (0, 102, 204),)

        # draw path
        for i in range(1,len(self.path)):
            c_pos_x = int((self.path[i][X] + self.offset)*self.scale)
            c_pos_y = int((self.path[i][Y] + self.offset)*self.scale)
            p_pos_x = int((self.path[i-1][X] + self.offset)*self.scale)
            p_pos_y = int((self.path[i-1][Y] + self.offset)*self.scale)
            gfxdraw.line(self.surf, p_pos_x, p_pos_y, c_pos_x, c_pos_y, (255,0,0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
  

    def ray_trace(self, angle, pose):
        """Returns the distance to the first obstacle from the particle. From course exercises, with sensor distance limiting"""
        wall_off = np.pi/2.
        cyl_off = np.pi
        def intersection_segment(x1, x2, y1, y2):
            # print(x1, x2, y1, y2)
            point1 = np.array([x1, y1], dtype=np.float32)
            point2 = np.array([x2, y2], dtype=np.float32)
            v1 = pose[:2] - point1
            v2 = point2 - point1
            v3 = np.array([np.cos(angle + pose[YAW] + wall_off), np.sin(angle + pose[YAW] + wall_off)],
                          dtype=np.float32)
            t1 = np.cross(v2, v1) / np.dot(v2, v3)
            t2 = np.dot(v1, v3) / np.dot(v2, v3)
            if t1 >= 0. and t2 >= 0. and t2 <= 1.:
                return t1
            return float('inf')

        def intersection_cylinder(x, y, r):
            center = np.array([x, y], dtype=np.float32)
            v = np.array([np.cos(angle + pose[YAW] + cyl_off), np.sin(angle + pose[YAW] + cyl_off)],
                         dtype=np.float32)

            v1 = center - pose[:2]
            a = v.dot(v)
            b = 2. * v.dot(v1)
            c = v1.dot(v1) - r ** 2.
            q = b ** 2. - 4. * a * c
            if q < 0.:
                return float('inf')
            g = 1. / (2. * a)
            q = g * np.sqrt(q)
            b = -b * g
            d = min(b + q, b - q)
            if d >= 0.:
                return d
            return float('inf')

        # distance to the walls
        d = min(intersection_segment(0, 0, 0, self.grid_size),
                intersection_segment(self.grid_size, self.grid_size, 0, self.grid_size),
                intersection_segment(0, self.grid_size, 0, 0),
                intersection_segment(0, self.grid_size, self.grid_size, self.grid_size))

        # distance to the obstacles
        if len(self.obstacles) > 0:
            d_obstacles = np.min([intersection_cylinder(o.x_coord, o.y_coord, o.radius) for o in self.obstacles])
            d = min(d, d_obstacles)

        # Remove inf and limit the sensor readings to the max sensor range
        final_sensor_reading = min(d, MAX_SENSOR_DISTANCE)
        assert final_sensor_reading >= 0.0
        return final_sensor_reading

    # observation is [robot position, goal position, sensor readings] (based on flocking example)
    def observation(self):
        # # print("OBSERVATION", self.robot.pose)
        # robot_x_y = self.robot.pose[:2]
        # goal_position_x_y = self.goal_position[:2]
        # # Distance to goal
        # distance = np.linalg.norm( goal_position_x_y - robot_x_y)
        # # Angle to goal
        # line_angle = np.arctan2(goal_position_x_y[Y]- robot_x_y[Y], goal_position_x_y[X] - robot_x_y[X])
        # # negative means distance is anticlockwise, postive means difference is clockwise
        # goal_angle = self.robot.pose[YAW] - line_angle
        sensor_readings = [self.ray_trace(a, self.robot.pose) for a in self.sensor_angles]
        # Make sensor reading in MAX_DISTANCE if infinity or over the max sensor reading distance
        # filter_sensor_readings = [MAX_SENSOR_DISTANCE if x == float('inf') else x for x in sensor_readings]
        # return np.concatenate((self.robot.pose, self.goal_position, filter_sensor_readings))
        return np.concatenate((self.robot.pose, self.goal_position, sensor_readings))

    def reset_positions(self):
        # TODO perhaps create goal and robot positions first to ensure we don't get stuck in random corners
        # Set random positions for obstacles
        for i in range(NUM_OBSTACLES):
            collide = True
            while collide:
                rand_pos = np.random.uniform(low=self.obstacles[i].radius, high=self.grid_size-self.obstacles[i].radius, size=(2))
                collide = False
                # Check that the obstacle does not collide with other obstacles
                for j in range(i):
                    o = self.obstacles[j]
                    if np.linalg.norm(o.position - rand_pos) < (o.radius + self.obstacles[i].radius):
                        collide = True
                        break
            self.obstacles[i].set_position(rand_pos)
    
        # minimum x and y values 
        def get_random_robot_pos(min, max): 
            collide = True
            while collide:
                rand_pos = np.random.uniform(low=min, high=max, size=(2))
                
                assert np.all(rand_pos < np.array(max)), "rand_pos out of bounds"
                assert np.all(rand_pos >= np.array(min)), "rand_pos out of bounds"

                # self.robot.set_pose(np.append(rand_pos,[0.0]))
                collide = False
                # Check that the position does not collide with other obstacles
                for o in self.obstacles:
                    fake_robot = SimpleRobot(np.append(rand_pos,[0.0]), ROBOT_RADIUS + 0.01)
                    collide = o.collide(fake_robot)
                    if collide:
                        break
            return np.append(rand_pos, [np.random.uniform(low=-np.pi, high=np.pi)])

        # Set random goal position, ensuring it does not clash with obstacles
        # Adding offset to ensure the robot doesn't spawn exactly touching the wall or too close to it that any movement causes crashing
        self.goal_position = get_random_robot_pos((0.01 + self.robot.radius , 0.01 + self.robot.radius), (self.grid_size-self.robot.radius - 0.01, self.grid_size-self.robot.radius - 0.01))
        
        # Set random robot position, a certain distance away from the goal
        min_x = max(0.01 + self.robot.radius, self.goal_position[X] - INIT_DISTANCE_FROM_GOAL)
        max_x = min(self.grid_size-self.robot.radius - 0.01, self.goal_position[X] + INIT_DISTANCE_FROM_GOAL)
        min_y = max(0.01 + self.robot.radius, self.goal_position[Y] - INIT_DISTANCE_FROM_GOAL)
        max_y = min(self.grid_size-self.robot.radius - 0.01, self.goal_position[Y] + INIT_DISTANCE_FROM_GOAL)

        self.robot.set_pose(get_random_robot_pos((min_x,min_y), (max_x, max_y)))


# Class representing a circular obstacle
class Obstacle():

    def __init__(self, x, y, radius) -> None:
        self._x_pos = x
        self._y_pos = y
        self._radius = radius

    @property
    def x_coord(self):
        return self._x_pos

    @property
    def y_coord(self):
        return self._y_pos

    @property
    def radius(self):
        return self._radius

    @property
    def position(self):
        return np.array([self._x_pos, self._y_pos])

    def set_position(self, position):
        self._x_pos = position[X]
        self._y_pos = position[Y]

    def collide(self, robot):
        if np.linalg.norm(robot.pose[:2] - self.position) <= self.radius + robot.radius:
            return True
        return False


# Class representing a simple robot, TurtleBot3 as in assignments
class SimpleRobot():

    def __init__(self, pose, radius) -> None:
         self._pose = pose
         self._radius = radius
    
    # Use semi-implicit Euler method, can use substepping if necessary, following code from Prorok package
    # Use equations from lecture slides to update position
    def update_pose(self, u, w, max_pos):
        # print(self.pose)
        # print(u, w)
        dt = 0.1
        self._pose[YAW] += dt*w
        # map to value in range -pi -> pi
        self._pose[YAW] = self.pose[YAW] % (2*np.pi)
        self._pose[YAW] = self.pose[YAW] - 2*np.pi if self.pose[YAW] > np.pi else self.pose[YAW]

        # update x and y values
        self._pose[X] += dt*u*np.cos(self.pose[YAW])
        self._pose[Y] += dt*u*np.sin(self.pose[YAW])
        # print(self.pose)

        # Limit robot positions to remaining inside the map
        clipped = np.clip(self.pose[:2], 0.0, max_pos)
        self._pose[X], self._pose[Y] = clipped[X], clipped[Y]

    def set_pose(self, pose):
        self._pose = pose

    @property
    def pose(self):
        return self._pose

    @property
    def radius(self):
        return self._radius


if __name__ == "__main__":
    env = SimpleRobotEnviroment(render_mode="rgb_array")
    # for i in range(5):
    #     print(env.step(env.action_space.sample()))
    # print(env.reset())
    # print(env.reset())
    # env.reset()

    # robot = SimpleRobot(np.array([0.5,0.5,0]),0.1)
    # robot.update_pose(0.1, np.pi)
    # print(robot.pose)
    # env.testy_boi()


    import matplotlib.pyplot as plt
    def displayImage(image):
        plt.imshow(image)
        plt.axis('off')
        plt.show()


    # print(env.observation())
    # print(env.step([0.1, np.pi/6]))
    # print(env.observation())

    x = env.render()
    displayImage(x)