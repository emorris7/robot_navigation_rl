from gym import Env
from gym import logger
from gym.spaces import Box
import numpy as np
from typing import Optional
from gym.error import DependencyNotInstalled

# Assume the robot cannot go backwards and does not stop moving (have max speed of 0.2 in assignments)
MAX_SPEED = 0.8
MIN_SPEED = 0.05
MAX_W = np.pi /4 # USE?

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Permissable distance from goal
GOAL_DISTANCE = 0.001
GOAL_ANGLE = 0.01
MAX_SENSOR_DISTANCE = 5.0


class SimpleRobotEnviroment(Env):

    # For rendering
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # define your environment
        # action space, observation space

        # Custom
        # Set the grid size that we're operating in, use continuous gridspace
        self.grid_size = 2
        # List of obstacles in the environment
        self.obstacles = np.array([])
        # TODO: randomly initiliase, ensure it does not clash with obstacles and is reachable (i.e. the robot is still fully in the grid if it reaches the space)
        # Goal position, randomly initiliase and ensure that it does not clash with any obstacles
        self.goal_position = np.array([1.0,1.0,-np.pi/2])
        # Robot start position, randomly initiliase and ensure that it does not clash with any obstacles
        self.num_sensors = 7
        self.sensor_angles = np.linspace(-np.pi/2, np.pi/2, self.num_sensors)
        # Robot position needs to be a float
        self.robot = SimpleRobot(np.array([0.5,0.5,0.0]), 0.105 / 2)

        # Define observation space, [distance to goal, angle to goal, sensor readings (check how many from assignment/Prorok code) (paper uses 30)]
        observation_shape = self.num_sensors+2
        obs_min = np.full((observation_shape,), 0.0)
        # min distance, min angle
        # TODO what should the max and min for angle between be
        obs_min[0], obs_min[1] = 0.0, -2*np.pi
        obs_max = np.full((observation_shape,), MAX_SENSOR_DISTANCE)
        # max distance (distance between two opposite corners of the square), max angle
        obs_max[0], obs_max[1] = np.sqrt(2)*self.grid_size, 2*np.pi
        self.observation_space = Box(low = obs_min, high = obs_max, shape=(observation_shape,), dtype = np.float32)

        # Define action spac, [forward velocity, angular velocity] 
        # CHANGE: MAX_W?
        self.action_space = Box(low=np.array([MIN_SPEED, -np.pi]), high=np.array([MAX_SPEED, np.pi]), shape=(2,), dtype=np.float32)


        # For rendering
        self.path = [np.copy(self.robot.pose[:2])]
        self.steps_beyond_terminated = None
        self.render_mode = render_mode

        self.offset = 30
        self.screen_width = 600 + self.offset
        self.screen_height = 600 + self.offset
        self.screen = None
        self.clock = None
  
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
        self.robot.update_pose(u,w)
        self.path.append(np.copy(self.robot.pose[:2]))

        # Compute observations
        # [distance, angle, sensor readings ...]
        observation = self.observation()

        # Compute reward
        # TODO: more sophisticated reward function
        reward = -observation[0]

        # Compute done
        done = False
        # if robot is outside the grid
        if (self.robot.pose[X] + self.robot.radius >= self.grid_size) or (self.robot.pose[Y] + self.robot.radius >= self.grid_size) or (self.robot.pose[X] < 0) or (self.robot.pose[Y] < 0):
            done = True
            reward -= 50
        else:
            # if the robot collides with an object
            collision = np.any([o.collide(self.robot) for o in self.obstacles])
            if collision:
                done = True
                reward -= 50
            # if the robot reaches the goal (is within some distance of the goal)
            elif observation[0] <= GOAL_DISTANCE:
                done = True
                reward += 50
        # Allow us to through warning and stop unexpected behaviour
        if done:
            self.steps_beyond_terminated = 0

        # Record dictionary
        info_dict = {}

        # Render the environment
        if self.render_mode == "human":
            self.render()

        return np.array(observation), reward, done, info_dict

    def reset(self):
        # reset your environment

        # Reset goal position
        self.goal_position = np.array([1.0,1.0,0.0])

        # Reset robot position
        self.robot.set_pose(np.array([0.5,0.5,0.0]))

        # Reset obstacles
        self.obstacles = np.array([])

        # Random bits and bobs
        self.steps_beyond_terminated = None
        self.screen = None
        self.clock = None
        self.path = [np.copy(self.robot.pose[:2])]

        return self.observation()

    def render(self):
        # render your environment (can be a visualisation/print)
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

        world_width = self.grid_size
        scale = self.screen_width / world_width
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # define function to get point on circle on line to visualize angles
        def get_point_on_circle(x,y,r,angle):
            x_circle = x + r*np.cos(angle)
            y_circle = y + r*np.sin(angle)
            return x_circle, y_circle

        # draw the box
        max_size_scale_offset = int(self.grid_size*scale) - self.offset
        min_size_scale_offset = 0 + self.offset
        gfxdraw.hline(self.surf, min_size_scale_offset, max_size_scale_offset, min_size_scale_offset, (0,0,0))
        gfxdraw.hline(self.surf, min_size_scale_offset, max_size_scale_offset, max_size_scale_offset, (0,0,0))
        gfxdraw.vline(self.surf, min_size_scale_offset, min_size_scale_offset, max_size_scale_offset, (0,0,0))
        gfxdraw.vline(self.surf, max_size_scale_offset, min_size_scale_offset, max_size_scale_offset, (0,0,0))
        

        # draw goal
        goal_x = self.goal_position[X]*scale + self.offset
        goal_y = self.goal_position[Y]*scale + self.offset
        goal_r = self.robot.radius*scale
        # print(goal_x, goal_y, goal_r)
        # may need to cast value to int
        gfxdraw.aacircle(self.surf, int(goal_x), int(goal_y), int(goal_r), (52, 175, 61),)
        # plot orientation at goal
        x_goal_circle, y_goal_circle = get_point_on_circle(self.goal_position[X], self.goal_position[Y], self.robot.radius, self.goal_position[YAW])
        x_goal_circle = x_goal_circle*scale + self.offset
        y_goal_circle = y_goal_circle*scale + self.offset
        gfxdraw.line(self.surf, int(goal_x), int(goal_y), int(x_goal_circle), int(y_goal_circle), (0,0,0))

        # draw robot
        robot_x = self.robot.pose[X]*scale + self.offset
        robot_y = self.robot.pose[Y]*scale + self.offset
        robot_r = self.robot.radius*scale
        # may need to cast value to int
        gfxdraw.aacircle(self.surf, int(robot_x), int(robot_y), int(robot_r), (159, 197, 236),)
        gfxdraw.filled_circle(self.surf, int(robot_x), int(robot_y), int(robot_r), (159, 197, 236),)
        # plot orientation of robot
        x_robot_circle, y_robot_circle = get_point_on_circle(self.robot.pose[X], self.robot.pose[Y], self.robot.radius, self.robot.pose[YAW])
        x_robot_circle = x_robot_circle*scale + self.offset
        y_robot_circle = y_robot_circle*scale + self.offset
        gfxdraw.line(self.surf, int(robot_x), int(robot_y), int(x_robot_circle), int(y_robot_circle), (0,0,0))

        #draw obstacles
        for o in self.obstacles:
            o_x = o.x_coord*scale + self.offset
            o_y = o.y_coord*scale + self.offset
            o_r = o.radius*scale
            # may need to cast value to int
            gfxdraw.aacircle(self.surf, int(o_x), int(o_y), int(o_r), (0, 102, 204),)
            gfxdraw.filled_circle(self.surf, int(o_x), int(o_y), int(o_r), (0, 102, 204),)

        # draw path
        # print(self.path)
        for i in range(1,len(self.path)):
            c_pos_x = self.path[i][X]*scale + self.offset
            # print(c_pos_x)
            c_pos_y = self.path[i][Y]*scale + self.offset
            p_pos_x = self.path[i-1][X]*scale + self.offset
            # print(p_pos_x)
            p_pos_y = self.path[i-1][Y]*scale + self.offset
            gfxdraw.line(self.surf, int(p_pos_x), int(p_pos_y), int(c_pos_x), int(c_pos_y), (255,0,0))

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
        # TODO limit distance
        """Returns the distance to the first obstacle from the particle. From course exercises"""
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
            d_obstacles = np.min([intersection_cylinder(o.x_pos, o.y_pos, o.radius) for o in self.obstacles])
            d = min(d, d_obstacles)

        return d

    def observation(self):
        # print("OBSERVATION", self.robot.pose)
        robot_x_y = self.robot.pose[:2]
        goal_position_x_y = self.goal_position[:2]
        # Distance to goal
        distance = np.linalg.norm( goal_position_x_y - robot_x_y)
        # Angle to goal
        line_angle = np.arctan2(goal_position_x_y[Y]- robot_x_y[Y], goal_position_x_y[X] - robot_x_y[X])
        # negative means distance is anticlockwise, postive means difference is clockwise
        goal_angle = self.robot.pose[YAW] - line_angle
        sensor_readings = [self.ray_trace(a, self.robot.pose) for a in self.sensor_angles]
        #TODO make inf into max reading?
        filter_sensor_readings = [MAX_SENSOR_DISTANCE if x == float('inf') else x for x in sensor_readings]
        return np.array([distance, goal_angle] + filter_sensor_readings)



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

    def collide(self, robot):
        if np.linalg.norm(robot.pose[:2] - np.array([self.x_coord, self.y_coord])) - robot.radius <= self.radius:
            return True
        return False


# Class representing a simple robot, TurtleBot3 as in assignments
class SimpleRobot():

    def __init__(self, pose, radius) -> None:
         self._pose = pose
         self._radius = radius
    
    # Use semi-implicit Euler method, can use substepping if necessary
    # Use equations from lecture slides to update position
    def update_pose(self, u, w):
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
    for i in range(20):
        env.step(env.action_space.sample())
    # print(env.reset())
    # print(env.reset())
    env.reset()

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