from env.SimpleEnvironment import SimpleRobotEnviroment, SimpleRobot
from typing import Optional
from env.SimpleEnvironment import MAX_SENSOR_DISTANCE, X, Y, YAW
import numpy as np
from gym.spaces import Box

class SimpleRobotEnviromentCO(SimpleRobotEnviroment):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)

        # Define observation space, [distance to goal, angle to goal position, difference between current and goal angles, sensor readings (check how many from assignment/Prorok code) (paper uses 30)]
        observation_shape = self.num_sensors + 3
        # observation_shape = self.num_sensors + 4
        obs_min = np.full((observation_shape,), 0.0)
        obs_min[1], obs_min[2] = -np.pi, -np.pi
        # obs_min[1], obs_min[2], obs_min[3]= -np.pi, -np.pi, -np.pi

        obs_max = np.full((observation_shape,), MAX_SENSOR_DISTANCE)
        max_dist = np.linalg.norm(np.array([self.grid_size, self.grid_size]))
        # max x position robot, max y position robot, max yaw robot, need to account for observing values when we move outside the grid
        obs_max[0], obs_max[1], obs_max[2] = max_dist, np.pi, np.pi
        # obs_max[0], obs_max[1], obs_max[2], obs_max[3] = max_dist, np.pi, np.pi, np.pi
        self.observation_space = Box(low = obs_min, high = obs_max, shape=(observation_shape,), dtype = np.float32)

        # self.goal_position = np.array([1.0,1.0,np.pi/2])
        # self.robot = SimpleRobot(np.array([0.7,0.7,-np.pi/1.5]), 0.105 / 2)


    # observation is [distance to goal, angle to goal position, difference between current and goal angles, sensor readings (check how many from assignment/Prorok code) (paper uses 30)]
    def observation(self):
        # # print("OBSERVATION", self.robot.pose)
        robot_x_y = self.robot.pose[:2]
        goal_position_x_y = self.goal_position[:2]
        # Distance to goal
        distance = np.linalg.norm(goal_position_x_y - robot_x_y)

        # Angle to goal from the point, taking difference between the current orientation of the robot and the angle of the straight line to the point
        robot_pos_angle = self.robot.pose[YAW] - np.arctan2(goal_position_x_y[Y]- robot_x_y[Y], goal_position_x_y[X] - robot_x_y[X])
        # always take the smallest angle between the two positions, negative means distance is anticlockwise, postive means difference is clockwise
        angle_sign = (-1) if robot_pos_angle >=0 else 1
        goal_angle = robot_pos_angle if abs(robot_pos_angle) <= np.pi else (angle_sign*(2*np.pi - abs(robot_pos_angle)))

        # robot_angle_diff = self.robot.pose[YAW] - self.goal_position[YAW]
        # # always take the smallest angle between the two positions, negative means distance is anticlockwise, postive means difference is clockwise
        # angle_diff_sign = (-1) if robot_angle_diff >=0 else 1
        # angle_difference = robot_angle_diff if abs(robot_angle_diff) <= np.pi else (angle_diff_sign*(2*np.pi - abs(robot_angle_diff)))
        angle_difference = self.robot.pose[YAW]

        sensor_readings = [self.ray_trace(a, self.robot.pose) for a in self.sensor_angles]

        return np.concatenate(([distance, goal_angle, angle_difference], sensor_readings))
        # return np.concatenate(([distance, goal_angle, self.robot.pose[YAW], self.goal_position[YAW]], sensor_readings))

if __name__ == "__main__":
    env = SimpleRobotEnviromentCO(render_mode="rgb_array")
    print(env.observation())
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

