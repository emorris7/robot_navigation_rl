from typing import Optional
from env.SimpleEnvironment import X,Y, YAW, SimpleRobot, Obstacle, SimpleRobotEnviroment
from env.SimpleEnvironment_condensed_obs import SimpleRobotEnviromentCO
from waypoints.a_star import plot_path
import numpy as np
from gym import logger
from gym.error import DependencyNotInstalled
import func_timeout

WAYPOINT_GRID_SIZE = 19

class SimpleRobotEnvironmentWP(SimpleRobotEnviroment):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)

        # self.goal_position = np.array([1.5,1.5,np.pi/2])
        # self.robot = SimpleRobot(np.array([0.3,0.3,-np.pi/1.5]), 0.105 / 2)
        # self.obstacles = [Obstacle(0.7, 0.7, 0.1)]
         
        self.way_points = None
        while self.way_points is None:
            # Dodgy hack to deal with when environments are created where its very hard/impossible to plot a path
            try:
                self.way_points = func_timeout.func_timeout(10, plot_path, args=[self.robot.pose[:2], self.goal_position[:2], self.grid_size, WAYPOINT_GRID_SIZE, self.obstacles])
            except func_timeout.FunctionTimedOut:
                # print("reset")
                self.reset_positions()
                # self.way_points = None
        # self.way_points = plot_path(start_position=self.robot.pose[:2], goal_position=self.goal_position[:2], continuous_size=self.grid_size, grid_size=WAYPOINT_GRID_SIZE, obstacles=self.obstacles)
        # print(self.way_points)
        # self.way_points = []


    def step(self, action):
       observation, reward, done, info_dict = super().step(action)

       if not done:
        for i, w in enumerate(self.way_points):
                # way points can be robot.radius away from an obstacle, ensure we don't reward robot for crashing by staying out of this range
                if np.linalg.norm(w - self.robot.pose[:2]) <= self.robot.radius+0.01:
                    # Give the robot a reward equal to the sum of all previous rewards not collected
                    reward += 25
                    # remove all waypoints from behind to ensure robot is only rewarded for following the path in the forward direction
                    self.way_points = self.way_points[i+1:]
                    break

       return observation, reward, done, info_dict

    def reset(self):
        observation = super().reset()

        # self.way_points = plot_path(start_position=self.robot.pose[:2], goal_position=self.goal_position[:2], continuous_size=self.grid_size, grid_size=WAYPOINT_GRID_SIZE, obstacles=self.obstacles)
        self.way_points = None
        while self.way_points is None:
            try:
                self.way_points = func_timeout.func_timeout(20, plot_path, args=[self.robot.pose[:2], self.goal_position[:2], self.grid_size, WAYPOINT_GRID_SIZE, self.obstacles])
            except func_timeout.FunctionTimedOut:
                self.reset_positions()

        return observation

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

        # plot way points
        for w in self.way_points:
            w_x = int((w[0] + self.offset)*self.scale)
            w_y = int((w[1] + self.offset)*self.scale)
            w_r = int((self.robot.radius+0.01)*self.scale)
            gfxdraw.aacircle(self.surf, w_x, w_y, w_r, (255, 153, 153),)
            gfxdraw.filled_circle(self.surf, w_x, w_y, w_r, (255, 153, 153),)

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



if __name__ == "__main__":
    env = SimpleEnvironmentWP(render_mode="rgb_array")
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