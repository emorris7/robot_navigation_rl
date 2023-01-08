# code taken from https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
import numpy as np
from env.SimpleEnvironment import Obstacle
from env.SimpleEnvironment import X,Y

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def continuous_to_grid(continuous_coord, block_size):
    grid = int(np.round(continuous_coord / block_size))
    print(grid)
    return grid

def grid_to_continuous(grid_coord, block_size):
    continuous_val = grid_coord*block_size
    return continuous_val

def create_grid(block_size, grid_size, obstacles: list[Obstacle]): 
    obstacle_map = np.zeros(shape=(grid_size,grid_size))

    for o in obstacles:
        min_x_min_y = o.position - o.radius
        max_x_max_y = o.position + o.radius

        min_x = continuous_to_grid(min_x_min_y[X], block_size=block_size)
        min_y = continuous_to_grid(min_x_min_y[Y], block_size=block_size)
        max_x = continuous_to_grid(max_x_max_y[X], block_size=block_size)
        max_y = continuous_to_grid(max_x_max_y[Y], block_size=block_size)

        for i in range(min_x, max_x+1):
            for j in range(min_y, max_y+1):
                obstacle_map[i][j] = 1

    return obstacle_map

def plot_path(start_position, goal_position, continuous_size, grid_size, obstacles: list[Obstacle]):
    block_size = continuous_size/grid_size 
    start_position_grid = (continuous_to_grid(start_position[X], block_size=block_size), continuous_to_grid(start_position[Y], block_size=block_size))
    goal_position_grid = (continuous_to_grid(goal_position[X], block_size=block_size), continuous_to_grid(goal_position[Y], block_size=block_size))

    grid = create_grid(block_size=block_size, grid_size=grid_size, obstacles=obstacles)
    path = astar(grid, start_position_grid, goal_position_grid)

    for i in path:
        grid[i[0]][i[1]] = 2
    print(grid)
    return path


def main():

    # maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # start = (0, 0)
    # end = (7, 6)

    # path = astar(maze, start, end)
    # print(path)
    path = plot_path([0.1,0.1], [0.8, 0.8], 2.0, 19, [Obstacle(0.4, 0.4, 0.1)])
    print(path)

if __name__ == '__main__':
    main()