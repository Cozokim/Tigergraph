import numpy as np
import matplotlib.pyplot as plt
import math

def create_grid_dict(n, include_diagonals=False):
    grid = {}
    for i in range(n):
        for j in range(n):
            neighbors = []
            # Direct neighbors
            if i > 0:
                neighbors.append((i-1, j))  # Up
            if i < n-1:
                neighbors.append((i+1, j))  # Down
            if j > 0:
                neighbors.append((i, j-1))  # Left
            if j < n-1:
                neighbors.append((i, j+1))  # Right
            # Diagonal neighbors
            if include_diagonals:
                if i > 0 and j > 0:
                    neighbors.append((i-1, j-1))  # Up-Left
                if i > 0 and j < n-1:
                    neighbors.append((i-1, j+1))  # Up-Right
                if i < n-1 and j > 0:
                    neighbors.append((i+1, j-1))  # Down-Left
                if i < n-1 and j < n-1:
                    neighbors.append((i+1, j+1))  # Down-Right
            grid[(i, j)] = neighbors
    return grid

def plot_nodes(ax, grid_dict):
    for (i, j) in grid_dict.keys():
        # Plot the node
        ax.plot(j, -i, 'bo')  # 'bo' means blue color, circle marker

def plot_grid_squares(ax, grid_dict):
    for (i, j) in grid_dict.keys():
        # Draw a square around each node
        square = plt.Rectangle((j - 0.5, -i - 0.5), 1, 1, fill=None, edgecolor='r')
        ax.add_patch(square)

def plot_neighbor_lines(ax, grid_dict):
    for (i, j), neighbors in grid_dict.items():
        for (ni, nj) in neighbors:
            # Draw a line to each neighbor
            ax.plot([j, nj], [-i, -ni], 'k-')  # 'k-' means black color, solid line

def plot_grid(grid_dict):
    fig, ax = plt.subplots()
    plot_nodes(ax, grid_dict)
    plot_grid_squares(ax, grid_dict)
    plot_neighbor_lines(ax, grid_dict)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Remove the background axis information
    ax.axis('off')

    plt.show()




def get_positions(grid_dict):
    # Get all the positions in the grid
    return list(grid_dict.keys())

def get_possible_actions(position, grid_dict):
    # Get the possible actions from a position in the grid
    actions = grid_dict[position]
    return actions

def get_reward(position, goal):
    if len(position) != len(goal):
        raise ValueError("Position and goal must have the same number of dimensions")
    
    distance = math.sqrt(sum((p - g) ** 2 for p, g in zip(position, goal)))
    reward = -distance
    return reward    

def get_action(possible_actions, goal, how="random"):
    if how == "random":
        return possible_actions[0]
    elif how == "best":
        best_action = None
        best_reward = float('-inf')

        for action in possible_actions:
            reward = get_reward(action, goal)
            if reward > best_reward:
                best_reward = reward
                best_action = action


        return best_action
    else:
        raise ValueError("how must be 'random' or 'best'")

def fill_square(ax, position, color='yellow'):
    i, j = position
    square = plt.Rectangle((j - 0.5, -i - 0.5), 1, 1, color=color, alpha=0.5)
    ax.add_patch(square)

def plot_frame(ax,position,goal,grid_dict,color='yellow'):
    plot_grid_squares(ax, grid_dict)
    fill_square(ax, position, color=color)
    fill_square(ax, goal, color='red')
    plot_nodes(ax, grid_dict)
    #plot_neighbor_lines(ax, grid_dict)
    #ax.set_aspect('equal')
    #ax.axis('off')
    #plt.show()


def plot_wall_restrictions(ax, wall_restrictions):
    for (i, j) in wall_restrictions:
        # Plot the wall restriction as a filled square
        square = plt.Rectangle((j - 0.5, -i - 0.5), 1, 1, color='black', alpha=0.7)
        ax.add_patch(square)

def plot_grid_with_path(grid_dict, path, position, goal,wall_restrictions):
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots()
    plot_grid_squares(ax, grid_dict)
    plot_wall_restrictions(ax, grid_dict)
    #plot_neighbor_lines(ax, grid_dict)

    # Plot the agent's path
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        ax.plot([p1[1], p2[1]], [-p1[0], -p2[0]], 'y-', linewidth=2)  # 'y-' means yellow color, solid line

    # Plot the current position as a circle
    ax.plot(position[1], -position[0], 'go', markersize=10)  # 'go' means green color, circle marker

    # Plot the goal as a star
    ax.plot(goal[1], -goal[0], 'g*', markersize=15)  # 'r*' means red color, star marker

    #plot wall_restrictions
    for i in range(len(wall_restrictions)):
        ax.plot(wall_restrictions[i][1], -wall_restrictions[i][0], 'bs', markersize=10)
    # Set the aspect of the plot to be equal

    ax.set_aspect('equal')

    # Remove the background axis information
    ax.axis('off')

    plt.show()
    plt.ion()  # Turn on interactive mode if needed

#################################################
# Example usage
import random
n = 8
grid_dict = create_grid_dict(n, include_diagonals=False)
wall_restrictions = [(1, 4), (4, 4), (5, 4), (6, 4),(7,4)] 
#dead_positions=[4,2]

# Initialize robot movement

good_paths=[]
for rep in range(1000):
    agent_location = (7, 0)
    init_loc=agent_location
    goal=(7,7)
    max_len_path=100
    flag=0
    agent_path=[agent_location]
    restrictions=wall_restrictions.copy()
    while flag==0:
        restrictions.append(agent_location)
        posible_actions=get_possible_actions(agent_location,grid_dict)
        posible_actions=[x for x in posible_actions if x not in restrictions]
        max_len_path-=1

        if len(posible_actions)==0:
            flag=1
            print("No possible actions")
        elif agent_location==goal:
            flag=1
        elif max_len_path==0:
            flag=1
            print("max_len_path reached")
        
        else:
            random.shuffle(posible_actions)

            threshold=0.5
            p=random.random()
            if p<threshold:
                agent_location=get_action(posible_actions,goal,how="random")
            else:
                agent_location=get_action(posible_actions,goal,how="best")

            agent_path.append(agent_location)
            print(agent_location)
            

        if flag==1:
            if agent_location==goal:
                print("Goal reached")
                good_paths.append(agent_path)


plot_grid_with_path(grid_dict, agent_path, init_loc, goal,wall_restrictions)

len(good_paths)
good_paths.sort(key=len)
locations=list(grid_dict.keys())
hits=np.zeros([n,n])#{key: 0 for key in locations}
for location in locations:
    for path in good_paths: 
        if location in path:
            hits[location]= (hits[location] +1)

img=hits/len(good_paths)
plt.imshow(img)
plt.show()