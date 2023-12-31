from itertools import count
import pygame as pg
import numpy as np
from queue import PriorityQueue
import queue
import os
import sys
import random
from collections import deque
import time
from queue import Queue
import heapq
import tkinter as tk
from sprite_game import Button, UIE
from tkinter import filedialog

# colors
one = (79, 189, 186)  # player
two = (206, 171, 147)  # wall
three = (227, 202, 165)
four = (255, 251, 233)
five = (246, 137, 137) #destination
six = (255, 0, 0)
seven = (0, 255, 0)
eight = (172, 243, 157)
Mindaro = "#C5D86D"
Black = "#fff"

pg.init()

# set size of each screen, screen 1 is the main screen, screen 2 is the button screen
size = (1176, 729)
screen1_size = (size[0] - 470, size[1])
screen2_size = (size[0] - 706, size[1])
screen1 = pg.display.set_mode(screen1_size)
screen2 = pg.display.set_mode(screen2_size)
screen = pg.display.set_mode(size, pg.RESIZABLE)

screen2_pos = (470, 729)
width, height = screen.get_size()
mouse_pos = pg.mouse.get_pos()

pg.display.set_caption("MAZE")

# size - "hard" level of maze; #MAX: 33
block = 20

width = 20
height = 20
margin = 2
count = 0
#flags
count_algo = 1
count_level = 0
count_map = 0
isReset = 0
isPlay = 0

grid = [[0 for x in range(block)] for y in range(block)]
gobalStartPoint = None
gobalEndPoint = None

done = False
clock = pg.time.Clock()
found = False
neighbour = []
button_list = []
line_list = []
algo = "BFS"
level = "EASY"
answer = ""
path = ""
node_visit = ""

root = tk.Tk()
root.withdraw()


def savegrid():
    global grid
    np.savetxt(r"./maze.txt", grid)


def loadgrid(index):
    global grid
    if index == 0:
        grid = np.loadtxt(r"./maingame/maze.txt").tolist()
    elif index == 1:
        grid = np.loadtxt(r"./mazemap/Maze1/maze.txt").tolist()
    elif index == 2:
        grid = np.loadtxt(r"./mazemap/Maze2/maze.txt").tolist()
    elif index == 3:
        grid = np.loadtxt(r"./mazemap/Maze3/maze.txt").tolist()
    elif index == 4:
        grid = np.loadtxt(r"./mazemap/Maze4/maze.txt").tolist()
    elif index == 5:
        grid = np.loadtxt(r"./mazemap/Maze5/maze.txt").tolist()


def loadgridWithLevel(index, level):
    global grid
    if level == "HARD":
        if index == 0:
            grid = np.loadtxt(r"./maingame/maze.txt").tolist()
        elif index == 1:
            grid = np.loadtxt(r"./mazemap/Maze1/maze.txt").tolist()
        elif index == 2:
            grid = np.loadtxt(r"./mazemap/Maze2/maze.txt").tolist()
        elif index == 3:
            grid = np.loadtxt(r"./mazemap/Maze3/maze.txt").tolist()
        elif index == 4:
            grid = np.loadtxt(r"./mazemap/Maze4/maze.txt").tolist()
        elif index == 5:
            grid = np.loadtxt(r"./mazemap/Maze5/maze.txt").tolist()
    if level == "INTERMEDIATE":
        if index == 0:
            grid = np.loadtxt(r"./mazemap/Maze6/maze6.txt").tolist()
        elif index == 1:
            grid = np.loadtxt(r"./mazemap/Maze7/maze7.txt").tolist()
        elif index == 2:
            grid = np.loadtxt(r"./mazemap/Maze8/maze8.txt").tolist()
        if index == 3:
            grid = np.loadtxt(r"./mazemap/Maze9/maze6.txt").tolist()
        elif index == 4:
            grid = np.loadtxt(r"./mazemap/Maze7/maze7.txt").tolist()
        elif index == 5:
            grid = np.loadtxt(r"./mazemap/Maze8/maze8.txt").tolist()
    if level == "EASY":
        if index == 0:
            grid = np.loadtxt(r"./mazemap/Maze11/maze11.txt").tolist()
        elif index == 1:
            grid = np.loadtxt(r"./mazemap/Maze12/maze12.txt").tolist()
        elif index == 2:
            grid = np.loadtxt(r"./mazemap/Maze13/maze13.txt").tolist()
        if index == 3:
            grid = np.loadtxt(r"./mazemap/Maze11/maze11.txt").tolist()
        elif index == 4:
            grid = np.loadtxt(r"./mazemap/Maze12/maze12.txt").tolist()
        elif index == 5:
            grid = np.loadtxt(r"./mazemap/Maze14/maze14.txt").tolist()


def startp(maze, i, j):
    for x in range(len(maze[0])):
        try:
            i = maze[x].index(2)
            j = x
            print(j)
            return i, j
        except:
            pass


def bfs():
    global grid, neighbour, answer, isReset, path, node_visit
    neighbourr()

    start, end = S_E(grid, 0, 0)
    end_final = end
    open_set = Queue()
    open_set.put(start)
    came_from = {}
    visited = set()

    while not open_set.empty():
        eventHandle()
        current = open_set.get()
        visited.add(current)

        for nei in neighbour[current[0] * len(grid[0]) + current[1]]:
            if nei not in visited:
                visited.add(nei)  # fix slow
                open_set.put(nei)
                came_from[nei] = current
                if nei == end:
                    answer = "Finished - solve by BFS"
                    path = f"Approved nodes: {short_path(came_from, end)} nodes"
                    node_visit = f"Visited nodes: {len(visited)} nodes"
                    isReset = 1
                    return True
        grid[current[0]][current[1]] = 5
        print_grid(grid)
        time.sleep(0.05)  # Add a delay to make it slower

    answer = "There is no path to reach the endpoint."
    return False


def eventHandle():
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()


def dfs():
    global grid, neighbour, answer, isReset, path, node_visit
    neighbourr()

    start, end = S_E(grid, 0, 0)

    stack = [start]
    came_from = {}
    visited = set()

    while stack:
        # Function to handle mouse operations, this will avoid the not responding situation
        eventHandle()
        # Get the element at the top of the stack
        current = stack[-1]

        if current == end:
            answer = "Finished - solve by DFS"
            node_visit = f"Visited nodes: {len(visited)} nodes"
            path = f"Approved nodes: {short_path(came_from, end)} nodes"
            isReset = 1
            return True

        if current not in visited:
            visited.add(current)
            # Simulation code to visualize the DFS process
            grid[current[0]][current[1]] = 5
            print_grid(grid)
            time.sleep(0.05)  # Add a delay to make it slower

        # call the function and add unvisited neighbors of current point to unvisited_neighbors[]
        unvisited_neighbors = [
            nei
            for nei in neighbour[current[0] * len(grid[0]) + current[1]]
            if nei not in visited
        ]

        if unvisited_neighbors:
            # Choose a random neighbor near the current point, then add it to the top of stack
            random.shuffle(unvisited_neighbors)
            next_neighbor = unvisited_neighbors[0]
            stack.append(next_neighbor)
            came_from[next_neighbor] = current
        else:
            # No unvisited neighbors, backtrack
            stack.pop()
    answer = "There is no path to reach the endpoint."

    return False


def greedy():
    global grid, neighbour, answer, isReset, path, node_visit
    neighbourr()

    start, end = S_E(grid, 0, 0)
    count = 0
    open_set = PriorityQueue()
    open_set.put((h(start, end), start))
    came_from = {}
    visited = set()

    while not open_set.empty():
        eventHandle()
        _, current = open_set.get()

        if current == end:
            answer = "Finished - solve by Greedy"
            node_visit = f"Visited nodes: {count} nodes"
            path = f"Approved nodes: {short_path(came_from, end)} nodes"
            isReset = 1
            return True
        
        if current not in visited:
            visited.add(current)
            grid[current[0]][current[1]] = 5  # Mark visited node
            print_grid(grid)  # Function to display the grid (customize as needed)
            time.sleep(0.05)  # Add a delay to make it slower

        for nei in neighbour[current[0] * len(grid[0]) + current[1]]:
            if nei not in visited:
                count += 1 #count the visited node
                open_set.put((h(nei, end), nei))
                came_from[nei] = current
                grid[nei[0]][nei[1]] = 5
        
    answer = "There is no path to reach the endpoint."
    return False


def dijkstra():
    global grid, neighbour, answer, isReset, path, node_visit
    neighbourr()

    start, end = S_E(grid, 0, 0)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while open_set:
        eventHandle()
        _, current = heapq.heappop(open_set)

        if current not in visited:
            visited.add(current)
            grid[current[0]][current[1]] = 5  # Mark visited node
            # Function to display the grid (customize as needed)
            print_grid(grid)
            time.sleep(0.1)  # Add a delay to make it slower

        if current == end:
            answer = "Finished - solve by Dijkstra"
            node_visit = f"Visited nodes: {len(visited)} nodes"
            path = f"Approved nodes: {short_path(came_from, end)} nodes"
            simulate_dijkstra_process(came_from, start, end, visited)
            isReset = 1
            return True

        for nei in neighbour[current[0] * len(grid[0]) + current[1]]:
            new_cost = cost_so_far[current] + 1
            if nei not in cost_so_far or new_cost < cost_so_far[nei]:
                cost_so_far[nei] = new_cost
                priority = new_cost
                heapq.heappush(open_set, (priority, nei))
                came_from[nei] = current
    answer = "There is no path to reach the endpoint."

    return False


def simulate_dijkstra_process(came_from, start, end, visited):
    # Backtrack to find the solution path
    current = end
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()

    # Mark the solution path and print it
    for node in visited:
        if node == start:
            grid[node[0]][node[1]] = 3  # Mark start node
        elif node == end:
            grid[node[0]][node[1]] = 3  # Mark end node
        elif node in path:
            grid[node[0]][node[1]] = 4  # Mark solution path nodes
        else:
            grid[node[0]][node[1]] = 6  # Mark other visited nodes



def print_grid(grid):
    global screen1
    global width, height, margin

    screen1.fill(two)

    for row in range(block):
        for column in range(block):
            if grid[row][column] == 1:
                color = three
            elif grid[row][column] == 2:
                color = one
            elif grid[row][column] == 3:
                color = five
            elif grid[row][column] == 4:
                color = one
            elif grid[row][column] == 5:
                color = seven
            elif grid[row][column] == 6:
                color = seven
            else:
                color = four

            pg.draw.rect(
                screen1,
                color,
                [
                    margin + (margin + width) * column,
                    margin + (margin + height) * row,
                    width,
                    height,
                ],
            )

    pg.display.flip()


def neighbourr():
    global grid, neighbour
    # Get the all neighbourr of all point in list and store in array
    neighbour = [[] for col in range(len(grid)) for row in range(len(grid))]
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid)):
            # neighbour[count] == []
            if i > 0 and grid[i - 1][j] != 1:
                neighbour[count].append((i - 1, j))
            if j > 0 and grid[i][j - 1] != 1:
                neighbour[count].append((i, j - 1))
            if i < len(grid) - 1 and grid[i + 1][j] != 1:
                neighbour[count].append((i + 1, j))
            if j < len(grid) - 1 and grid[i][j + 1] != 1:
                neighbour[count].append((i, j + 1))
            count += 1


# herulics function
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def S_E(maze, start, end):
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y] == 2:
                start = x, y
            if grid[x][y] == 3:
                end = x, y

    return start, end


def short_path(came_from, current):
    count = 1
    grid[current[0]][current[1]] = 3 #end node
    while current in came_from:
        current = came_from[current]
        if current not in came_from:  
            grid[current[0]][current[1]] = 3 #start node
        else:
            grid[current[0]][current[1]] = 4
        count += 1
    return count


def a_star():
    global grid, neighbour, answer, isReset, path, node_visit
    neighbourr()

    start, end = S_E(grid, 0, 0)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    open_set_his = {start}
    came_from = {}

    # create a set of g score and f score, set g score of start point = 0 and calculate the f point
    g_score = [float("inf") for row in grid for spot in row]
    g_score[start[0] * len(grid[0]) + start[1]] = 0
    f_score = [float("inf") for row in grid for spot in row]
    f_score[start[0] * len(grid[0]) + start[1]] = h(start, end)

    while not open_set.empty():
        eventHandle()
        current = open_set.get()[2]
        open_set_his.remove(current)
        grid[current[0]][current[1]] = 5  # Mark visited nodes
        print_grid(grid)
        time.sleep(0.05)

        if current == end:
            answer = "Finished - solve by A*"
            node_visit = f"Visited nodes: {count} nodes"
            path = f"Approved nodes: {short_path(came_from, end)} nodes"
            isReset = 1
            return True
        # Go through every neighboir point of current point
        for nei in neighbour[current[0] * len(grid[0]) + current[1]]:
            # We go to the neighbor so the g score is increase by one
            temp_g_score = g_score[current[0] * len(grid[0]) + current[1]] + 1
            if temp_g_score < g_score[nei[0] * len(grid[0]) + nei[1]]:
                came_from[nei] = current
                g_score[nei[0] * len(grid[0]) + nei[1]] = temp_g_score
                f_score[nei[0] * len(grid[0]) + nei[1]] = temp_g_score + h(nei, end)
                if nei not in open_set_his:
                    count += 1
                    open_set.put((f_score[nei[0] * len(grid[0]) + nei[1]], count, nei))
                    open_set_his.add(nei)

    answer = "There is no path to reach the endpoint."
    return False


def generate_solvability_maze_with_user_points(start_x, start_y, end_x, end_y):
    global grid

    def is_valid(x, y):
        return 0 <= x < block and 0 <= y < block

    def get_neighbours(x, y):
        neighbours = []
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            new_x, new_y = x + dx, y + dy
            if is_valid(new_x, new_y):
                neighbours.append((new_x, new_y))
        return neighbours

    def create_maze(x, y):
        grid[y][x] = 0  # Mark the cell as visited
        neighbours = get_neighbours(x, y)
        random.shuffle(neighbours)

        for next_x, next_y in neighbours:
            if grid[next_y][next_x] == 1:
                grid[(y + next_y) // 2][(x + next_x) // 2] = 0
                create_maze(next_x, next_y)

    # Initialize the maze with walls (1s)
    grid = [[1 for _ in range(block)] for _ in range(block)]

    # Make sure the start and end points are valid
    start_x, start_y = max(0, min(start_x, 31)), max(0, min(start_y, 31))
    end_x, end_y = max(0, min(end_x, 31)), max(0, min(end_y, 31))

    # Generate the maze
    create_maze(start_x, start_y)

    # Mark the start and end points
    grid[start_x][start_y] = 2
    grid[end_x][end_y] = 3

    # grid[start_x][start_y] = 2
    # grid[end_y][end_x] = 3

    return grid


def get_font(size):  # set the font and size
    return pg.font.Font("images/Aller_Rg.ttf", size)


def create_buttons():  # create button
    button_list.append(
        Button(
            800,
            20,
            120,
            60,
            "PLAY",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            925,
            20,
            120,
            60,
            "MAPS",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            800,
            90,
            120,
            60,
            "RANDOM",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            925,
            90,
            120,
            60,
            "NEW GAME",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            800,
            160,
            120,
            60,
            "SAVE",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            925,
            160,
            120,
            60,
            "LOAD",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            800,
            230,
            120,
            60,
            "ALGORITHMS",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            800,
            300,
            120,
            60,
            "LEVELS",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )
    button_list.append(
        Button(
            800,
            370,
            120,
            60,
            "RESET",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )

    button_list.append(
        Button(
            870,
            660,
            120,
            60,
            "QUIT",
            font=get_font(20),
            colour=Mindaro,
            tcolour=pg.Color("black"),
        )
    )

    for button in button_list:  # draw the buttons
        button.draw(screen2)


def play_button():  # this will check the algo, whether it is BFS, DFS, A* or DIJKSTRA, based on the global variable algo
    global algo, answer
    savegrid()
    if (sum(x.count(2) for x in grid)) == 1:
        if any(element == 2 for row in grid for element in row) and any(
            element == 3 for row in grid for element in row
        ):
            print("Solving")
            match algo:
                case "BFS":
                    bfs()
                case "DFS":
                    dfs()
                case "A*":
                    a_star()
                case "DIJKSTRA":
                    dijkstra()
                case "GREEDY":
                    greedy()
        else:
            answer = "Please choose start and end point"
    else:
        answer = "Please choose start and end point"


def maps_button():  # this will set the map for the game, every click is a new map
    global count_map, level
    count_map += 1
    if count_map > 5:  # 5 maps in total, reach 5 then return to 1
        count_map = 1
    loadgridWithLevel(count_map, level)


def rd_button():  # this will generate a map between 2 start point and end point
    global answer
    print("Creating and loading a random maze")
    try:
        generate_solvability_maze_with_user_points(
            gobalStartPoint[0], gobalStartPoint[1], gobalEndPoint[0], gobalEndPoint[1]
        )
        savegrid()  # Save the generated maze
        np.savetxt(r"./maze.txt", grid)  # Load the generated maze
    except:
        answer = "Please choose start and end point" # if user didn't set the start point and end point, announce them to do that


def ng_buttom():  # this will erase everything on the grid and set it to the initial grid
    global grid
    global gobalStartPoint, gobalEndPoint
    grid = [[0 for x in range(block)] for y in range(block)]
    savegrid()

    gobalStartPoint = None
    gobalEndPoint = None

def save_button():  # this will save the game
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_path:
        # Perform save operation here
        global grid
        np.savetxt(file_path, grid)
        print("File saved:", file_path)


def load_button():  # this will load the game
    file_path = filedialog.askopenfile(defaultextension=".txt")
    if file_path:
        global grid
        grid = np.genfromtxt(file_path, dtype=float, invalid_raise=False)
        grid = np.nan_to_num(grid, nan=0.0)  # Replace NaN values with 0.0
        grid = grid.astype(np.float64)
        np.savetxt("./maze.txt", grid)
        grid = np.loadtxt(r"./maze.txt").tolist()


def algorithms_button():  # this will set the algorithm for algo
    global count_algo, algo
    algorithms = ["", "BFS", "DFS", "A*", "DIJKSTRA", "GREEDY"]
    count_algo += 1
    if (
        count_algo > 5
    ):  # there are 5 algorithms, reach the last one will return to the first one
        count_algo = 1
    algo = algor
    
def levels_button():  # this will set the level
    global count_level, level, block, grid
    levels = ["EASY", "INTERMEDIATE", "HARD"]
    count_level += 1
    if (
        count_level > 2
    ):  # there are 4 algorithms, reach the last one will return to the first one
        count_level = 0
    level = levels[count_level]
    match level:
        case "EASY":
            block = 20
            grid = [[0 for x in range(block)] for y in range(block)]
            print_grid(grid)
        case "INTERMEDIATE":
            block = 27
            grid = [[0 for x in range(block)] for y in range(block)]
            print_grid(grid)
        case "HARD":
            block = 33
            grid = [[0 for x in range(block)] for y in range(block)]
            print_grid(grid)


def reset_button():  # reset no dang ko set duoc cai neu m nhan nut new game a
    global grid
    grid = np.loadtxt(r"./maze.txt").tolist()
    
def text_output():
    global algo, level, answer, path, node_visit
    uie_algo = UIE(925, 250, text=algo, font=get_font(20), colour=pg.Color("black"), width=200)
    uie_algo.draw(screen2)
    uie_level = UIE(925, 320, text=level, font=get_font(20), colour=pg.Color("black"), width=200)
    uie_level.draw(screen2)
    uie_answer = UIE(800, 480, text=answer, font=get_font(20), colour=pg.Color("black"), width=300)
    uie_answer.draw(screen2)
    uie_node = UIE(800, 510, text=node_visit, font=get_font(20), colour=pg.Color("black"), width=300)
    uie_node.draw(screen2)
    uie_path = UIE(800, 540, text=path, font=get_font(20), colour=pg.Color("black"), width=300)
    uie_path.draw(screen2) 

gobalStartPoint
gobalEndPoint

while not done:
    pos = pg.mouse.get_pos()
    x = pos[0]
    y = pos[1]
    screen1.fill(two)
    screen2.fill(two)

    # Blit the two screens onto the main display surface
    pg.display.get_surface().blit(screen1, (0, 0))
    pg.display.get_surface().blit(screen2, (screen1.get_width(), 0))
    # Create the buttons
    create_buttons()
    # Create the text
    text_output()
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        elif event.type == pg.MOUSEBUTTONDOWN:
            mp_x, mp_y = pg.mouse.get_pos()
            for button in button_list:
                if button.click(mp_x, mp_y):
                    match button.text:
                        case "PLAY":
                            if isReset == 0:
                                play_button()
                            else: 
                                answer = "Please click the reset button"
                                path = ""
                                node_visit = ""
                            isPlay = 1
                            break
                        case "MAPS":
                            isReset = 0
                            answer = ""
                            path = ""
                            node_visit = ""
                            maps_button()
                            break
                        case "RANDOM":
                            rd_button()
                            break
                        case "NEW GAME":
                            ng_buttom()
                            isReset = 0
                            answer = ""
                            path = ""
                            node_visit = ""
                            break
                        case "SAVE":
                            save_button()
                            break
                        case "LOAD":
                            load_button()
                            break
                        case "ALGORITHMS":
                            algorithms_button()
                            break
                        case "LEVELS":
                            levels_button()
                            break
                        case "RESET":
                            isReset = 0
                            if isPlay == 1:
                                reset_button()
                            answer = ""
                            path = ""
                            node_visit = ""
                            break
                        case "QUIT":
                            done = True
                            break
        
        if pg.mouse.get_pressed()[2]:
            column = pos[0] // (width + margin)
            row = pos[1] // (height + margin)
            if (sum(x.count(2) for x in grid)) < 1 or (
                sum(x.count(3) for x in grid)
            ) < 1:
                if (sum(x.count(2) for x in grid)) == 0:
                    if grid[row][column] == 2:
                        grid[row][column] = 0
                    elif grid[row][column] == 3:
                        grid[row][column] = 0
                    else:
                        grid[row][column] = 2
                        gobalStartPoint = (row, column)
                        

                else:
                    if grid[row][column] == 3:
                        grid[row][column] = 0
                    elif grid[row][column] == 2:
                        grid[row][column] = 0
                    else:
                        grid[row][column] = 3
                        gobalEndPoint = (row, column)

            else:
                if grid[row][column] == 2:
                    grid[row][column] = 0
                if grid[row][column] == 3:
                    grid[row][column] = 0
                if grid[row][column] == 1:
                    grid[row][column] = 0

        if pg.mouse.get_pressed()[0]:
            column = pos[0] // (width + margin)
            row = pos[1] // (height + margin)
            if row < 0 or row >= len(grid) or column < 0 or column >= len(grid[0]):
                continue
            else:
                print("left click")
                grid[row][column] = 1

    for row in range(block):
        for column in range(block):
            if grid[row][column] == 1:
                color = three
            elif grid[row][column] == 2:
                color = one
            elif grid[row][column] == 3:
                color = five
            elif grid[row][column] == 4:
                color = one
            elif grid[row][column] == 5:
                color = seven
            elif grid[row][column] == 6:
                color = seven
            else:
                color = four
            pg.draw.rect(
                screen1,
                color,
                [
                    margin + (margin + width) * column,
                    margin + (margin + height) * row,
                    width,
                    height,
                ],
      
    # screen.fill((0, 0, 0))
    pg.display.flip()
    clock.tick(60)
    pg.display.update()

pg.quit()
