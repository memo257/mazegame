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
from Button_game import Button
from tkinter import filedialog

#colors
one = (79, 189, 186)
two = (206, 171, 147)
three = (227, 202, 165)
four = (255, 251, 233)
five = (246, 137, 137)
six = (255, 0, 0)
seven = (0, 255, 0)
Mindaro = '#C5D86D'
Black = '#fff'

pg.init()

#set size of each screen, screen 1 is the main screen, screen 2 is the button screen
size = (1126, 729)
screen1_size = (size[0] - 420, size[1])
screen2_size = (size[0] - 706, size[1])
screen1 = pg.display.set_mode(screen1_size)
screen2 = pg.display.set_mode(screen2_size)
screen = pg.display.set_mode(size, pg.RESIZABLE)

width, height = screen.get_size()
mouse_pos = pg.mouse.get_pos()

pg.display.set_caption("MAZE")

width = 20
height = 20
margin = 2
count = 0

grid = [[0 for x in range(33)] for y in range(33)]
gobalStartPoint = None
gobalEndPoint = None

done = False
clock = pg.time.Clock()
found = False
neighbour=[]
button_list = []

def savegrid():
    global grid
    
    np.savetxt(r"./maze.txt",grid)
def loadgrid(index):
    global grid
    if(index ==0):
        grid = np.loadtxt(r"./maingame/maze.txt").tolist()
    elif(index ==1):
        grid = np.loadtxt(r'./mazemap/Maze1/maze.txt').tolist()
    elif(index ==2):
        grid = np.loadtxt(r'./mazemap/Maze2/maze.txt').tolist()
    elif(index ==3):
        grid = np.loadtxt(r'./mazemap/Maze3/maze.txt').tolist()
    elif(index ==4):
        grid = np.loadtxt(r'./mazemap/Maze4/maze.txt').tolist()
    elif(index ==5):
        grid = np.loadtxt(r'./mazemap/Maze5/maze.txt').tolist()
        
def bfs_shortestpath(maze, path=""):
    global grid
    i,j=startp(maze,0,0)
    pos = set()
    for move in path:
        if move == "L":
            i -= 1

        elif move == "R":
            i += 1

        elif move == "U":
            j -= 1

        elif move == "D":
            j += 1
        pos.add((j, i))
    
    for j, row in enumerate(maze):
        for i, col in enumerate(row):
            if (j, i) in pos:
                
                grid[j][i] = 4
                
def startp(maze,i,j):
    for x in range(len(maze[0])):
        try:
            i =(maze[x].index(2))
            j = x
            print(j)
            return i,j
        except:
            pass

def bfs():
    global grid, neighbour
    neighbourr()

    start, end = S_E(grid, 0, 0)

    open_set = Queue()
    open_set.put(start)
    came_from = {}
    visited = set()

    while not open_set.empty():
        current = open_set.get()

        if current == end:
            print("Finishing")
            short_path(came_from, end)
            return True

        visited.add(current)

        for nei in neighbour[current[0] * len(grid[0]) + current[1]]:
            if nei not in visited:
                open_set.put(nei)
                came_from[nei] = current

    return False

def dfs():
    global grid, neighbour
    neighbourr()

    start, end = S_E(grid, 0, 0)

    stack = [start]
    came_from = {}
    visited = set()

    while stack:
        current = stack[-1]

        if current == end:
            print("Finishing - solve by dfs")
            short_path(came_from, end)
            return True

        if current not in visited:
            visited.add(current)
            # Simulation code to visualize the DFS process
            grid[current[0]][current[1]] = 5
            print_grid(grid)  # Function to display the grid (customize as needed)
            time.sleep(0.05)  # Add a delay to make it slower

        unvisited_neighbors = [nei for nei in neighbour[current[0] * len(grid[0]) + current[1]] if nei not in visited]

        if unvisited_neighbors:
            next_neighbor = unvisited_neighbors[0]
            stack.append(next_neighbor)
            came_from[next_neighbor] = current
        else:
            # No unvisited neighbors, backtrack
            stack.pop()
    
    return False


def dijkstra():
    global grid, neighbour
    neighbourr()

    start, end = S_E(grid, 0, 0)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current not in visited:
            visited.add(current)
            grid[current[0]][current[1]] = 5  # Mark visited node
            print_grid(grid)  # Function to display the grid (customize as needed)
            time.sleep(0.1)  # Add a delay to make it slower

        if current == end:
            print("Finishing - solve by Dijkstra ")
            short_path(came_from, end)
            simulate_dijkstra_process(came_from, start, end, visited)
            return True

        for nei in neighbour[current[0] * len(grid[0]) + current[1]]:
            new_cost = cost_so_far[current] + 1
            if nei not in cost_so_far or new_cost < cost_so_far[nei]:
                cost_so_far[nei] = new_cost
                priority = new_cost
                heapq.heappush(open_set, (priority, nei))
                came_from[nei] = current

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
            grid[node[0]][node[1]] = 2  # Mark start node
        elif node == end:
            grid[node[0]][node[1]] = 3  # Mark end node
        elif node in path:
            grid[node[0]][node[1]] = 4  # Mark solution path nodes
        else:
            grid[node[0]][node[1]] = 6  # Mark other visited nodes

    print("Solution Path:")
    print(path)


def print_grid(grid):
    global screen1
    global width, height, margin

    screen1.fill(two)

    for row in range(33):
        for column in range(33):
            if grid[row][column] == 1:
                color = three
            elif grid[row][column] == 2:
                color = one
            elif grid[row][column] == 3:
                color = five
            elif grid[row][column] == 4:
                color = one
            elif grid[row][column] == 5:
                color = six
            elif grid[row][column] == 6:
                color = seven
            else:
                color = four

            pg.draw.rect(screen1, color, [margin + (margin + width) * column, margin + (margin + height) * row, width, height])

    pg.display.flip()

def neighbourr():
    global grid,neighbour
    neighbour = [[]for col in range(len(grid)) for row in range(len(grid))]
    count=0
    for i in range(len(grid)):
        for j in range(len(grid)):
            neighbour[count] == []
            if (i > 0 and grid[i - 1][j] != 1):
                neighbour[count].append((i-1,j))
            if (j > 0 and grid[i][j - 1] != 1):
                neighbour[count].append((i,j-1))
            if (i < len(grid) - 1 and grid[i + 1][j] != 1):
                neighbour[count].append((i+1,j))
            if (j < len(grid) - 1 and grid[i][j + 1] != 1):
                neighbour[count].append((i,j+1))
            count+=1
            
def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

def S_E(maze,start,end):
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if(grid[x][y]==2):
                start =x,y
            if(grid[x][y]==3):
                end =x,y
       
    return start,end

def short_path(came_from, current):
     grid[current[0]][current[1]] = 4
     while current in came_from:
         current = came_from[current]
         grid[current[0]][current[1]] = 4
        
def a_star():
    global grid, neighbour
    neighbourr()

    start,end = S_E(grid,0,0)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    open_set_his = {start}
    came_from = {}
    
    g_score = [float("inf") for row in grid for spot in row ]
    g_score[start[0]*len(grid[0]) +start[1]] = 0
    f_score = [ float("inf") for row in grid for spot in row ]
    f_score[start[0]*len(grid[0]) +start[1]] = h(start, end)
    
    while not open_set.empty():
        current = open_set.get()[2]
        open_set_his.remove(current)
        if current == end:
            print("finishing")
            short_path(came_from, end)
            return True
        for nei in neighbour[current[0]*len(grid[0]) +current[1]]:
            temp_g_score = g_score[current[0]*len(grid[0]) +current[1]] + 1
            if temp_g_score < g_score[nei[0]*len(grid[0]) +nei[1]]:
                came_from[nei] = current
                g_score[nei[0]*len(grid[0]) +nei[1]] = temp_g_score
                f_score[nei[0]*len(grid[0]) +nei[1]] = temp_g_score + h(nei, end)
                if nei not in open_set_his:
                    count += 1
                    open_set.put((f_score[nei[0]*len(grid[0]) +nei[1]], count, nei))
                    open_set_his.add(nei)
    return False

def generate_solvability_maze_with_user_points(start_x, start_y, end_x, end_y):
    global grid

    def is_valid(x, y):
        return 0 <= x < 32 and 0 <= y < 32

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
    grid = [[1 for _ in range(33)] for _ in range(33)]

    # Make sure the start and end points are valid
    start_x, start_y = max(1, min(start_x, 31)), max(1, min(start_y, 31))
    end_x, end_y = max(1, min(end_x, 31)), max(1, min(end_y, 31))

    # Generate the maze
    create_maze(start_x, start_y)

    # Mark the start and end points

    grid[start_x][start_y] = 2
    grid[end_x][end_y] = 3

    # grid[start_x][start_y] = 2
    # grid[end_y][end_x] = 3

    return grid

def get_font(size):
    return pg.font.Font("images/Debrosee-ALPnL.ttf", size)

def create_buttons():
    button_list.append(Button(800, 50, 120, 60, "PLAY", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    button_list.append(Button(800, 120, 120, 60, "CHANGE MAP", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    button_list.append(Button(800, 190, 120, 60, "RANDOM", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    button_list.append(Button(800, 260, 120, 60, "NEW GAME", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    button_list.append(Button(800, 330, 120, 60, "SAVE", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    button_list.append(Button(800, 400, 120, 60, "LOAD", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    button_list.append(Button(800, 600, 120, 60, "QUIT", font=get_font(20), colour=Mindaro, tcolour=pg.Color("black")))
    

    for button in button_list:
        button.draw(screen2)

def play_button():
    if((sum(x.count(2) for x in grid)) == 1):
        print("Solving")
        #bfs()
        #a_star()
        #dfs()
        #dfs_simulation()
        dijkstra()

def cm_button():
    global count
    count += 1
    if count > 5:
        count = 1
    loadgrid(count)
    
def rd_button():
    print("Creating and loading a random maze")
    try:
        generate_solvability_maze_with_user_points(gobalStartPoint[0], gobalStartPoint[1], gobalEndPoint[0], gobalEndPoint[1])
        savegrid()  # Save the generated maze
        np.savetxt(r"./maze.txt",grid) # Load the generated maze
    except:
        print("Please choose start and end point")
    
def ng_buttom():
    global grid
    grid = [[0 for x in range(33)] for y in range(33)]
    
def save_button():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_path:
        # Perform save operation here
        global grid
        np.savetxt(file_path, grid)
        print("File saved:", file_path)
        
def load_button():
    file_path = filedialog.askopenfile(defaultextension=".txt")
    if file_path:
        global grid
        grid = np.genfromtxt(file_path, dtype=float, invalid_raise=False)
        grid = np.nan_to_num(grid, nan=0.0)  # Replace NaN values with 0.0
        grid = grid.astype(np.float64)

while not done:
    gobalStartPoint 
    gobalEndPoint 
    
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

    for event in pg.event.get(): 
        if event.type == pg.QUIT:
            done = True
            
        elif event.type == pg.KEYDOWN:
            '''if event.key == pg.K_ESCAPE:
                print("Exit")
                pg.quit()
            if event.key == pg.K_s:
                print("Saving Maze")
                savegrid()
            if event.key == pg.K_l:
                print("Loading Maze")
                loadgrid(0)'''
            if event.key == pg.K_f:
                print("Filling Maze")
                grid = [[1 for x in range(33)] for y in range(33)]
        
        elif event.type == pg.MOUSEBUTTONDOWN:
            mp_x, mp_y = pg.mouse.get_pos()
            for button in button_list:
                if button.click(mp_x, mp_y):
                    match button.text:
                        case "PLAY":
                            play_button()
                            break
                        case "CHANGE MAP":
                            cm_button()  
                            break
                        case "RANDOM":
                            rd_button()     
                            break  
                        case "NEW GAME":
                            ng_buttom()
                            break
                        case "SAVE":
                            save_button()
                            break
                        case "LOAD":
                            load_button()
                            break
                        case "QUIT":
                            print("Exit")
                            pg.quit()
                            break
                
        if pg.mouse.get_pressed()[2]:
            column = pos[0] // (width + margin)
            row = pos[1] // (height + margin)
            if((sum(x.count(2) for x in grid)) < 1 or (sum(x.count(3) for x in grid)) < 1):
                if((sum(x.count(2) for x in grid)) == 0):
                    if(grid[row][column] == 2):
                        grid[row][column] = 0
                    elif(grid[row][column] == 3):
                        grid[row][column] = 0
                    else:
                        grid[row][column]  = 2
                        gobalStartPoint = (row, column)
                else:
                    if(grid[row][column] == 3):
                        grid[row][column] = 0
                    elif(grid[row][column] == 2):
                        grid[row][column] = 0
                    else:
                        grid[row][column]  = 3
                        gobalEndPoint = (row, column)

            else:
                if(grid[row][column] == 2):
                    grid[row][column] = 0
                if(grid[row][column] == 3):
                    grid[row][column] = 0
                if(grid[row][column] == 1):
                    grid[row][column] = 0
                    
        if pg.mouse.get_pressed()[0]:
            column = pos[0] // (width + margin)
            row = pos[1] // (height + margin)
            if row < 0 or row >= len(grid) or column < 0 or column >= len(grid[0]):
                continue
            else: 
                print("left click")
                grid[row][column] = 1
    
    for row in range(33):
        for column in range(33):
            if grid[row][column] == 1:
                color = three
            elif grid[row][column] == 2:
                color = one
            elif grid[row][column] == 3:
                color = five
            elif grid[row][column] == 4:
                color = one
            elif grid[row][column] == 5:
                color = six
            elif grid[row][column] == 6:
                color = seven
            else:
                color = four
            pg.draw.rect(screen1, color, [margin + (margin + width) * column, margin + (margin + height) * row, width, height])
    pg.display.flip()
    clock.tick(60)
    pg.display.update()
pg.quit()