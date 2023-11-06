import pygame as pg
import numpy as np
import sys
from Button_main import Button
import subprocess

pg.init()

BG = pg.image.load("images/background.png")
SCREEN = pg.display.set_mode((706, 706))
pg.display.set_caption("Welcome srceen")

def get_font(size):
    return pg.font.Font("images/Debrosee-ALPnL.ttf", size)

def play_button():
    subprocess.run(["python", "game_screens\Maze.py"])
    
def howtoplay_button(): #how to play screen
    while True:
        OPTIONS_MOUSE_POS = pg.mouse.get_pos()

        SCREEN.fill("white")

        OPTIONS_TEXT = get_font(45).render("This is the How to play screen.", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(355, 260))
        SCREEN.blit(OPTIONS_TEXT, OPTIONS_RECT)

        OPTIONS_BACK = Button(image=None, pos=(355, 460), 
                            text_input="BACK", font=get_font(75), base_color="Black", hovering_color="Green")

        OPTIONS_BACK.changeColor(OPTIONS_MOUSE_POS)
        OPTIONS_BACK.update(SCREEN)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkforInput(OPTIONS_MOUSE_POS):
                    pg.display.set_caption("Welcome srceen")
                    mainscreen()

        pg.display.update()
        

def mainscreen(): #main menu screen
    while True:
        SCREEN.blit(BG, (0, 0))
        MENU_MOUSE_POS = pg.mouse.get_pos()
        
        MENU_TEXT = get_font(52).render("The Amazing World Of Mazes", True, '#B68F40')
        MENU_RECT = MENU_TEXT.get_rect(center=(355, 150))
        
        PLAY_BUTTON = Button(image = pg.image.load("images/Play Rect.png"), pos=(355, 270), 
                            text_input="PLAY", font=get_font(50), base_color="#d7fcd4", hovering_color="#6699EE")
        HOWTOPLAY_BUTTON = Button(image = pg.image.load("images/Options Rect.png"), pos=(355, 420), 
                            text_input="HOW TO PLAY", font=get_font(50), base_color="#d7fcd4", hovering_color="#6699EE")
        QUIT_BUTTON = Button(image = pg.image.load("images/Quit Rect.png"), pos=(355, 570), 
                            text_input="QUIT", font=get_font(50), base_color="#d7fcd4", hovering_color="#6699EE")
        
        SCREEN.blit(MENU_TEXT, MENU_RECT)
        
        # Only call changeColor() on the button that the mouse is hovering over
        for button in [PLAY_BUTTON, HOWTOPLAY_BUTTON, QUIT_BUTTON]:
            if button.rect.collidepoint(MENU_MOUSE_POS):
                button.changeColor(MENU_MOUSE_POS)

        for button in [PLAY_BUTTON, HOWTOPLAY_BUTTON, QUIT_BUTTON]:
            button.update(SCREEN)
            
        for events in pg.event.get():
            if events.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if events.type == pg.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkforInput(MENU_MOUSE_POS):
                    play_button()
                    break
                if HOWTOPLAY_BUTTON.checkforInput(MENU_MOUSE_POS):
                    pg.display.set_caption("How to play srceen")
                    howtoplay_button()
                    break
                if QUIT_BUTTON.checkforInput(MENU_MOUSE_POS):
                    pg.quit()
                    sys.exit()
        
        pg.display.update()
        
        
mainscreen()   
    