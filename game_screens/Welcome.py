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

def get_font_2(size):
    return pg.font.Font("images/Asap-Regular.otf", size)

def render_and_blit(text, pos):
    TEXT = get_font_2(40).render(text, True, "Black")
    RECT = TEXT.get_rect(center=pos)
    SCREEN.blit(TEXT, RECT)

def play_button():
    subprocess.run(["python", "game_screens\Maze.py"])
    
def credit_button(): #how to play screen
    while True:
        OPTIONS_MOUSE_POS = pg.mouse.get_pos()

        SCREEN.fill("white")

        render_and_blit("Subject: Artificial Intelligent", (355, 50))
        render_and_blit("Project: Design and implement a", (355, 150))
        render_and_blit("Maze game using AI algorithms", (355, 200))
        render_and_blit("Group 8", (355, 300))
        render_and_blit("Nguyễn Văn Anh Đồng - 21110016", (355, 350))
        render_and_blit("Bùi Chiến Thắng - 21110798", (355, 400))
        render_and_blit("Võ Đăng Trình - 21110098", (355, 450))
        

        OPTIONS_BACK = Button(image=None, pos=(355, 650), 
                            text_input="BACK", font=get_font_2(45), base_color="Black", hovering_color="Green")

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
        
        MENU_TEXT = get_font(52).render("The Amazing World Of Mazes", True, '#6EEB83')
        MENU_RECT = MENU_TEXT.get_rect(center=(355, 150))
        
        PLAY_BUTTON = Button(image = pg.image.load("images/Play Rect.png"), pos=(355, 270), 
                            text_input="PLAY", font=get_font(50), base_color="#d7fcd4", hovering_color="#6699EE")
        CREDIT_BUTTON = Button(image = pg.image.load("images/Play Rect.png"), pos=(355, 420), 
                            text_input="CREDIT", font=get_font(50), base_color="#d7fcd4", hovering_color="#6699EE")
        QUIT_BUTTON = Button(image = pg.image.load("images/Quit Rect.png"), pos=(355, 570), 
                            text_input="QUIT", font=get_font(50), base_color="#d7fcd4", hovering_color="#6699EE")
        
        SCREEN.blit(MENU_TEXT, MENU_RECT)
        
        # Only call changeColor() on the button that the mouse is hovering over
        for button in [PLAY_BUTTON, CREDIT_BUTTON, QUIT_BUTTON]:
            if button.rect.collidepoint(MENU_MOUSE_POS):
                button.changeColor(MENU_MOUSE_POS)

        for button in [PLAY_BUTTON, CREDIT_BUTTON, QUIT_BUTTON]:
            button.update(SCREEN)
            
        for events in pg.event.get():
            if events.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if events.type == pg.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkforInput(MENU_MOUSE_POS):
                    play_button()
                    break
                if CREDIT_BUTTON.checkforInput(MENU_MOUSE_POS):
                    pg.display.set_caption("Credit srceen")
                    credit_button()
                    break
                if QUIT_BUTTON.checkforInput(MENU_MOUSE_POS):
                    pg.quit()
                    sys.exit()
        
        pg.display.update()
        
        
mainscreen()   
    