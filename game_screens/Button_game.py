import pygame as pg

class Button:
    def __init__(self, x, y, width, height, text, font, colour, tcolour):
        self.colour, self.tcolour = colour, tcolour
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.text = text
        self.font = font
        
    def draw(self, screen):
        pg.draw.rect(screen, self.colour, (self.x, self.y, self.width, self.height))
        font = self.font
        self.font_size = font.size(self.text)
        text = font.render(self.text, True, self.tcolour)
        position_x = self.x + (self.width / 2) - self.font_size[0] /2
        position_y = self.y + (self.height / 2) - self.font_size[1] /2
        screen.blit(text, (position_x, position_y))
        
    def click(self, mouce_x, mouce_y):
        return self.x <= mouce_x <= self.x + self.width and self.y <= mouce_y <= self.y + self.height