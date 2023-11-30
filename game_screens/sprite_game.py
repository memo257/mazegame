import pygame as pg

class Button: #design the button
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
    
class UIE: #design the text 
    def __init__(self, x, y, text, font, colour, width):
        self.x, self.y = x, y
        self.text = text
        self.font = font
        self.colour = colour
        self.width = width

    def wrap_text(self):
        words = self.text.split(' ')
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width, _ = self.font.size(word + ' ')
            if current_width + word_width <= self.width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width

        lines.append(' '.join(current_line))
        return lines

    def draw(self, screen):
        lines = self.wrap_text()
        for i, line in enumerate(lines):
            text = self.font.render(line, True, self.colour)
            screen.blit(text, (self.x, self.y + i * self.font.get_height()))
        
class Player: #setup movement for player
    def __init__(self, start_pos):
        self.row, self.col = start_pos

    def move(self, direction, maze):
        new_row, new_col = self.row, self.col

        if direction == "UP" and self.row > 0 and maze[self.row - 1][self.col] != 1:
            new_row -= 1
        elif (
            direction == "DOWN"
            and self.row < len(maze) - 1
            and maze[self.row + 1][self.col] != 1
        ):
            new_row += 1
        elif direction == "LEFT" and self.col > 0 and maze[self.row][self.col - 1] != 1:
            new_col -= 1
        elif (
            direction == "RIGHT"
            and self.col < len(maze[0]) - 1
            and maze[self.row][self.col + 1] != 1
        ):
            new_col += 1

        # Update the player's position
        self.row, self.col = new_row, new_col

    def get_position(self):
        return self.row, self.col

    def has_reached_end(self, end_pos):
        return (self.row, self.col) == end_pos