
#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import os
import kezmenu
import numpy as np

from tetrominoes import Tetromino, list_of_tetrominoes
from tetrominoes import rotate

from scores import load_score, write_score

class GameOver(Exception):
    """Exception used for its control flow properties"""

def get_sound(filename):
    return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22

LEFT_MARGIN = 340

WIDTH = MATRIX_WIDTH*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2 + LEFT_MARGIN
HEIGHT = (MATRIX_HEIGHT-2)*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

TRICKY_CENTERX = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2

screen = pygame.display.set_mode((WIDTH, HEIGHT))

class Matris(object):
    def __init__(self):
        self.surface = screen.subsurface(Rect((MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH),
                                              (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT-2) * BLOCKSIZE)))

        self.matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                self.matrix[(y,x)] = None
        """
        `self.matrix` is the current state of the tetris board, that is, it records which squares are
        currently occupied. It does not include the falling tetromino. The information relating to the
        falling tetromino is managed by `self.set_tetrominoes` instead. When the falling tetromino "dies",
        it will be placed in `self.matrix`.
        """
        self.bag= random.sample(list_of_tetrominoes, len(list_of_tetrominoes))
        self.next_tetromino = self.get_next_tetromino()
        self.held_tetromino = None

        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.8 # Move down every 800 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0



        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.holes = 0
        self.col_holes = 0
        self.bumpiness = 0
        self.lines_cleared_last_move = 0
        self.deepest_well = 0
        self.row_transisions = 0
        self.col_transisions = 0
        self.height = 0
        self.score_last = 0
        self.combo_last = 1
        self.num_pits = 0

        self.paused = False

        self.hold = False

        self.highscore = load_score()
        self.played_highscorebeaten_sound = False

        self.levelup_sound  = get_sound("levelup.wav")
        self.gameover_sound = get_sound("gameover.wav")
        self.linescleared_sound = get_sound("linecleared.wav")
        self.highscorebeaten_sound = get_sound("highscorebeaten.wav")


    def get_score(self):
        return self.score

    def set_tetrominoes(self):
        """
        Sets information for the current and next tetrominos
        """
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = self.get_next_tetromino()
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.surface_of_held_tetromino = self.construct_surface_of_held_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        # Disable shadow for now
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)
        self.hold = False

    def get_next_tetromino(self):
        tetromino = self.bag.pop()
        if len(self.bag) == 0:
            self.bag = random.sample(list_of_tetrominoes, len(list_of_tetrominoes))
        return tetromino
    def hard_drop(self,place_block= True):
        """
        Instantly places tetrominos in the cells below
        """
        amount = 0
        while self.request_movement('down'):
            amount += 1
        # No extra score for hard drop
        # self.score += 10*amount
        if place_block:
            self.lock_tetromino()
        else: 
            return self.lock_tetromino_GA()

    def hold_tetromino(self):
        if self.hold:
            pass
        elif self.held_tetromino is None:
            self.held_tetromino = self.current_tetromino
            self.next_tetromino = self.get_next_tetromino()
            self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        else:
            tetromino = self.held_tetromino
            self.held_tetromino = self.current_tetromino
            self.current_tetromino = tetromino

        self.surface_of_held_tetromino = self.construct_surface_of_held_tetromino()
        self.tetromino_position = (0, 4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        # Disable shadow for now
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)
        self.hold = True
        # And more!
        pass

    def update(self, timepassed):
        """
        Main game loop
        """
        self.needs_redraw = False
        
        pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
        unpressed = lambda key: event.type == pygame.KEYUP and event.key == key

        events = pygame.event.get()
        #Controls pausing and quitting the game.
        for event in events:
            if pressed(pygame.K_p):
                self.surface.fill((0,0,0))
                self.needs_redraw = True
                self.paused = not self.paused
            elif event.type == pygame.QUIT:
                self.gameover(full_exit=True)
            elif pressed(pygame.K_ESCAPE):
                self.gameover()

        if self.paused:
            return self.needs_redraw

        for event in events:
            #Controls movement of the tetromino
            if pressed(pygame.K_SPACE):
                self.hard_drop()
            elif pressed(pygame.K_h):
                self.hold_tetromino()
            elif pressed(pygame.K_UP) or pressed(pygame.K_w):
                self.request_rotation()
            elif pressed(pygame.K_LEFT) or pressed(pygame.K_a):
                self.request_movement('left')
                self.movement_keys['left'] = 1
            elif pressed(pygame.K_RIGHT) or pressed(pygame.K_d):
                self.request_movement('right')
                self.movement_keys['right'] = 1

            elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
                self.movement_keys['left'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2
            elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
                self.movement_keys['right'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2



        self.downwards_speed = (0.8-((self.level-1)*0.007))**(self.level-1)
        self.downwards_timer += timepassed
        downwards_speed = self.downwards_speed*0.10 if any([pygame.key.get_pressed()[pygame.K_DOWN],
                                                            pygame.key.get_pressed()[pygame.K_s]]) else self.downwards_speed
        
        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'): #Places tetromino if it cannot move further down
                self.lock_tetromino()

            self.downwards_timer %= downwards_speed


        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        if self.movement_keys_timer > self.movement_keys_speed:
            self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed
        
        return self.needs_redraw

    def draw_surface(self):
        """
        Draws the image of the current tetromino
        """
        with_tetromino = self.blend(matrix=self.place_shadow())

        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):

                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x*BLOCKSIZE, (y*BLOCKSIZE - 2*BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)
                    
    def gameover(self, full_exit=False):
        """
        Gameover occurs when a new tetromino does not fit after the old one has died, either
        after a "natural" drop or a hard drop by the player. That is why `self.lock_tetromino`
        is responsible for checking if it's game over.
        """

        write_score(self.score)
        if full_exit:
            exit()
        else:
            raise GameOver("Sucker!")

    def place_shadow(self):
        """
        Draws shadow of tetromino so player can see where it will be placed
        """
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY-1, posX)

        return self.blend(position=position, shadow=True)

    def fits_in_matrix(self, shape, position):
        """
        Checks if tetromino fits on the board
        """
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]: # outside matrix
                    return False

        return position
                    

    def request_rotation(self):
        """
        Checks if tetromino can rotate
        Returns the tetromino's rotation position if possible
        """
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ That's how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            
            self.needs_redraw = True
            return self.tetromino_rotation
        else:
            return False
            
    def request_movement(self, direction):
        """
        Checks if teteromino can move in the given direction and returns its new position if movement is possible
        """
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            self.tetromino_position = (posY, posX-1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        """
        Rotates tetromino
        """
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        """
        Sets visual information for tetromino
        """
        colors = {'blue':   (105, 105, 255),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}


        if shadow:
            end = [90] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c*0.5, colors[color])) + end)

        borderwidth = 2

        box = Surface((BLOCKSIZE-borderwidth*2, BLOCKSIZE-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color])) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))


        return border

    def lock_tetromino(self):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        """
        self.matrix = self.blend()

        lines_cleared = self.remove_lines()
        self.lines_cleared_last_move= lines_cleared
        self.lines += lines_cleared

        if lines_cleared:
            if lines_cleared >= 4:
                self.linescleared_sound.play()
            self.score += 100 * (lines_cleared**2) * self.combo

            if not self.played_highscorebeaten_sound and self.score > self.highscore:
                if self.highscore != 0:
                    self.highscorebeaten_sound.play()
                self.played_highscorebeaten_sound = True

        if self.lines >= self.level*5:
            self.levelup_sound.play()
            self.level += 1

        self.combo = self.combo + 1 if lines_cleared else 1

        self.set_tetrominoes()

        #States
        column_heights = self.get_column_heights()
        self.num_pits = np.count_nonzero(column_heights==0)
        self.row_transisions = self.get_row_transistions()
        self.col_transisions = self.get_col_transistions()
        self.height = np.sum(column_heights)
        self.holes, self.col_holes = self.get_holes()
        self.bumpiness = self.get_bumpiness(column_heights)
        self.deepest_well = self.get_depth_deepest_well(column_heights)

        if not self.blend():
            return self.gameover()
            
        self.needs_redraw = True

    def lock_tetromino_GA(self):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        """
        old = self.matrix
        self.matrix = self.blend()

        lines_cleared = self.remove_lines()

        # self.lines += lines_cleared
        
        if lines_cleared:
            self.score_last = 100 * (lines_cleared**2) * self.combo

        # Combo can be added again but its not official right now
        self.combo_last = self.combo + 1 if lines_cleared else 1


        self.set_current_tetromino(self.current_tetromino,self.next_tetromino)

        # States
        column_heights = self.get_column_heights()
        self.num_pits = np.count_nonzero(column_heights==0)
        self.holes, self.col_holes = self.get_holes()
        self.row_transisions = self.get_row_transistions()
        self.col_transisions = self.get_col_transistions()
        self.bumpiness = self.get_bumpiness(column_heights)
        self.deepest_well = self.get_depth_deepest_well(column_heights)
        self.lines_cleared_last_move= lines_cleared
        self.height = np.sum(column_heights)
            
        self.needs_redraw = False
        self.matrix = old
        return old

    def get_row_transistions(self):
        trans = 0
        for item in self.matrix.items():
            block = item[1]
            idx = item[0]
            # left
            if (idx[0],idx[1]-1) in self.matrix.keys() and not block and self.matrix[idx[0],idx[1]-1]:
                trans += 1
            # Right
            if (idx[0],idx[1]+1) in self.matrix.keys() and not block and self.matrix[idx[0],idx[1]+1] :
                trans +=1
        return trans
    
    def get_col_transistions(self):
        trans = 0
        for item in self.matrix.items():
            block = item[1]
            idx = item[0]
            # above
            if (idx[0]-1,idx[1]) in self.matrix.keys() and not block and self.matrix[idx[0]-1,idx[1]]:
                trans += 1
            # left
            if (idx[0]+1,idx[1]) in self.matrix.keys() and not block and self.matrix[idx[0]+1,idx[1]] :
                trans +=1
        return trans

    def get_holes(self):
        holes = 0
        col_with_holes = []
        for item in self.matrix.items():
            block = item[1]
            idx = item[0]
            # look items around
            item_around = 0
            if block is None:
                # above
                if (idx[0]-1,idx[1]) not in self.matrix.keys() or self.matrix[idx[0]-1,idx[1]]:
                    item_around += 1
                # below
                if (idx[0]+1,idx[1]) not in self.matrix.keys() or self.matrix[idx[0]+1,idx[1]]:
                    item_around += 1
                # left
                if (idx[0],idx[1]-1) not in self.matrix.keys() or self.matrix[idx[0],idx[1]-1]:
                    item_around += 1
                # right
                if (idx[0],idx[1]+1) not in self.matrix.keys() or self.matrix[idx[0],idx[1]+1]:
                    item_around += 1
                if item_around >=3 and (idx[0]-1,idx[1]) in self.matrix.keys() and self.matrix[idx[0]-1,idx[1]]:
                    holes+=1
                    col_with_holes.append(idx)
        return holes, len(set(col_with_holes))

    def remove_lines(self):
        """
        Removes lines from the board
        """
        lines = []
        for y in range(MATRIX_HEIGHT):
            #Checks if row if full, for each row
            line = (y, [])
            for x in range(MATRIX_WIDTH):
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == MATRIX_WIDTH:
                lines.append(y)

        for line in sorted(lines):
            #Moves lines down one row
            for x in range(MATRIX_WIDTH):
                self.matrix[(line,x)] = None
            for y in range(0, line+1)[::-1]:
                for x in range(MATRIX_WIDTH):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)

        return len(lines)

    def check_lines(self):
        """
        Removes lines from the board
        """
        lines = []
        for y in range(MATRIX_HEIGHT):
            #Checks if row if full, for each row
            line = (y, [])
            for x in range(MATRIX_WIDTH):
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == MATRIX_WIDTH:
                lines.append(y)
        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, shadow=False):
        """
        Does `shape` at `position` fit in `matrix`? If so, return a new copy of `matrix` where all
        the squares of `shape` have been placed in `matrix`. Otherwise, return False.
        
        This method is often used simply as a test, for example to see if an action by the player is valid.
        It is also used in `self.draw_surface` to paint the falling tetromino and its shadow on the screen.
        """
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if (copy.get((y, x), False) is False and shape[y-posY][x-posX] # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    copy.get((y,x)) and shape[y-posY][x-posX] and copy[(y,x)][0] != 'shadow'):

                    return False # Blend failed; `shape` at `position` breaks the matrix

                elif shape[y-posY][x-posX]:
                    copy[(y,x)] = ('shadow', self.shadow_block) if shadow else ('block', self.tetromino_block)

        return copy

    def construct_surface_of_next_tetromino(self):
        """
        Draws the image of the next tetromino
        """
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

    def construct_surface_of_held_tetromino(self):
        """
        Draws the image of the held tetromino   ---- change to making one function to construct surf and input of tetromino
        """
        if self.held_tetromino is None:
            shape = ""
        else:
            shape = self.held_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.held_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf


    def get_column_heights(self):
        height_array = np.zeros(MATRIX_WIDTH)
      
        for row_idx in range(MATRIX_HEIGHT):
            for col_idx in range(MATRIX_WIDTH):
                if self.matrix[(row_idx, col_idx)] is not None:
                    # If the MATRIX_HEIGHT - row_idx of a certain column is higher than the value in the height_array,
                    # then replace the value. Reasoning for MATRIX_HEIGHT - row_idx is,
                    # because in the matrix the bottom is 21 and runs back to 0 (0 = the top).
                    if height_array[col_idx] < MATRIX_HEIGHT - row_idx:
                        height_array[col_idx] = MATRIX_HEIGHT - row_idx

        return height_array

    def get_bumpiness(self, column_heights):
        bumpiness = 0
        for idx, _ in enumerate(column_heights):
            if idx+1 < len(column_heights)-1:
                bumpiness += np.absolute(column_heights[idx] - column_heights[idx+1])
        return bumpiness

    def get_same_height_indices(self, start_idx, next_idx, origin_idx, values, column_heights):
        if next_idx == len(column_heights): # This will give array out of bound index so here is an exit statement
            # If at the end and the origin (where we started is not the end), then something is the same till the end
            if start_idx != origin_idx:
                values.append([origin_idx, start_idx])
            else: 
                # Last column is a pillar/well but has a different height than the one left of it
                values.append([start_idx, start_idx])
            return values
        
        if column_heights[start_idx] == column_heights[next_idx]:
            # So this is forming a bottom
            # Check if the next one is also the same
            return self.get_same_height_indices(next_idx, next_idx+1, origin_idx, values, column_heights)
        elif column_heights[start_idx] != column_heights[next_idx]:
            # So if there now is height difference push the start_idx together with the orig_idx
            values.append([origin_idx, start_idx])
            return self.get_same_height_indices(next_idx, next_idx+1, next_idx, values, column_heights)

        return values

    def get_depth_of_wells(self, wells, column_heights):
        res = []
        for start_idx, end_idx in wells:
            # Get the left and right height --> take the min
            left_height = None
            right_height = None
            if start_idx == 0 and end_idx+1 < len(column_heights):
                # left there is nothing to check so only check the right height diff
                right_height = column_heights[end_idx+1]
            elif end_idx+1 >= len(column_heights):
                left_height = column_heights[start_idx-1]
            else:
                left_height = column_heights[start_idx-1]
                right_height = column_heights[end_idx+1]

            if left_height is None:
                minimum = right_height
            elif right_height is None:
                minimum = left_height
            else:
                minimum = min(left_height, right_height)   

            bottom_height = column_heights[start_idx]

            height_difference = minimum - bottom_height

            if height_difference > 0:
                res.append([start_idx, end_idx, height_difference])
            # else: 
                # Means that we are currently on a pillar and not in a well

        return res

    def get_depth_deepest_well(self, column_heights):
        """Returns the first and last column index of the DEEPEST well"""
        heights = self.get_same_height_indices(0, 1, 0, [], column_heights)
        wells = self.get_depth_of_wells(heights, column_heights)
        if wells:
            deepest_index = np.argmax([depth for start, end, depth in wells])
            return wells[deepest_index][2]
        else: return [0]


    def place_block(self, pos,rot,place_block=True):
        for _ in range(abs(rot)):
            self.request_rotation()
        if pos < 0:
            for _ in range(abs(pos)):
                self.request_movement('left')
        elif pos > 0:
            for _ in range(abs(pos)):
                self.request_movement('right')
        return self.hard_drop(place_block)
    
    def set_matrix(self,matrix):
        self.matrix = matrix
    
    def get_current_tetromino(self):
        return self.current_tetromino, self.next_tetromino
        
    def set_current_tetromino(self,curr,next):
        """
        Sets information for the current and next tetrominos
        """
        self.current_tetromino = curr
        self.next_tetromino = next
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)
    
    def get_state(self):
        # return [self.bumpiness,self.holes,self.lines_cleared_last_move, np.max(self.deepest_well),self.height, self.combo_last, self.score_last, self.num_pits]
        return [self.bumpiness,self.holes,self.lines_cleared_last_move, self.deepest_well,self.height, self.num_pits, self.row_transisions, self.col_transisions, self.col_holes]


class Game(object):
    def main(self):
        """
        Main loop for game
        Redraws scores and next tetromino each time the loop is passed through
        """
        clock = pygame.time.Clock()
        self.matris = Matris()
        
        screen.blit(construct_nightmare(screen.get_size()), (0,0))
        
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
        
        self.redraw()

        while True:
                try:
                    timepassed = clock.tick(50)
                    if self.matris.update((timepassed / 1000.) if not self.matris.paused else 0):
                        self.redraw()
                except GameOver:
                    return True 

    def redraw(self):
        """
        Redraws the information panel and next termoino panel
        """
        if not self.matris.paused:
            self.blit_next_tetromino(self.matris.surface_of_next_tetromino)
            self.blit_held_tetromino(self.matris.surface_of_held_tetromino)

            self.blit_info()

            self.matris.draw_surface()

        pygame.display.flip()


    def blit_info(self):
        """
        Draws information panel
        """
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 20 )
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf
        
        #Resizes side panel to allow for all information to be display there.
        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))
        holessurf = renderpair("Info", ' Holes: {}, Col_Holes: {}, Bumpiness: {}'.format(self.matris.holes,self.matris.col_holes,self.matris.bumpiness))
        wellsurf = renderpair("", ' Pits: {}, Deepest Well: {}, Height: {}'.format(self.matris.num_pits, self.matris.deepest_well,self.matris.height,self.matris.row_transisions, self.matris.col_transisions))
        infosurf = renderpair("",  'rol_trans: {}, col_trans: {}'.format(self.matris.row_transisions, self.matris.col_transisions))
        height = 20 + (levelsurf.get_rect().height + 
                            scoresurf.get_rect().height +
                            linessurf.get_rect().height + 
                            combosurf.get_rect().height +
                            holessurf.get_rect().height +
                            wellsurf.get_rect().height+
                            infosurf.get_rect().height
                            )
        
        #Colours side panel
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))
        
        #Draws side panel
        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))
        area.blit(holessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height))
        area.blit(wellsurf, (0,  levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height+ wellsurf.get_rect().height))
        area.blit(infosurf, (0,  levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height+ wellsurf.get_rect().height+  infosurf.get_rect().height))
        screen.blit(area, area.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=TRICKY_CENTERX))


    def blit_next_tetromino(self, tetromino_surf):
        """
        Draws the next tetromino in a box to the side of the board
        """
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 20)
        width = (WIDTH - (MATRIS_OFFSET + BLOCKSIZE * MATRIX_WIDTH + BORDERWIDTH * 2)) - MATRIS_OFFSET * 2

        def txt(text):
            text = font.render(text, True, textcolor)
            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)
            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            return surf

        title = txt("Next")
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(title, (0,BLOCKSIZE*3))
        area.blit(tetromino_surf, (center, center))

        screen.blit(area, area.get_rect(top=MATRIS_OFFSET, centerx=TRICKY_CENTERX))

    def blit_held_tetromino(self, tetromino_surf):
        """
        Draws the next tetromino in a box to the side of the board
        """
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 20)
        width = (WIDTH - (MATRIS_OFFSET + BLOCKSIZE * MATRIX_WIDTH + BORDERWIDTH * 2)) - MATRIS_OFFSET * 2

        def txt(text):
            text = font.render(text, True, textcolor)
            surf = Surface((width, text.get_rect().height + BORDERWIDTH * 2), pygame.SRCALPHA, 32)
            surf.blit(text, text.get_rect(top=BORDERWIDTH + 10, left=BORDERWIDTH + 10))
            return surf

        title = txt("Hold (h)")

        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(title, (0,BLOCKSIZE*3))
        area.blit(tetromino_surf, (center, center))

        screen.blit(area, area.get_rect(top=2* MATRIS_OFFSET+areasize, centerx=TRICKY_CENTERX))

class Menu(object):
    """
    Creates main menu
    """
    running = True
    def main(self, screen):
        clock = pygame.time.Clock()
        menu = kezmenu.KezMenu(
            ['Play!', lambda: Game().main(screen)],
            ['Quit', lambda: setattr(self, 'running', False)],
        )
        menu.position = (50, 50)
        menu.enableEffect('enlarge-font-on-focus', font=None, size=60, enlarge_factor=1.2, enlarge_time=0.3)
        menu.color = (255,255,255)
        menu.focus_color = (40, 200, 40)
        
        nightmare = construct_nightmare(screen.get_size())
        highscoresurf = self.construct_highscoresurf() #Loads highscore onto menu

        timepassed = clock.tick(30) / 1000.

        while self.running:
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    exit()

            menu.update(events, timepassed)

            timepassed = clock.tick(30) / 1000.

            if timepassed > 1: # A game has most likely been played 
                highscoresurf = self.construct_highscoresurf()

            screen.blit(nightmare, (0,0))
            screen.blit(highscoresurf, highscoresurf.get_rect(right=WIDTH-50, bottom=HEIGHT-50))
            menu.draw(screen)
            pygame.display.flip()

    def construct_highscoresurf(self):
        """
        Loads high score from file
        """
        font = pygame.font.Font(None, 50)
        highscore = load_score()
        text = "Highscore: {}".format(highscore)
        return font.render(text, True, (255,255,255))

def construct_nightmare(size):
    """
    Constructs background image
    """
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in range(0, len(arr), boxsize):
        for y in range(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in range(x, x+(boxsize - bordersize)):
                for LY in range(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf


class GameGA(Game):
    def main(self, user,info):
        """
        Main loop for game
        Redraws scores and next tetromino each time the loop is passed through
        """
        clock = pygame.time.Clock()
        self.matris = Matris()
        self.user = user
        self.info = info

        screen.blit(construct_nightmare(screen.get_size()), (0,0))
        
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
        
        self.redraw()
        while True:
            try:   
                self.matris.update(9999999999999)
                scores = []
                positions = []
                rotations = []
                curr, next = self.matris.get_current_tetromino()
                pos_left, pos_right = self.get_possible_pos(curr)
                rot_range = self.get_possible_rot(curr)
                for rot in rot_range:
                    for pos in range(pos_left,pos_right+1):
                        self.matris.place_block(pos,rot,False)
                        state = self.matris.get_state()
                        scores.append((np.sum(state*user)))
                        positions.append(pos)
                        rotations.append(rot)
                max_value = max(scores)
                max_index = scores.index(max_value)
                best_pos = positions[max_index]
                best_rot = rotations[max_index]
                self.matris.place_block(best_pos,best_rot,True)
                self.redraw()
            except GameOver:
                return self.matris.get_score()

    def get_possible_pos(self, tetr):
        color = tetr.color
        # Z and S
        if color == 'red' or  color == 'green':
            return  -4,5
        # 0
        if color == 'yellow':
            return -4, 4
        # I
        if color == 'blue':
            return -5, 4
        #  L and J and T
        return -4, 5

    def get_possible_rot(self, tetr):
        color = tetr.color
        # Z and S and I
        if color == 'red' or  color == 'green' or color == 'blue':
            return [0,1]
        if color == 'yellow':
            return [0]
        # L, J
        if color == 'cyan' or color == 'orange':
            return [0,1,2]
        #  T
        return [0,1,2,3]

    def blit_info(self):
        """
        Draws information panel
        """
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 20)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf
        
        #Resizes side panel to allow for all information to be display there.
        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))
        holessurf = renderpair("Info", ' Holes: {}, Col_Holes: {}, Bumpiness: {}'.format(self.matris.holes,self.matris.col_holes,self.matris.bumpiness))
        wellsurf = renderpair("", ' Pits: {}, Deepest Well: {}, Height: {}'.format(self.matris.num_pits, self.matris.deepest_well,self.matris.height,self.matris.row_transisions, self.matris.col_transisions))
        infosurf = renderpair("",  'rol_trans: {}, col_trans: {}'.format(self.matris.row_transisions, self.matris.col_transisions))
        mutationsurf = renderpair("Info","Generation: {}/{}, Mutation: {}/{} ({}/{})".format(self.info[0],self.info[1],self.info[2],self.info[3],self.info[4],self.info[5]))

        height = 20 + (levelsurf.get_rect().height + 
                    scoresurf.get_rect().height +
                    linessurf.get_rect().height + 
                    combosurf.get_rect().height +
                    holessurf.get_rect().height +
                    wellsurf.get_rect().height+
                    infosurf.get_rect().height+
                    mutationsurf.get_rect().height
                    )
        
        #Colours side panel
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))
        
        #Draws side panel
        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))
        area.blit(holessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height))
        area.blit(wellsurf, (0,  levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height+ wellsurf.get_rect().height))
        area.blit(infosurf, (0,  levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height+ wellsurf.get_rect().height+  infosurf.get_rect().height))
        area.blit(mutationsurf, (0, infosurf.get_rect().height+ mutationsurf.get_rect().height + wellsurf.get_rect().height + levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height +holessurf.get_rect().height))
        screen.blit(area, area.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=TRICKY_CENTERX))
        




def start_game():
    pygame.init()
    pygame.display.set_caption("MaTris")


def start_round():
    Game().main()


def start_round_GA(user,info):
    return GameGA().main(user,info)
  

if __name__ == '__main__':
    start_game()
    start_round()



