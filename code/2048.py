"""
Classic 2048 — Pygame Edition (original style remix)
Author(s): Kalen Luraschi, (add names here when worked on)

Arrow keys = move tiles
Press ESC to quit
"""

import pygame
import random
import math
from copy import deepcopy

pygame.init()
FPS = 60
SIZE = 4
WIDTH, HEIGHT = 600, 600
TILE_SIZE = WIDTH // SIZE
BG_COLOR = (205, 192, 180)
GRID_COLOR = (187, 173, 160)
FONT_COLOR = (119, 110, 101)
FONT = pygame.font.SysFont("comicsans", 55, bold=True)
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048 Classic")

class Tile:
    COLORS = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    }

    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.x = col * TILE_SIZE
        self.y = row * TILE_SIZE

    def draw(self, win):
        rect = (self.x, self.y, TILE_SIZE, TILE_SIZE)
        color = self.COLORS.get(self.value, (60, 58, 50))
        pygame.draw.rect(win, color, rect)
        if self.value != 0:
            text = FONT.render(str(self.value), True, FONT_COLOR if self.value < 8 else (255, 255, 255))
            win.blit(
                text,
                (
                    self.x + TILE_SIZE / 2 - text.get_width() / 2,
                    self.y + TILE_SIZE / 2 - text.get_height() / 2,
                ),
            )

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.grid = [[0]*size for _ in range(size)]
        self.spawn_tile()
        self.spawn_tile()
        self.score = 0

    def spawn_tile(self):
        empty = [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r][c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        self.grid[r][c] = random.choices([2, 4], weights=[0.9, 0.1])[0]

    def compress(self, row):
        """Slide all non-zero elements left"""
        new_row = [v for v in row if v != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def merge(self, row):
        """Merge equal adjacent tiles"""
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0
        return row

    def move_left(self):
        moved = False
        new_grid = []
        for row in self.grid:
            compressed = self.compress(row)
            merged = self.merge(compressed)
            final = self.compress(merged)
            if final != row:
                moved = True
            new_grid.append(final)
        if moved:
            self.grid = new_grid
            self.spawn_tile()
        return moved

    def rotate_clockwise(self):
        self.grid = [list(row) for row in zip(*self.grid[::-1])]

    def move(self, direction):
        # 2=up,1=right,0=down,3=left
        rotations = {0: 1, 1: 2, 2: 3, 3: 0}
        times = rotations[direction]
        for _ in range(times):
            self.rotate_clockwise()
        moved = self.move_left()
        for _ in range((4 - times) % 4):
            self.rotate_clockwise()
        return moved

    def is_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        for r in range(self.size):
            for c in range(self.size):
                v = self.grid[r][c]
                for dr, dc in [(1, 0), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if rr < self.size and cc < self.size and self.grid[rr][cc] == v:
                        return False
        return True
    
    #####################################################################
    #                    minmax functions below                         #
    #####################################################################
    
    def clone(self):
        g = Game2048(self.size)
        g.grid = [row[:] for row in self.grid]
        g.score = self.score
        return g
    
    # for when minmax has to check the future branches
    def apply_move_no_spawn(self, direction):
        grid_copy = [row[:] for row in self.grid]
        score_before = self.score

        rotations = {0: 1, 1: 2, 2: 3, 3: 0}
        times = rotations[direction]
        for _ in range(times):
            grid_copy = [list(row) for row in zip(*grid_copy[::-1])]

        moved = False
        new_grid = []
        temp_score = 0

        for row in grid_copy:
            new_row = [v for v in row if v != 0]
            new_row += [0] * (self.size - len(new_row))

            for i in range(self.size - 1):
                if new_row[i] != 0 and new_row[i] == new_row[i + 1]:
                    new_row[i] *= 2
                    temp_score += new_row[i]
                    new_row[i + 1] = 0

            new_row = [v for v in new_row if v != 0]
            new_row += [0] * (self.size - len(new_row))
            new_grid.append(new_row)

        for _ in range((4 - times) % 4):
            new_grid = [list(row) for row in zip(*new_grid[::-1])]

        if new_grid != self.grid:
            moved = True

        return moved, new_grid, temp_score, score_before

    
def draw_grid(win, game):
    win.fill(BG_COLOR)
    for r in range(game.size):
        for c in range(game.size):
            val = game.grid[r][c]
            tile = Tile(val, r, c)
            tile.draw(win)
    for i in range(1, game.size):
        pygame.draw.line(win, GRID_COLOR, (0, i * TILE_SIZE), (WIDTH, i * TILE_SIZE), 6)
        pygame.draw.line(win, GRID_COLOR, (i * TILE_SIZE, 0), (i * TILE_SIZE, HEIGHT), 6)
    pygame.display.update()

def draw_grid(win, game):
    win.fill(BG_COLOR)

    for r in range(game.size):
        for c in range(game.size):
            val = game.grid[r][c]
            tile = Tile(val, r, c)
            tile.draw(win)

    for i in range(1, game.size):
        pygame.draw.line(win, GRID_COLOR, (0, i * TILE_SIZE), (WIDTH, i * TILE_SIZE), 6)
        pygame.draw.line(win, GRID_COLOR, (i * TILE_SIZE, 0), (i * TILE_SIZE, HEIGHT), 6)

    score_font = pygame.font.SysFont("comicsans", 36, bold=True)
    score_text = score_font.render(f"Score: {game.score}", True, (80, 70, 60))
    win.blit(score_text, (15, 10))

    pygame.display.update()

##############################################################################
class minmax_agent:
    """
    first we get the root which is the current game state
        then we have our 4 directions we can move the board
        we move in all directions to get our child branches (4) with the new move
            for each direction we can move, we place a 2 and a 4(for now just 2) in each spot(if spot is open) and take the average h of all possible random spawns
        we then choose the best h from the 4 directions
    """
    def __init__(self, depth=3):
        self.depth = depth
    
    def act_random(self, game):
        """Returns a random move direction"""
        valid_moves = []

        #for each direction
        for move in range(4):
            moved, new_grid, gained_score, old_score = game.apply_move_no_spawn(move)
            if moved:
                valid_moves.append(move)

        if not valid_moves:
            return None
            
        return random.choice(valid_moves)
    
    def evaluate_h_v1(self, game):
        """h: prefer high score, more empty spaces, and big tiles."""
        empty_tiles = sum(row.count(0) for row in game.grid)
        max_tile = max(max(row) for row in game.grid)
        return game.score + empty_tiles * 10 + max_tile / 4
    
    def act_simple(self, game):
        """depth of one, picks best h (evaluate_h_v1)"""
        best_move = None
        best_score = -float('inf')
        for move in range(4):  
            moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
            if moved:
                temp = game.clone()
                temp.grid = new_grid
                temp.score += gained_score
                value = self.evaluate_h_v1(temp)

                if value > best_score:
                    best_score = value
                    best_move = move
        return best_move
    
    def act_simple_minmax(self, game, depth=2):
        """first minmax try, picks best h (evaluate_h_v1)"""
        best_move = None
        best_score = -float('inf')

        for move in range(4):
            #for first 4 options
            moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
            if not moved:
                continue
        
            temp = game.clone()
            temp.grid = new_grid
            temp.score += gained_score

            #go down a branch
            value = self.minmax(temp, depth -1, False)

            if value > best_score:
                best_score = value
                best_move = move

        return best_move
    
    def minmax(self, game, depth, maximizing):
        if depth == 0 or game.is_game_over():
            return game.score
        
        if maximizing:
            #starts this branch by doing possible 4 move
            best_score = -float('inf')
            for move in range(4):
                moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
                if not moved:
                    continue
            
                temp = game.clone()
                temp.grid = new_grid
                temp.score += gained_score

                value = self.minmax(temp, depth -1, False)

                if value > best_score:
                    best_score = value
                
            return best_score if best_score != -float('inf') else game.score
        else:
            #starts (supposed to) this branch by creating an instance of every possible random spot filled by a 2 or 4
            #next if idk
            worst_score = float('inf')
            #implement part that calculates random spawns here
            for _ in range(2):
                temp = game.clone()
                temp.spawn_tile()
                value = self.minmax(temp, depth - 1, True)
                if value < worst_score:
                    worst_score = value
            return worst_score
        
        
############################################################

def main():
    clock = pygame.time.Clock()
    game = Game2048()
    running = True
    game_over = False
    ai = minmax_agent()
    use_ai = True

    while running:
        clock.tick(FPS)
        draw_grid(WINDOW, game)

        if use_ai and not game_over:
            #move = ai.act_random(game) #moves randomly
            #move = ai.act_simple(game) #only looking for best h for one layer
            move = ai.act_simple_minmax(game, depth = 8) #very simple minmax algorithm
            if move is not None:
                game.move(move)

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            WINDOW.blit(overlay, (0, 0))
            msg_font = pygame.font.SysFont("comicsans", 70, bold=True)
            msg = msg_font.render("GAME OVER", True, (255, 255, 255))
            WINDOW.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - msg.get_height()))
            sub_font = pygame.font.SysFont("comicsans", 36)
            sub = sub_font.render("Press R to Restart or ESC to Quit", True, (220, 220, 220))
            WINDOW.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 40))
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if game_over and event.key == pygame.K_r:
                    game = Game2048()
                    game_over = False
                    continue

                if game_over:
                    continue

                elif event.key == pygame.K_UP:
                    game.move(2)
                elif event.key == pygame.K_RIGHT:
                    game.move(1)
                elif event.key == pygame.K_DOWN:
                    game.move(0)
                elif event.key == pygame.K_LEFT:
                    game.move(3)

        if not game_over and game.is_game_over():
            game_over = True

    pygame.quit()

if __name__ == "__main__":
    main()