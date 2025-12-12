"""
Classic 2048 — Pygame Edition (original style remix)
Author(s): Kalen Luraschi, Daniel Birley
Arrow keys = move tiles
Press ESC to quit
"""

import pygame
import random
import math
from copy import deepcopy
import time
import os
import sys

# Set headless mode for pygame if benchmarking
if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
    pygame.init()
    FPS = 60
    SIZE = 4
    WIDTH, HEIGHT = 600, 600
    TILE_SIZE = WIDTH // SIZE
    BG_COLOR = (205, 192, 180)
    GRID_COLOR = (187, 173, 160)
    FONT_COLOR = (119, 110, 101)
    if len(sys.argv) <= 1 or sys.argv[1] != "benchmark":
        FONT = pygame.font.SysFont("comicsans", 55, bold=True)
        WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2048 Classic")
    else:
        FONT = None
        WINDOW = None
except:
    # Fallback if pygame fails
    FPS = 60
    SIZE = 4
    WIDTH, HEIGHT = 600, 600
    TILE_SIZE = WIDTH // SIZE
    BG_COLOR = (205, 192, 180)
    GRID_COLOR = (187, 173, 160)
    FONT_COLOR = (119, 110, 101)
    FONT = None
    WINDOW = None

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
    #                    Helper functions below                         #
    #####################################################################
    
    def clone(self):
        g = Game2048(self.size)
        g.grid = [row[:] for row in self.grid]
        g.score = self.score
        return g
    
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

    score_font = pygame.font.SysFont("comicsans", 36, bold=True)
    score_text = score_font.render(f"Score: {game.score}", True, (80, 70, 60))
    win.blit(score_text, (15, 10))

    pygame.display.update()

def show_ai_menu():
    """Display menu for selecting AI method"""
    menu_font = pygame.font.SysFont("comicsans", 45, bold=True)
    sub_font = pygame.font.SysFont("comicsans", 28, bold=True)
    small_font = pygame.font.SysFont("comicsans", 18)
    
    WINDOW.fill(BG_COLOR)
    
    title = menu_font.render("Select AI Method", True, (80, 70, 60))
    WINDOW.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))
    
    options = [
        ("1", "Random", "Random moves"),
        ("2", "Simple Greedy", "One-step lookahead"),
        ("3", "Minmax", "Multi-step search"),
        ("4", "Alpha-Beta", "Minmax with pruning"),
        ("5", "Expectimax", "Probabilistic expectations"),
        ("M", "Manual", "Play yourself (arrow keys)"),
    ]
    
    # Better spacing to fit all options
    y_start = 110
    spacing = 75
    for i, (key, name, desc) in enumerate(options):
        y = y_start + i * spacing
        # Key and name on same line
        key_text = sub_font.render(f"[{key}] {name}", True, (80, 70, 60))
        WINDOW.blit(key_text, (WIDTH // 2 - key_text.get_width() // 2, y))
        
        # Description on next line, indented
        desc_text = small_font.render(desc, True, (120, 110, 100))
        WINDOW.blit(desc_text, (WIDTH // 2 - desc_text.get_width() // 2, y + 32))
    
    # Hint at bottom
    hint = small_font.render("Press a number key (1-5) or M to select, ESC to quit", True, (150, 140, 130))
    WINDOW.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 40))
    
    pygame.display.update()
    
    waiting = True
    selected = None
    
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_1:
                    selected = "random"
                    waiting = False
                elif event.key == pygame.K_2:
                    selected = "simple"
                    waiting = False
                elif event.key == pygame.K_3:
                    selected = "minmax"
                    waiting = False
                elif event.key == pygame.K_4:
                    selected = "alpha_beta"
                    waiting = False
                elif event.key == pygame.K_5:
                    selected = "expectimax"
                    waiting = False
                elif event.key == pygame.K_m:
                    selected = "manual"
                    waiting = False
    
    return selected

##############################################################################
class minmax_agent:
    """
    first we get the root which is the current game state
        then we have our 4 directions we can move the board
        we move in all directions to get our child branches (4) with the new move
            for each direction we can move, we place a 2 and a 4 in each spot(if spot is open) and take the average h of all possible random spawns
        we then choose the best h from the 4 directions
    """
    def __init__(self, depth=3):
        self.depth = depth
        self.transposition_table = {}  # Cache for expectimax positions
        self.max_tt_size = 100000  # Limit transposition table size
    
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
    
    def evaluate_h_v2(self, game):
        grid = game.grid

        # 1. Empty tiles (hugely important)
        empty = sum(row.count(0) for row in grid)
        empty_score = empty * 300  # main driver for survival

        # 2. Smoothness (penalize big jumps)
        smoothness = 0
        for r in range(game.size):
            for c in range(game.size - 1):
                if grid[r][c] and grid[r][c+1]:
                    smoothness -= abs(grid[r][c] - grid[r][c+1])

        for c in range(game.size):
            for r in range(game.size - 1):
                if grid[r][c] and grid[r+1][c]:
                    smoothness -= abs(grid[r][c] - grid[r+1][c])

        smoothness *= 0.5

        # 3. Monotonicity (reward snake-like shapes)
        mono = 0
        for row in grid:
            nonzero = [x for x in row if x != 0]
            if nonzero == sorted(nonzero) or nonzero == sorted(nonzero, reverse=True):
                mono += 100

        for col in zip(*grid):
            nonzero = [x for x in col if x != 0]
            if nonzero == sorted(nonzero) or nonzero == sorted(nonzero, reverse=True):
                mono += 100

        # 4. Corner bonus
        max_tile = max(max(row) for row in grid)
        corners = [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]
        corner_bonus = 300 if max_tile in corners else 0

        # 5. Merge reward (exponentially better for big merges)
        merge_reward = 0
        for r in range(game.size):
            for c in range(game.size - 1):
                if grid[r][c] == grid[r][c+1] and grid[r][c] != 0:
                    v = grid[r][c] * 2
                    merge_reward += v * math.log2(v)

        for c in range(game.size):
            for r in range(game.size - 1):
                if grid[r][c] == grid[r+1][c] and grid[r][c] != 0:
                    v = grid[r][c] * 2
                    merge_reward += v * math.log2(v)

        merge_reward *= 1.2

        snake_weights = [
            [8,  32,  4096, 12500],
            [4,  64,  2048, 16384],
            [2,  128, 1024, 32768],
            [1,  256, 512,  70000],
        ]
        
        snake_score = 0
        for r in range(game.size):
            for c in range(game.size):
                snake_score += grid[r][c] * snake_weights[r][c]

        return empty_score + smoothness + mono + corner_bonus + merge_reward + (snake_score * 0.5)

    def evaluate_h_expectimax(self, game):
        """
        Optimized heuristic for expectimax based on Stack Overflow best practices.
        Includes monotonicity penalty and improved merge counting.
        """
        grid = game.grid
        
        # 1. Empty tiles (more is better)
        empty_tiles = sum(row.count(0) for row in game.grid)
        
        # 2. Monotonicity penalty (non-monotonic rows/cols hurt more as values increase)
        monotonicity = 0
        for row in grid:
            # Filter out zeros and check monotonicity
            non_zero = [val for val in row if val != 0]
            if len(non_zero) > 1:
                increasing = all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1))
                decreasing = all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1))
                if not (increasing or decreasing):
                    # Penalty increases with max value in row
                    max_val = max(row) if row else 0
                    if max_val > 0:
                        monotonicity -= max_val * math.log2(max_val) * 2
        
        for col in zip(*grid):
            col = list(col)
            non_zero = [val for val in col if val != 0]
            if len(non_zero) > 1:
                increasing = all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1))
                decreasing = all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1))
                if not (increasing or decreasing):
                    max_val = max(col) if col else 0
                    if max_val > 0:
                        monotonicity -= max_val * math.log2(max_val) * 2
        
        # 3. Smoothness (penalty for large differences between neighbors)
        smoothness = 0
        for r in range(game.size):
            for c in range(game.size - 1):
                if grid[r][c] != 0 and grid[r][c+1] != 0:
                    smoothness -= abs(grid[r][c] - grid[r][c+1])
        for c in range(game.size):
            for r in range(game.size - 1):
                if grid[r][c] != 0 and grid[r+1][c] != 0:
                    smoothness -= abs(grid[r][c] - grid[r+1][c])
        
        # 4. Max tile in corner bonus
        max_tile_corner = max(max(row) for row in grid)
        corner_bonus = 0
        corners = [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]
        if max_tile_corner in corners:
            corner_bonus = max_tile_corner * 10
        
        # 5. Merge potential (adjacent equal values)
        merges = 0
        for r in range(game.size):
            for c in range(game.size - 1):
                if grid[r][c] == grid[r][c+1] and grid[r][c] != 0:
                    v = grid[r][c] * 2
                    merges += v * math.log2(v) if v > 0 else 0
        for c in range(game.size):
            for r in range(game.size - 1):
                if grid[r][c] == grid[r+1][c] and grid[r][c] != 0:
                    v = grid[r][c] * 2
                    merges += v * math.log2(v) if v > 0 else 0
        
        # 6. Max tile value
        max_tile = max(max(row) for row in game.grid)

        # Weighted combination (tuned for expectimax)
        # Note: game.score is NOT included as it's too heavily weighted toward immediate merges
        return (empty_tiles * 2.7 + 
                monotonicity * 1.0 +
                smoothness * 0.1 + 
                corner_bonus * 10.0 + 
                merges * 2.5 + 
                max_tile * 1.0) 
    
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
    
    def act_minimax(self, game, depth=2):
        """first minmax try, picks best h (evaluate_h_v2)"""
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
        if game.is_game_over():
            return 0
        if depth == 0:
            return self.evaluate_h_v2(game)
        
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
                
            return best_score if best_score != -float('inf') else self.evaluate_h_v2(game)
        else:
            # random turn — simulate the computer placing a new 2 or 4 (just 2 for now)
            worst_score = float('inf')

            empty = [(r, c) for r in range(game.size) for c in range(game.size) if game.grid[r][c] == 0]
            if not empty:
                return self.evaluate_h_v2(game)

            for (r, c) in empty:
                temp2 = game.clone()
                for i in range(2):
                    temp2.grid[r][c] = (i+1)*2
                    value = self.minmax(temp2, depth - 1, True)
                    if value < worst_score:
                        worst_score = value

            return worst_score
    
    def act_alpha_beta(self, game, depth=2):
        """Alpha-beta pruning version of minmax, uses evaluate_h_v2"""
        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in range(4):
            moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
            if not moved:
                continue
        
            temp = game.clone()
            temp.grid = new_grid
            temp.score += gained_score

            #go down a branch with alpha-beta pruning
            value = self.minmax_alpha_beta(temp, depth - 1, False, alpha, beta)

            if value > best_score:
                best_score = value
                best_move = move
            
            # Update alpha at root level
            alpha = max(alpha, value)

        return best_move
    
    def minmax_alpha_beta(self, game, depth, maximizing, alpha=-float('inf'), beta=float('inf')):
        """Minmax with alpha-beta pruning optimization"""
        if game.is_game_over():
            return 0
        if depth == 0:
            return self.evaluate_h_v2(game)
        
        if maximizing:
            # Player's turn - maximize score
            best_score = -float('inf')
            for move in range(4):
                moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
                if not moved:
                    continue
            
                temp = game.clone()
                temp.grid = new_grid
                temp.score += gained_score

                value = self.minmax_alpha_beta(temp, depth - 1, False, alpha, beta)

                if value > best_score:
                    best_score = value
                
                # Alpha-beta pruning: update alpha and check for beta cutoff
                alpha = max(alpha, value)
                if value >= beta:
                    return value  # Beta cutoff - prune remaining branches
            
            return best_score if best_score != -float('inf') else self.evaluate_h_v2(game)
        else:
            # Opponent's turn - minimize score (worst case spawn)
            worst_score = float('inf')

            empty = [(r, c) for r in range(game.size) for c in range(game.size) if game.grid[r][c] == 0]
            if not empty:
                return self.evaluate_h_v2(game)  # no empty cells, return current evaluation

            for (r, c) in empty:
                # For each empty cell, find the worst spawn (minimum value)
                cell_worst = float('inf')
                for i in range(2):
                    temp2 = game.clone()
                    temp2.grid[r][c] = (i+1)*2  # Fixed: should be (i+1)*2 to get 2 and 4
                    value = self.minmax_alpha_beta(temp2, depth - 1, True, alpha, beta)
                    if value < cell_worst:
                        cell_worst = value
                    
                    # Alpha-beta pruning: if this spawn gives value <= alpha, we can prune
                    # because the maximizer won't choose this branch
                    if value <= alpha:
                        break  # Alpha cutoff - prune remaining spawns for this cell
                
                # Update worst_score with the worst value from this cell
                if cell_worst < worst_score:
                    worst_score = cell_worst
                
                # Update beta to be the minimum value found so far
                beta = min(beta, worst_score)
                
                # If worst_score <= alpha, we can prune remaining empty cells
                if worst_score <= alpha:
                    break  # Alpha cutoff - prune remaining empty cells

            return worst_score
    
    def _board_hash(self, game):
        """Create a hash of the board state for transposition table"""
        # Create a tuple representation of the grid
        return tuple(tuple(row) for row in game.grid)
    
    def act_expectimax(self, game, depth=4):
        """Expectimax algorithm - uses probabilistic expectations instead of worst-case"""
        # Clear transposition table for new search
        self.transposition_table.clear()
        
        best_move = None
        best_score = -float('inf')

        for move in range(4):
            moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
            if not moved:
                continue
        
            temp = game.clone()
            temp.grid = new_grid
            temp.score += gained_score

            # Go down a branch with expectimax (expectation layer)
            value = self.expectimax(temp, depth - 1, False)

            if value > best_score:
                best_score = value
                best_move = move

        return best_move
    
    def expectimax(self, game, depth, maximizing):
        """Expectimax algorithm: maximizing player moves, expectation over random spawns"""
        # Check transposition table
        board_hash = self._board_hash(game)
        cache_key = (board_hash, depth, maximizing)
        if cache_key in self.transposition_table:
            return self.transposition_table[cache_key]
        
        # Limit transposition table size
        if len(self.transposition_table) > self.max_tt_size:
            # Clear half of the table (simple FIFO-like behavior)
            keys_to_remove = list(self.transposition_table.keys())[:self.max_tt_size // 2]
            for key in keys_to_remove:
                del self.transposition_table[key]
        
        if game.is_game_over():
            result = 0
            self.transposition_table[cache_key] = result
            return result
        
        if depth == 0:
            result = self.evaluate_h_expectimax(game)
            self.transposition_table[cache_key] = result
            return result
        
        if maximizing:
            # Player's turn - maximize score
            best_score = -float('inf')
            for move in range(4):
                moved, new_grid, gained_score, _ = game.apply_move_no_spawn(move)
                if not moved:
                    continue
            
                temp = game.clone()
                temp.grid = new_grid
                temp.score += gained_score

                value = self.expectimax(temp, depth - 1, False)

                if value > best_score:
                    best_score = value
            
            result = best_score if best_score != -float('inf') else self.evaluate_h_expectimax(game)
            self.transposition_table[cache_key] = result
            return result
        else:
            # Expectation layer - weighted average over random tile spawns
            empty = [(r, c) for r in range(game.size) for c in range(game.size) if game.grid[r][c] == 0]
            if not empty:
                result = self.evaluate_h_expectimax(game)
                self.transposition_table[cache_key] = result
                return result

            expected_value = 0.0
            
            for (r, c) in empty:
                # Evaluate spawn=2 (90% probability)
                temp2 = game.clone()
                temp2.grid[r][c] = 2
                value_2 = self.expectimax(temp2, depth - 1, True)
                
                # Evaluate spawn=4 (10% probability)
                temp4 = game.clone()
                temp4.grid[r][c] = 4
                value_4 = self.expectimax(temp4, depth - 1, True)
                
                # Weighted average for this cell: 0.9 * value_2 + 0.1 * value_4
                cell_expected = 0.9 * value_2 + 0.1 * value_4
                expected_value += cell_expected
            
            # Average across all empty cells
            result = expected_value / len(empty)
            self.transposition_table[cache_key] = result
            return result
        
        
############################################################

def benchmark_ai_methods(num_games=10, depth=4):
    """
    Benchmark all AI methods with enhanced metrics:
    - average score
    - highest tile
    - time per move
    - moves per second
    - time to perform 10 moves
    """
    
    print(f"\n{'='*65}")
    print(f" Benchmarking AI Methods ({num_games} games each, depth={depth})")
    print(f"{'='*65}\n")
    
    ai = minmax_agent()
    methods = {
        "random": lambda game: ai.act_random(game),
        "simple": lambda game: ai.act_simple(game),
        "minmax": lambda game: ai.act_minimax(game, depth=depth),
        "alpha_beta": lambda game: ai.act_alpha_beta(game, depth=depth),
        "expectimax": lambda game: ai.act_expectimax(game, depth=depth),
    }
    
    results = {}

    for method_name, method_func in methods.items():
        print(f"Testing {method_name}...", end=" ", flush=True)

        scores = []
        highest_tiles = []
        total_moves = 0
        total_time = 0.0

        # ==============================
        # Measure time for 10 consecutive moves
        # ==============================
        temp_game = Game2048()
        ten_move_start = time.time()
        m = 0
        while m < 10 and not temp_game.is_game_over():
            move = method_func(temp_game)
            if move is None:
                break
            temp_game.move(move)
            m += 1
        ten_move_time = time.time() - ten_move_start

        # ==============================
        # Main benchmark loop
        # ==============================
        for game_num in range(num_games):
            game = Game2048()
            moves = 0
            max_moves = 5000  # prevent infinite loops
            
            start = time.time()

            while not game.is_game_over() and moves < max_moves:
                move = method_func(game)
                if move is None:
                    break
                game.move(move)
                moves += 1

            elapsed = time.time() - start
            
            # Record metrics
            scores.append(game.score)
            highest_tiles.append(max(max(row) for row in game.grid))
            total_moves += moves
            total_time += elapsed

            if num_games > 5 and (game_num + 1) % max(1, num_games // 5) == 0:
                print(".", end="", flush=True)

        # Compute stats
        avg_score = sum(scores) / len(scores)
        avg_highest_tile = sum(highest_tiles) / len(highest_tiles)

        avg_time_per_move = total_time / total_moves if total_moves else 0
        moves_per_second = total_moves / total_time if total_time else 0
        
        results[method_name] = {
            'avg_score': avg_score,
            'max_score': max(scores),
            'min_score': min(scores),
            'avg_highest_tile': avg_highest_tile,
            'max_highest_tile': max(highest_tiles),
            'avg_time_per_move': avg_time_per_move,
            'moves_per_second': moves_per_second,
            'time_for_10_moves': ten_move_time
        }

        print(f" Done! Score={avg_score:.0f}, Tile={avg_highest_tile:.0f}, t/10={ten_move_time:.4f}s")

    print(f"\n{'='*70}")
    print(" Ranking by Average Score")
    print(f"{'='*70}\n")

    ranked = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)

    # Clean table header
    print("ALGORITHM      AVG SCORE   MAX SCORE   MIN SCORE   AVG TILE   MAX TILE   T/MOVE      T/10       MOVES/SEC")
    print("-" * 70)

    for method_name, stats in ranked:
        print(f"{method_name:<14}"
              f"{stats['avg_score']:>10.0f}   "
              f"{stats['max_score']:>9.0f}   "
              f"{stats['min_score']:>9.0f}   "
              f"{stats['avg_highest_tile']:>9.0f}   "
              f"{stats['max_highest_tile']:>8.0f}   "
              f"{stats['avg_time_per_move']:.5f}s   "
              f"{stats['time_for_10_moves']:.4f}s   "
              f"{stats['moves_per_second']:.1f}")   

    print(f"{'='*70}\n")
    return results

def main():
    clock = pygame.time.Clock()
    
    # Show menu to select AI method
    ai_method = show_ai_menu()
    if ai_method is None:
        pygame.quit()
        return
    
    game = Game2048()
    running = True
    game_over = False
    ai = minmax_agent()
    ai_depth = 4
    
    # AI method names for display
    ai_names = {
        "random": "Random",
        "simple": "Simple Greedy",
        "minmax": "Minmax",
        "alpha_beta": "Alpha-Beta",
        "expectimax": "Expectimax",
        "manual": "Manual"
    }

    while running:
        clock.tick(FPS)
        draw_grid(WINDOW, game)
        
        # Display current AI method (after grid so it's on top)
        method_font = pygame.font.SysFont("comicsans", 20)
        if ai_method == "manual":
            method_text = method_font.render(f"Mode: {ai_names[ai_method]}", True, (100, 90, 80))
        else:
            method_text = method_font.render(f"AI: {ai_names[ai_method]}", True, (100, 90, 80))
        WINDOW.blit(method_text, (15, HEIGHT - 30))
        pygame.display.update()  # Update again to show method text

        if ai_method != "manual" and not game_over:
            # Execute selected AI method
            if ai_method == "random":
                move = ai.act_random(game)
            elif ai_method == "simple":
                move = ai.act_simple(game)
            elif ai_method == "minmax":
                move = ai.act_minimax(game, depth=ai_depth)
            elif ai_method == "alpha_beta":
                move = ai.act_alpha_beta(game, depth=ai_depth)
            elif ai_method == "expectimax":
                move = ai.act_expectimax(game, depth=ai_depth)
            else:
                move = None
            
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

                # Manual controls (arrow keys always work for manual override)
                if event.key == pygame.K_UP:
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
    import sys
    
    # Check if user wants to run benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        benchmark_ai_methods(num_games=num_games, depth=depth)
    else:
        main()