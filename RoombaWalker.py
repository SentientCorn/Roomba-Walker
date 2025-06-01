import pygame
import random
from enum import Enum
from collections import deque
import json
import os


pygame.init()

# Constants and Cnfigurations
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

GRID_WIDTH = 50
GRID_HEIGHT = 40 
TILE_SIZE = 20

ROBOT_COUNT = 4
ALL_LONG_RANGE = True

SIMULATION_DURATION_MINUTES = 2  # Set to 0 for no time limit, or specify a positive integer for duration limit

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
BROWN = (139, 69, 19)

class SimulationMode(Enum):
    SIMULATION = 0
    EDITOR = 1

class TileType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    ALT_OBSTACLE = 1  
    CHARGING_STATION = 2
    CLEANED = 3
    VOID = 4  

class RobotState(Enum):
    EXPLORING = 0
    CLEANING = 1
    RETURNING_TO_CHARGE = 2
    CHARGING = 3

class Room:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[TileType.EMPTY for _ in range(width)] for _ in range(height)]
        self.charging_stations = []
        self.station_occupants = {pos: deque() for pos in self.charging_stations}
        self.generate_room()
    
    """Generate a random room layout with obstacles and charging stations"""
    def generate_room(self):
        # Add obstacles randomly (10-15% of the room)
        obstacle_count = int(self.width * self.height * 0.12)
        for _ in range(obstacle_count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid[y][x] = TileType.OBSTACLE
        
        # Add charging stations (2-3 stations)
        station_count = random.randint(2, 3)
        for _ in range(station_count):
            while True:
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if self.grid[y][x] == TileType.EMPTY:
                    self.grid[y][x] = TileType.CHARGING_STATION
                    self.charging_stations.append((x, y))
                    break
    
    def save_to_file(self, filename):
        """Save room layout to JSON file"""
        room_data = {
            'width': self.width,
            'height': self.height,
            'grid': [[tile.value for tile in row] for row in self.grid],
            'charging_stations': self.charging_stations
        }
        
        # Create rooms directory if it doesn't exist
        os.makedirs('rooms', exist_ok=True)
        
        with open(f'rooms/{filename}.json', 'w') as f:
            json.dump(room_data, f, indent=2)
    
    def load_from_file(self, filename):
        """Load room layout from JSON file"""
        try:
            with open(f'rooms/{filename}.json', 'r') as f:
                room_data = json.load(f)
            
            self.width = room_data['width']
            self.height = room_data['height']
            self.grid = [[TileType(tile_value) for tile_value in row] for row in room_data['grid']]
            self.charging_stations = room_data['charging_stations']
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def get_available_rooms(self):
        """Get list of available saved rooms"""
        if not os.path.exists('rooms'):
            return []
        
        rooms = []
        for filename in os.listdir('rooms'):
            if filename.endswith('.json'):
                rooms.append(filename[:-5]) 
        return sorted(rooms)
    
    def clear_room(self):
        """Clear the room to all empty tiles"""
        self.grid = [[TileType.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.charging_stations = []
    
    def set_tile(self, x, y, tile_type):
        """Set tile type at specific position"""
        if self.is_valid_position(x, y):
            if self.grid[y][x] == TileType.CHARGING_STATION and (x, y) in self.charging_stations:
                self.charging_stations.remove((x, y))
            
            if tile_type == TileType.CHARGING_STATION and (x, y) not in self.charging_stations:
                self.charging_stations.append((x, y))
            
            self.grid[y][x] = tile_type
    
    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_passable(self, x, y):
        if not self.is_valid_position(x, y):
            return False
        tile_type = self.grid[y][x]
        return tile_type != TileType.OBSTACLE and tile_type != TileType.VOID 
    
    def get_tile_type(self, x, y):
        if not self.is_valid_position(x, y):
            return TileType.OBSTACLE
        return self.grid[y][x]
    
    def clean_tile(self, x, y):
        if self.is_valid_position(x, y) and self.grid[y][x] == TileType.EMPTY:
            # Probabilistic cleaning success (90% chance)
            if random.random() < 0.9:
                self.grid[y][x] = TileType.CLEANED
                return True
        return False

class Robot:
    def __init__(self, robot_id, start_x, start_y, room, robot_type="standard"):
        self.id = robot_id
        self.robot_type = robot_type
        self.x = start_x
        self.y = start_y
        self.room = room
        self.battery = 100
        self.max_battery = 100
        self.state = RobotState.EXPLORING
        self.known_map = {}  # Dictionary to store known tiles
        self.target_x = None
        self.target_y = None
        self.path = []
        self.last_direction = None  
        self.pre_charge_position = None  # Position to return to after charging
        self.last_move_time = 0
        self.move_delay = 200  
        self.charging_time = 0
        self.shared_knowledge = {}  # Knowledge shared by other robots
        
        # Statistics tracking
        self.charge_count = 0  # Number of times this robot has charged
        self.total_charging_time = 0  # Cumulative time spent charging (in milliseconds)
        self.charging_start_time = 0  # When current charging session started
        self.tiles_cleaned = 0  # Total tiles cleaned by this robot
        self.ineffective_moves = 0  # Number of moves onto already cleaned tiles

        if robot_type == "long_range":
            self.view_range = 3 
            self.energy_consumption_rate = 0.04 
            self.move_delay = 150  
        else:  # standard robot
            self.view_range = 1
            self.energy_consumption_rate = 0.02
            self.move_delay = 200
            
        # Initialize known map with current position
        self.update_perception()
    
    def update_perception(self):
        """Update robot's knowledge of surrounding tiles with variable range"""
        directions = []

        # Generate all positions within view range
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                directions.append((dx, dy))

        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if self.room.is_valid_position(nx, ny):
                self.known_map[(nx, ny)] = self.room.get_tile_type(nx, ny)
    
    def get_adjacent_positions(self):
        """Get valid adjacent positions"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        adjacent = []
        
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if self.room.is_passable(nx, ny):
                adjacent.append((nx, ny))
        
        return adjacent
    
    def find_path_to_charging_station(self):
        """Find path to nearest charging station using A*"""
        if not self.room.charging_stations:
            return []
        
        # Find nearest charging station
        nearest_station = min(self.room.charging_stations, 
                            key=lambda station: abs(station[0] - self.x) + abs(station[1] - self.y))
        
        return self.find_path(self.x, self.y, nearest_station[0], nearest_station[1])
    
    def find_path(self, start_x, start_y, target_x, target_y):
        """A* pathfinding algorithm with penalty for cleaned tiles"""
        open_set = [(start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, target_x, target_y)}

        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            open_set.remove(current)

            if current == (target_x, target_y):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.room.is_passable(neighbor[0], neighbor[1]):
                    continue
                
                # Base movement cost
                tentative_g = g_score[current] + 1

                # Add penalty for cleaned tiles
                if self.room.get_tile_type(neighbor[0], neighbor[1]) == TileType.CLEANED:
                    tentative_g += 2  

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor[0], neighbor[1], target_x, target_y)

                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return []
    
    def heuristic(self, x1, y1, x2, y2):
        """Manhattan distance heuristic"""
        return abs(x1 - x2) + abs(y1 - y2)
    def find_unexplored_in_view_range(self):
        """Find unexplored tiles within the robot's view range"""
        unexplored_in_view = []

        # Check all positions within view range
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                if dx == 0 and dy == 0:  # Skip current position
                    continue

                nx, ny = self.x + dx, self.y + dy

                # Check if position is valid and unexplored
                if (self.room.is_valid_position(nx, ny) and 
                    (nx, ny) not in self.known_map and 
                    (nx, ny) not in self.shared_knowledge):
                    unexplored_in_view.append((nx, ny))

        return unexplored_in_view
    
    def find_unexplored_target(self):
        """Find nearest unexplored cleanable tile, prioritizing unvisited areas and avoiding cleaned tiles"""
        # First, check for unexplored tiles within view range
        unexplored_in_view = self.find_unexplored_in_view_range()

        if unexplored_in_view:
            # Return nearest unexplored tile within view range
            return min(unexplored_in_view, key=lambda pos: abs(pos[0] - self.x) + abs(pos[1] - self.y))

        # Create a combined knowledge map from own knowledge and shared knowledge
        combined_map = {**self.known_map, **self.shared_knowledge}

        # Look for unknown tiles adjacent to known tiles (frontier exploration)
        frontier_tiles = []
        for (kx, ky), tile_type in combined_map.items():
            if tile_type != TileType.OBSTACLE and tile_type != TileType.VOID:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = kx + dx, ky + dy
                    if ((nx, ny) not in combined_map and 
                        self.room.is_valid_position(nx, ny)):
                        frontier_tiles.append((nx, ny))

        # If we have frontier tiles, prioritize them
        if frontier_tiles:
            return min(frontier_tiles, key=lambda pos: abs(pos[0] - self.x) + abs(pos[1] - self.y))

        # Look for known empty tiles that haven't been cleaned yet
        unvisited_empty = []
        for (kx, ky), tile_type in combined_map.items():
            if tile_type == TileType.EMPTY:
                unvisited_empty.append((kx, ky))

        if unvisited_empty:
            return min(unvisited_empty, key=lambda pos: abs(pos[0] - self.x) + abs(pos[1] - self.y))

        return None
    
        
    def share_knowledge_with_nearby_robots(self, robots):
        """Enhanced knowledge sharing with better coordination to avoid clustering"""
        for other_robot in robots:
            if other_robot.id != self.id:
                distance = abs(other_robot.x - self.x) + abs(other_robot.y - self.y)
                if distance <= 2:  
                    # Share knowledge
                    for pos, tile_type in self.known_map.items():
                        if pos not in other_robot.shared_knowledge:
                            other_robot.shared_knowledge[pos] = tile_type

                    # Receive knowledge
                    for pos, tile_type in other_robot.known_map.items():
                        if pos not in self.shared_knowledge:
                            self.shared_knowledge[pos] = tile_type
                            self.known_map[pos] = tile_type

                    # If robots are too close and have similar targets, encourage dispersion
                    if (distance <= 1 and 
                        self.target_x and other_robot.target_x and
                        abs(self.target_x - other_robot.target_x) <= 2 and
                        abs(self.target_y - other_robot.target_y) <= 2):

                        # Lower ID robot keeps target, higher ID robot finds new target
                        if self.id > other_robot.id:
                            self.target_x = None
                            self.target_y = None
                            self.path = []
    
    def update(self, current_time, robots):
        """Main update function for robot behavior with type-specific energy consumption"""
        # Update perception
        self.update_perception()

        # Share knowledge with nearby robots
        self.share_knowledge_with_nearby_robots(robots)

        # Consume battery stochastically with robot-type specific rate
        if self.state != RobotState.CHARGING and random.random() < self.energy_consumption_rate:
            self.battery = max(0, self.battery - random.randint(1, 3))

        # State machine
        if self.state == RobotState.CHARGING:
            if self.battery < self.max_battery:
                self.battery = min(self.max_battery, self.battery + 1)
                self.charging_time += 1
                # Track total charging time
                if self.charging_start_time == 0:
                    self.charging_start_time = current_time
            else:
                    # Charging complete
                    if self.charging_start_time > 0:
                        self.total_charging_time += current_time - self.charging_start_time
                        self.charging_start_time = 0
                    # Leave queue
                    if self.room.get_tile_type(self.x, self.y) == TileType.CHARGING_STATION:
                        station_queue = self.room.station_occupants.get((self.x, self.y), deque())
                        if station_queue and station_queue[0] == self.id:
                            station_queue.popleft()

                    # Set target to return to pre-charge position if it exists
                    if self.pre_charge_position:
                        self.target_x, self.target_y = self.pre_charge_position
                        self.path = self.find_path(self.x, self.y, self.target_x, self.target_y)
                        self.pre_charge_position = None  

                    self.state = RobotState.EXPLORING
                    self.charging_time = 0


        elif self.battery < 20:  
            if self.state != RobotState.RETURNING_TO_CHARGE:
                # Save current position before going to charge (if not already at charging station)
                if self.room.get_tile_type(self.x, self.y) != TileType.CHARGING_STATION:
                    self.pre_charge_position = (self.x, self.y)

                self.state = RobotState.RETURNING_TO_CHARGE
                self.path = []  
            if not self.path:
                self.path = self.find_path_to_charging_station()


        elif self.state == RobotState.EXPLORING:
            # Try to clean current tile
            if self.room.clean_tile(self.x, self.y):
                self.tiles_cleaned += 1

            # Find new target if needed
            if not self.target_x or not self.target_y or (self.x == self.target_x and self.y == self.target_y):
                target = self.find_unexplored_target()
                if target:
                    self.target_x, self.target_y = target
                    self.path = self.find_path(self.x, self.y, self.target_x, self.target_y)

        # Movement
        if self.state != RobotState.CHARGING and current_time - self.last_move_time > self.move_delay:
            self.move(current_time)
            self.last_move_time = current_time


    def move(self, current_time):
        """Move robot based on current path or intelligent exploration with straight-line tendency"""
        if self.path:
            next_pos = self.path.pop(0)
            if self.room.is_passable(next_pos[0], next_pos[1]):
                # Check if moving to an already cleaned tile (ineffective travel)
                if self.room.get_tile_type(next_pos[0], next_pos[1]) == TileType.CLEANED:
                    self.ineffective_moves += 1
    
                # Update last direction
                self.last_direction = (next_pos[0] - self.x, next_pos[1] - self.y)
                self.x, self.y = next_pos
    
                # Check if reached charging station
                if self.room.get_tile_type(self.x, self.y) == TileType.CHARGING_STATION:
                    if not hasattr(self.room, 'station_occupants'):
                        self.room.station_occupants = {}
                    if self.state == RobotState.RETURNING_TO_CHARGE:
                        station_queue = self.room.station_occupants.setdefault((self.x, self.y), deque())
                        if not station_queue or station_queue[0] == self.id:
                            if self.id not in station_queue:
                                station_queue.append(self.id)
                            self.state = RobotState.CHARGING
                            self.charge_count += 1
                            self.charging_start_time = current_time
                        else:
                            # Wait nearby if station is occupied
                            self.path = []
        else:
            adjacent = self.get_adjacent_positions()
            if adjacent:
                # Create a combined knowledge map
                combined_map = {**self.known_map, **self.shared_knowledge}
    
                # Check if there are any unexplored tiles in view range
                unexplored_in_view = self.find_unexplored_in_view_range()
                
                # Priority 1: Unexplored directions
                unexplored = [(x, y) for x, y in adjacent if (x, y) not in combined_map]
    
                # Priority 2: Empty tiles (not yet cleaned)
                empty_tiles = [(x, y) for x, y in adjacent 
                              if combined_map.get((x, y)) == TileType.EMPTY]
    
                # Priority 3: Charging stations (if we need to charge)
                charging_tiles = [(x, y) for x, y in adjacent 
                                 if combined_map.get((x, y)) == TileType.CHARGING_STATION]
    
                # Priority 4: Already cleaned tiles (last resort)
                cleaned_tiles = [(x, y) for x, y in adjacent 
                                if combined_map.get((x, y)) == TileType.CLEANED]
    
                chosen_pos = None
    
                # Always prioritize moving towards unexplored tiles in vision range
                if unexplored_in_view:
                    # Find the nearest unexplored tile in vision range
                    nearest_unexplored = min(unexplored_in_view, 
                                           key=lambda pos: abs(pos[0] - self.x) + abs(pos[1] - self.y))
                    
                    # Choose adjacent tile that gets us closer to the nearest unexplored tile
                    best_adjacent = min(adjacent, 
                                      key=lambda pos: abs(pos[0] - nearest_unexplored[0]) + abs(pos[1] - nearest_unexplored[1]))
                    chosen_pos = best_adjacent
                
                # Straight-line movement logic when no unexplored tiles in view
                elif self.last_direction:
                    # Try to continue in the same direction
                    straight_x = self.x + self.last_direction[0]
                    straight_y = self.y + self.last_direction[1]
                    
                    if (straight_x, straight_y) in adjacent:
                        # Prefer continuing straight if the tile isn't an obstacle
                        tile_type = combined_map.get((straight_x, straight_y))
                        if tile_type != TileType.OBSTACLE and tile_type != TileType.VOID:
                            chosen_pos = (straight_x, straight_y)
    
                # If neither unexplored-seeking nor straight-line movement was chosen, use original priority system
                if not chosen_pos:
                    if unexplored:
                        chosen_pos = random.choice(unexplored)
                    elif empty_tiles:
                        chosen_pos = random.choice(empty_tiles)
                    elif self.state == RobotState.RETURNING_TO_CHARGE and charging_tiles:
                        chosen_pos = random.choice(charging_tiles)
                    elif cleaned_tiles:
                        chosen_pos = random.choice(cleaned_tiles)
                    else:
                        chosen_pos = random.choice(adjacent)
    
                # Check if moving to an already cleaned tile (ineffective travel)
                if self.room.get_tile_type(chosen_pos[0], chosen_pos[1]) == TileType.CLEANED:
                    self.ineffective_moves += 1
    
                # Update last direction and position
                self.last_direction = (chosen_pos[0] - self.x, chosen_pos[1] - self.y)
                self.x, self.y = chosen_pos


class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Cleaner Robot Simulation")
        self.clock = pygame.time.Clock()
        self.room = Room(GRID_WIDTH, GRID_HEIGHT)
        self.robots = []
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.mode = SimulationMode.SIMULATION
        self.selected_tile_type = TileType.OBSTACLE
        self.mouse_held = False
        self.input_text = ""
        self.input_active = False
        self.message = ""
        self.message_timer = 0
        
        # Simulation statistics
        self.simulation_start_time = pygame.time.get_ticks()
        self.simulation_duration = 0

        self.simulation_paused = False
        self.max_duration = SIMULATION_DURATION_MINUTES * 60 * 1000 if SIMULATION_DURATION_MINUTES > 0 else 0  # Convert to milliseconds
        self.duration_reached = False
        
        # Create robots
        self.create_robots()
    
    def create_robots(self):
        """Create robots and place them in the room"""
        self.robots = []
        for i in range(min(ROBOT_COUNT, GRID_WIDTH * GRID_HEIGHT)):
            # Place robots at charging stations if available 
            if i < len(self.room.charging_stations):
                x, y = self.room.charging_stations[i]
            else:
                # Spawn randomly
                while True:
                    x = random.randint(0, GRID_WIDTH - 1)
                    y = random.randint(0, GRID_HEIGHT - 1)
                    if self.room.get_tile_type(x, y) == TileType.EMPTY:
                        break
            if ALL_LONG_RANGE:
                robot_type = "long_range"
            else:
                robot_type = "standard"
            
            self.robots.append(Robot(i, x, y, self.room, robot_type))
            

        # Reset simulation timer when creating new robots
        self.simulation_start_time = pygame.time.get_ticks()
        self.simulation_duration = 0
        self.duration_reached = False
        self.simulation_paused = False
    
    def show_message(self, text, duration=3000):
        """Show a temporary message"""
        self.message = text
        self.message_timer = pygame.time.get_ticks() + duration
    
    def draw_grid(self):
        """Draw the room grid"""
        for y in range(self.room.height):
            for x in range(self.room.width):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile_type = self.room.get_tile_type(x, y)

                if tile_type == TileType.EMPTY:
                    pygame.draw.rect(self.screen, WHITE, rect)
                elif tile_type == TileType.OBSTACLE:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                elif tile_type == TileType.CHARGING_STATION:
                    pygame.draw.rect(self.screen, YELLOW, rect)
                elif tile_type == TileType.CLEANED:
                    pygame.draw.rect(self.screen, LIGHT_BLUE, rect)
                elif tile_type == TileType.VOID:
                    pygame.draw.rect(self.screen, GRAY, rect) 

                # Highlight selected tile in editor mode
                if self.mode == SimulationMode.EDITOR:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    grid_x, grid_y = mouse_x // TILE_SIZE, mouse_y // TILE_SIZE
                    if x == grid_x and y == grid_y and self.room.is_valid_position(grid_x, grid_y):
                        pygame.draw.rect(self.screen, RED, rect, 3)

                pygame.draw.rect(self.screen, GRAY, rect, 1)

    def draw_robots(self):
        """Draw robots with different colors and battery indicators"""
        colors = [RED, BLUE, GREEN, ORANGE]
        
        for i, robot in enumerate(self.robots):
            color = colors[i % len(colors)]
            rect = pygame.Rect(robot.x * TILE_SIZE + 2, robot.y * TILE_SIZE + 2, 
                             TILE_SIZE - 4, TILE_SIZE - 4)
            pygame.draw.rect(self.screen, color, rect)
            
            # Draw different shape for long-range robots 
            if robot.robot_type == "long_range":
               
                center_x = robot.x * TILE_SIZE + TILE_SIZE // 2
                center_y = robot.y * TILE_SIZE + TILE_SIZE // 2
                half_size = (TILE_SIZE - 4) // 2
                diamond_points = [
                    (center_x, center_y - half_size),  
                    (center_x + half_size, center_y), 
                    (center_x, center_y + half_size),  
                    (center_x - half_size, center_y)   
                ]
                pygame.draw.polygon(self.screen, color, diamond_points)
            
            # Draw battery level as a small bar
            battery_width = int((TILE_SIZE - 4) * (robot.battery / 100))
            battery_rect = pygame.Rect(robot.x * TILE_SIZE + 2, robot.y * TILE_SIZE + TILE_SIZE - 4,
                                     battery_width, 2)
            battery_color = GREEN if robot.battery > 50 else ORANGE if robot.battery > 20 else RED
            pygame.draw.rect(self.screen, battery_color, battery_rect)
            
            # Draw robot ID
            text = self.font.render(str(robot.id), True, WHITE)
            self.screen.blit(text, (robot.x * TILE_SIZE + 5, robot.y * TILE_SIZE + 5))
    
    def draw_ui(self):
        """Draw user interface information"""
        ui_x = GRID_WIDTH * TILE_SIZE + 10
        
        if self.mode == SimulationMode.SIMULATION:
            self.draw_simulation_ui(ui_x)
        else:
            self.draw_editor_ui(ui_x)
        
        # Draw message if active
        if self.message and pygame.time.get_ticks() < self.message_timer:
            message_surface = self.font.render(self.message, True, BLACK)
            message_rect = message_surface.get_rect()
            message_rect.center = (WINDOW_WIDTH // 2, 50)
            pygame.draw.rect(self.screen, WHITE, message_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, BLACK, message_rect.inflate(20, 10), 2)
            self.screen.blit(message_surface, message_rect)
    
    def draw_simulation_ui(self, ui_x):
        """Draw simulation mode UI with compact robot stats including ineffective travel"""
        # Title
        title_text = self.font.render("Cleaning Robot Simulation", True, BLACK)
        self.screen.blit(title_text, (ui_x, 10))

        # Mode indicator with pause status
        mode_text = "Mode: SIMULATION"
        if self.simulation_paused:
            mode_text += " (PAUSED)"
            mode_color = RED
        else:
            mode_color = GREEN

        mode_surface = self.font.render(mode_text, True, mode_color)
        self.screen.blit(mode_surface, (ui_x, 35))

        # Simulation statistics with duration limit info
        sim_stats = [
            f"Duration: {self.format_time(self.simulation_duration)}"
        ]

        # Add duration limit info if set
        if self.max_duration > 0:
            remaining = max(0, self.max_duration - self.simulation_duration)
            sim_stats.append(f"Limit: {self.format_time(self.max_duration)} (Remaining: {self.format_time(remaining)})")

        for i, stat in enumerate(sim_stats):
            text = self.small_font.render(stat, True, BLACK)
            self.screen.blit(text, (ui_x, 60 + i * 18))

        # Controls
        controls = [
            "SPACE - Switch to Editor",
            "P - Pause/Resume Simulation",
            "R - Reset/Generate New Room",
            "S - Save Room",
            "L - Load Room"
        ]

        control_start_y = 60 + len(sim_stats) * 18 + 10
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, BLACK)
            self.screen.blit(text, (ui_x, control_start_y + i * 18))
        y_offset = control_start_y + len(controls) * 18 + 20

        # Robot status
        for robot in self.robots:
            color_rect = pygame.Rect(ui_x, y_offset, 15, 15)
            colors = [RED, BLUE, GREEN, ORANGE]
            pygame.draw.rect(self.screen, colors[robot.id % len(colors)], color_rect)

            
            if robot.robot_type == "long_range":
                center_x, center_y = ui_x + 7, y_offset + 7
                diamond_points = [
                    (center_x, center_y - 6),
                    (center_x + 6, center_y),
                    (center_x, center_y + 6),
                    (center_x - 6, center_y)
                ]
                pygame.draw.polygon(self.screen, colors[robot.id % len(colors)], diamond_points)

            type_indicator = "LR" if robot.robot_type == "long_range" else "ST"
            state_short = robot.state.name[:4]  # Truncate state name

            # Line 1: ID, type, state, battery, position
            line1 = f"R{robot.id}({type_indicator}) {state_short} Bat:{robot.battery}% ({robot.x},{robot.y})"

            # Line 2: Charges, cleaned tiles, total charging time
            line2 = f"Chg:{robot.charge_count} Clean:{robot.tiles_cleaned} Time:{self.format_time_compact(robot.total_charging_time)}"

            # Line 3: Ineffective moves
            line3 = f"Ineffective moves: {robot.ineffective_moves}"

            # Add current charging time if actively charging
            if robot.state == RobotState.CHARGING and robot.charging_start_time > 0:
                current_charge_time = pygame.time.get_ticks() - robot.charging_start_time
                line2 += f" (Now:{self.format_time_compact(current_charge_time)})"

            # Render the three lines
            line1_text = self.small_font.render(line1, True, BLACK)
            line2_text = self.small_font.render(line2, True, BLACK)
            line3_text = self.small_font.render(line3, True, BLACK)

            self.screen.blit(line1_text, (ui_x + 20, y_offset))
            self.screen.blit(line2_text, (ui_x + 20, y_offset + 15))
            self.screen.blit(line3_text, (ui_x + 20, y_offset + 30))

            y_offset += 50  

        # Overall statistics
        total_tiles = GRID_WIDTH * GRID_HEIGHT
        cleaned_tiles = sum(1 for row in self.room.grid for tile in row if tile == TileType.CLEANED)
        obstacle_tiles = sum(1 for row in self.room.grid for tile in row if tile == TileType.OBSTACLE)
        void_tiles = sum(1 for row in self.room.grid for tile in row if tile == TileType.VOID)
        cleanable_tiles = total_tiles - obstacle_tiles - void_tiles - len(self.room.charging_stations)

        # Cumulative statistics
        total_charges = sum(robot.charge_count for robot in self.robots)
        total_cleaned = sum(robot.tiles_cleaned for robot in self.robots)
        total_charging_time = sum(robot.total_charging_time for robot in self.robots)
        total_ineffective = sum(robot.ineffective_moves for robot in self.robots)

        # Add current charging time for robots currently charging
        for robot in self.robots:
            if robot.state == RobotState.CHARGING and robot.charging_start_time > 0:
                total_charging_time += pygame.time.get_ticks() - robot.charging_start_time

        stats_y = y_offset + 20
        progress_pct = cleaned_tiles/cleanable_tiles*100 if cleanable_tiles > 0 else 0

        stats_text = [
            "=== OVERALL STATS ===",
            f"Progress: {cleaned_tiles}/{cleanable_tiles} ({progress_pct:.1f}%)",
            f"Charges: {total_charges} | Cleaned: {total_cleaned}",
            f"Total Charge Time: {self.format_time_compact(total_charging_time)}",
            f"Total Ineffective Moves: {total_ineffective}"
        ]

        for i, text in enumerate(stats_text):
            color = BLACK if i > 0 else BLUE
            rendered_text = self.small_font.render(text, True, color)
            self.screen.blit(rendered_text, (ui_x, stats_y + i * 18))

    def format_time(self, milliseconds):
        """Format time in milliseconds to readable format"""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        hours = minutes // 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes%60:02d}:{seconds%60:02d}"
        elif minutes > 0:
            return f"{minutes:02d}:{seconds%60:02d}"
        else:
            return f"{seconds:02d}s"
    
    def format_time_compact(self, milliseconds):
        """Format time in milliseconds to compact readable format"""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        hours = minutes // 60

        if hours > 0:
            return f"{hours}h{minutes%60}m"
        elif minutes > 0:
            return f"{minutes}m{seconds%60}s"
        else:
            return f"{seconds}s"
    
    def draw_editor_ui(self, ui_x):
        """Draw editor mode UI"""
        # Title
        title_text = self.font.render("Room Editor", True, BLACK)
        self.screen.blit(title_text, (ui_x, 10))
        
        # Mode indicator
        mode_text = self.font.render("Mode: EDITOR", True, BLUE)
        self.screen.blit(mode_text, (ui_x, 35))
        
        # Controls
        controls = [
            "SPACE - Switch to Simulation",
            "1 - Empty Tile",
            "2 - Obstacle", 
            "3 - Charging Station",
            "4 - Void Tile", 
            "C - Clear Room",
            "G - Generate Random Room",
            "S - Save Room",
            "L - Load Room"
        ]
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, BLACK)
            self.screen.blit(text, (ui_x, 60 + i * 18))
        
        # Current tool
        tool_text = f"Current Tool: {self.selected_tile_type.name}"
        text = self.font.render(tool_text, True, BLACK)
        self.screen.blit(text, (ui_x, 220))
        

        # Tool preview
        tool_rect = pygame.Rect(ui_x, 245, 30, 30)
        if self.selected_tile_type == TileType.EMPTY:
            pygame.draw.rect(self.screen, WHITE, tool_rect)
        elif self.selected_tile_type == TileType.OBSTACLE:
            pygame.draw.rect(self.screen, DARK_GRAY, tool_rect)
        elif self.selected_tile_type == TileType.CHARGING_STATION:
            pygame.draw.rect(self.screen, YELLOW, tool_rect)
        elif self.selected_tile_type == TileType.VOID:  
            pygame.draw.rect(self.screen, GRAY, tool_rect)
        pygame.draw.rect(self.screen, BLACK, tool_rect, 2)
        
        # Input field for save/load
        if self.input_active:
            input_label = "Enter filename:"
            label_text = self.font.render(input_label, True, BLACK)
            self.screen.blit(label_text, (ui_x, 300))
            
            input_rect = pygame.Rect(ui_x, 325, 200, 30)
            color = WHITE if self.input_active else LIGHT_BLUE
            pygame.draw.rect(self.screen, color, input_rect)
            pygame.draw.rect(self.screen, BLACK, input_rect, 2)
            
            input_surface = self.font.render(self.input_text, True, BLACK)
            self.screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
        
        # Available rooms
        available_rooms = self.room.get_available_rooms()
        if available_rooms:
            rooms_label = "Available Rooms:"
            label_text = self.small_font.render(rooms_label, True, BLACK)
            self.screen.blit(label_text, (ui_x, 380))
            
            for i, room_name in enumerate(available_rooms[:10]):  # Show max 10 rooms
                room_text = self.small_font.render(f"- {room_name}", True, BLACK)
                self.screen.blit(room_text, (ui_x, 400 + i * 15))
    
    def handle_editor_input(self, event):
        """Handle input events in editor mode"""
        if event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    # Process the input
                    if hasattr(self, 'pending_action'):
                        if self.pending_action == 'save':
                            if self.input_text.strip():
                                self.room.save_to_file(self.input_text.strip())
                                self.show_message(f"Room saved as '{self.input_text.strip()}'")
                            else:
                                self.show_message("Please enter a filename")
                        elif self.pending_action == 'load':
                            if self.input_text.strip():
                                if self.room.load_from_file(self.input_text.strip()):
                                    self.show_message(f"Room '{self.input_text.strip()}' loaded")
                                    self.create_robots()
                                else:
                                    self.show_message(f"Failed to load room '{self.input_text.strip()}'")
                            else:
                                self.show_message("Please enter a filename")
                        delattr(self, 'pending_action')
                    self.input_active = False
                    self.input_text = ""
                elif event.key == pygame.K_ESCAPE:
                    self.input_active = False
                    self.input_text = ""
                    if hasattr(self, 'pending_action'):
                        delattr(self, 'pending_action')
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    if len(self.input_text) < 20:
                        self.input_text += event.unicode
            else:
                if event.key == pygame.K_1:
                    self.selected_tile_type = TileType.EMPTY
                elif event.key == pygame.K_2:
                    self.selected_tile_type = TileType.OBSTACLE
                elif event.key == pygame.K_3:
                    self.selected_tile_type = TileType.CHARGING_STATION
                elif event.key == pygame.K_4:  
                    self.selected_tile_type = TileType.VOID
                elif event.key == pygame.K_c:
                    self.room.clear_room()
                    self.create_robots()
                    self.show_message("Room cleared")
                elif event.key == pygame.K_g:
                    self.room.clear_room()
                    self.room.generate_room()
                    self.create_robots()
                    self.show_message("New room generated")
                elif event.key == pygame.K_s:
                    self.input_active = True
                    self.pending_action = 'save'
                elif event.key == pygame.K_l:
                    self.input_active = True
                    self.pending_action = 'load'
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.mouse_held = True
                self.handle_tile_click()
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                self.mouse_held = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_held:
                self.handle_tile_click()
    
    def handle_tile_click(self):
        """Handle clicking on tiles in editor mode"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x, grid_y = mouse_x // TILE_SIZE, mouse_y // TILE_SIZE
        
        if self.room.is_valid_position(grid_x, grid_y):
            self.room.set_tile(grid_x, grid_y, self.selected_tile_type)
            
            # Recreate robots if charging stations changed
            if self.selected_tile_type == TileType.CHARGING_STATION or \
                self.room.get_tile_type(grid_x, grid_y) == TileType.CHARGING_STATION:
                self.create_robots()
    def run(self):
        """Main simulation loop with optional duration limit"""
        running = True

        while running:
            current_time = pygame.time.get_ticks()

            # Check if simulation duration limit has been reached
            if (self.max_duration > 0 and 
                not self.duration_reached and 
                self.simulation_duration >= self.max_duration and 
                self.mode == SimulationMode.SIMULATION):
                self.simulation_paused = True
                self.duration_reached = True
                self.show_message(f"Simulation paused - Duration limit ({SIMULATION_DURATION_MINUTES} min) reached! Press P to resume.", 5000)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Toggle between simulation and editor mode
                        if self.mode == SimulationMode.SIMULATION:
                            self.mode = SimulationMode.EDITOR
                            self.show_message("Switched to Editor Mode")
                        else:
                            self.mode = SimulationMode.SIMULATION
                            self.create_robots()
                            self.show_message("Switched to Simulation Mode")
                    elif event.key == pygame.K_p and self.mode == SimulationMode.SIMULATION:
                        # Toggle pause/resume
                        self.simulation_paused = not self.simulation_paused
                        if self.simulation_paused:
                            self.show_message("Simulation PAUSED - Press P to resume")
                        else:
                            self.show_message("Simulation RESUMED")
                            # Reset the simulation start time to account for paused time
                            pause_duration = current_time - (self.simulation_start_time + self.simulation_duration)
                            self.simulation_start_time += pause_duration
                    elif event.key == pygame.K_r and self.mode == SimulationMode.SIMULATION:
                        # Reset simulation
                        self.room = Room(GRID_WIDTH, GRID_HEIGHT)
                        self.create_robots()
                        self.duration_reached = False
                        self.simulation_paused = False
                        self.show_message("New room generated")
                    elif event.key == pygame.K_s and self.mode == SimulationMode.SIMULATION:
                        # Quick save in simulation mode
                        self.room.save_to_file("quick_save")
                        self.show_message("Room quick-saved")
                    elif event.key == pygame.K_l and self.mode == SimulationMode.SIMULATION:
                        # Quick load in simulation mode
                        if self.room.load_from_file("quick_save"):
                            self.create_robots()
                            self.duration_reached = False
                            self.simulation_paused = False
                            self.show_message("Room quick-loaded")
                        else:
                            self.show_message("No quick-save found")

                # Handle editor-specific input
                if self.mode == SimulationMode.EDITOR:
                    self.handle_editor_input(event)

            # Update robots only in simulation mode and when not paused
            if self.mode == SimulationMode.SIMULATION and not self.simulation_paused:
                # Update simulation duration
                self.simulation_duration = current_time - self.simulation_start_time

                for robot in self.robots:
                    robot.update(current_time, self.robots)
            elif self.mode == SimulationMode.SIMULATION and self.simulation_paused:
                # Keep duration static when paused
                pass
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_grid()

            # Only draw robots in simulation mode
            if self.mode == SimulationMode.SIMULATION:
                self.draw_robots()

            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()