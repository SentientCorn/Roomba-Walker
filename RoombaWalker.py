import random
import json
import time
from enum import Enum

class FloorState(Enum):
    CLEAN = 0
    DIRTY = 1
    OBSTACLE = 2
    ROOMBA = 8

    def __str__(self):
        return self.name.lower()

    def __int__(self):
        return self.value

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)

    def __str__(self):
        return self.name

class Room:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[FloorState.CLEAN for _ in range(width)] for _ in range(height)]

    def __str__(self):
        return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in self.grid)

    def populate(self, dirty_percentage=0.2, obstacle_percentage=0.1):
        for y in range(self.height):
            for x in range(self.width):
                rand_val = random.random()
                if rand_val < obstacle_percentage:
                    self.set_tile(x, y, FloorState.OBSTACLE)
                elif rand_val < obstacle_percentage + dirty_percentage:
                    self.set_tile(x, y, FloorState.DIRTY)

    def set_tile(self, x, y, state):
        if self.is_within_bounds(x, y):
            self.grid[y][x] = state
        else:
            raise IndexError("Coordinates out of bounds")

    def get_tile(self, x, y):
        if self.is_within_bounds(x, y):
            return self.grid[y][x]
        raise IndexError("Coordinates out of bounds")

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def save(self, filename):
        with open(filename + '.room', 'w') as f:
            for row in self.grid:
                f.write(' '.join(str(int(cell)) for cell in row) + '\n')

    def load(self, filename):
        with open(filename, 'r') as f:
            self.grid = [
                [FloorState(int(val)) for val in line.strip().split()]
                for line in f.readlines()
            ]
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0

class Roomba:
    def __init__(self, strategy_name=''):
        if not strategy_name:
            raise ValueError("No strategy provided.")
        if strategy_name not in Strategies.get_registry():
            raise ValueError(f"Strategy '{strategy_name}' not found in registry.")
        self.x = 0
        self.y = 0
        self.strategy_name = strategy_name
        self.strategy = Strategies.get_registry()[strategy_name]

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y

    def decide_direction(self, room):
        return self.strategy(room, self)

class Strategies:
    @staticmethod
    def random_strategy(room, roomba):
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.STAY]
        return random.choice(directions)

    @staticmethod
    def clean_strategy(room, roomba):
        return Direction.STAY  # Placeholder for future logic

    @staticmethod
    def get_registry():
        return {
            "random_strategy": Strategies.random_strategy,
            "clean_strategy": Strategies.clean_strategy
        }

class Simulation:
    def __init__(self, room, roomba):
        self.room = room
        self.roomba = roomba

    def __str__(self):
        return f"Simulation with Roomba at ({self.roomba.x}, {self.roomba.y}) with strategy {self.roomba.strategy_name} in room:\n{self.room}"

    def step(self):
        direction = self.roomba.decide_direction(self.room)
        if direction == Direction.STAY:
            return

        # Clean current tile
        self.room.set_tile(self.roomba.x, self.roomba.y, FloorState.CLEAN)
        dx, dy = direction.value
        new_x = self.roomba.x + dx
        new_y = self.roomba.y + dy

        if not self.room.is_within_bounds(new_x, new_y):
            return

        next_tile = self.room.get_tile(new_x, new_y)
        if next_tile == FloorState.OBSTACLE:
            return

        if next_tile == FloorState.DIRTY:
            self.room.set_tile(new_x, new_y, FloorState.CLEAN)

        self.roomba.set_position(new_x, new_y)
        self.room.set_tile(new_x, new_y, FloorState.ROOMBA)

    def render(self):
        return f"{self.room}\nRoomba at {self.roomba.get_position()} using '{self.roomba.strategy_name}' strategy"

    def save(self, filename):
        self.room.save(filename)
        data = {
            "roomfile": filename + '.room',
            "roomba": {
                "x": self.roomba.x,
                "y": self.roomba.y,
                "strategy": self.roomba.strategy_name
            }
        }
        with open(filename + '.simul', 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.room.load(data['roomfile'])
            self.roomba.set_position(data['roomba']['x'], data['roomba']['y'])
            self.roomba.strategy_name = data['roomba']['strategy']
            if self.roomba.strategy_name not in Strategies.get_registry():
                raise ValueError(f"Strategy '{self.roomba.strategy_name}' not found in registry.")
            self.roomba.strategy = Strategies.get_registry()[self.roomba.strategy_name]

if __name__ == "__main__":
    room = Room(5, 5)
    room.populate(dirty_percentage=0.2, obstacle_percentage=0.1)
    room.save('room')

    roomba = Roomba('random_strategy')
    roomba.set_position(0, 0)
    simulation = Simulation(room, roomba)

    for _ in range(10):
        simulation.step()
        print(simulation.render())
        time.sleep(1)
