# Cleaner Robot Simulation

A Python-based simulation of autonomous cleaning robots navigating and cleaning a room environment. The simulation features intelligent pathfinding, battery management, and collaborative behavior between multiple robots.

# DISCLAIMER: 
The pathfinding AI is far from perfect

## Features

- **Multi-Robot Simulation**
- **Two Robot Types**:
  - Standard robots: 1-tile vision range, lower energy consumption
  - Long-range robots: 3-tile vision range, higher energy consumption
- **Intelligent Behavior**: A* pathfinding, battery management, and knowledge sharing
- **Room Editor**: Create and modify room layouts with obstacles and charging stations
- **Real-time Statistics**: Track cleaning progress, battery usage, and efficiency metrics
- **Save/Load System**: Store and retrieve custom room layouts

## Requirements

- Python 3.7+
- Pygame library

```bash
pip install pygame
```

## Installation & Usage

1. Clone or download the simulation files
2. Install dependencies: `pip install pygame`
3. Run the simulation: `python robot_simulation.py`

## Controls

### Simulation Mode
- **SPACE** - Switch to Editor Mode
- **P** - Pause/Resume simulation
- **R** - Reset and generate new room
- **S** - Quick save current room
- **L** - Quick load saved room

### Editor Mode
- **SPACE** - Switch to Simulation Mode
- **1** - Empty tile tool
- **2** - Obstacle tool
- **3** - Charging station tool
- **4** - Void tile tool
- **C** - Clear entire room
- **G** - Generate random room
- **S** - Save room (enter filename)
- **L** - Load room (enter filename)
- **Mouse** - Click and drag to place/remove tiles

## Configuration

Edit the constants at the top of the file to customize:

- `ROBOT_COUNT`: Number of robots (1-4)
- `ALL_LONG_RANGE`: Set to `True` for all long-range robots
- `SIMULATION_DURATION_MINUTES`: Time limit (0 for unlimited)
- `GRID_WIDTH/HEIGHT`: Room dimensions

## Room Layout

- **White tiles**: Empty spaces to be cleaned
- **Dark gray**: Obstacles (impassable)
- **Yellow**: Charging stations
- **Light blue**: Cleaned tiles
- **Gray**: Void tiles (impassable)

## Robot Behavior

Robots autonomously:
- Explore unknown areas 
- Clean tiles as they move
- Return to charging stations when battery is low
- Share knowledge with nearby robots
- Avoid clustering by coordinating targets

## Statistics Tracked

- Cleaning progress and efficiency
- Battery usage and charging cycles
- Individual robot performance
- Ineffective movement patterns
- Total simulation time

## File Structure

- Saved rooms are stored in the `rooms/` directory as JSON files
- Quick-save functionality creates `rooms/quick_save.json`
