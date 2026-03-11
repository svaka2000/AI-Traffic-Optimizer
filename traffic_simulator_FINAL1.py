# AI Traffic Simulator & Optimizer
# Stage 5: Complete System with Metrics & CSV Logging
#
# Author: High School Science Fair Project
# Description: Full comparison system with data collection for science fair analysis

import pygame
import sys
import random
import csv
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 200, 0)
ORANGE = (255, 165, 0)
DARK_RED = (139, 0, 0)
DARK_GREEN = (0, 100, 0)
DARK_YELLOW = (139, 139, 0)
CYAN = (0, 255, 255)
PURPLE = (200, 0, 200)

ROAD_WIDTH = 200
LANE_WIDTH = ROAD_WIDTH // 2

CAR_WIDTH = 30
CAR_HEIGHT = 50
CAR_SPEED = 3
CAR_SPACING = 10

LIGHT_RADIUS = 15
LIGHT_OFFSET = 80

GREEN_DURATION = 10
YELLOW_DURATION = 2

AI_DECISION_INTERVAL = 60
AI_MIN_GREEN_TIME = 180
AI_YELLOW_TIME = 120

SPAWN_RATE = 60

# Simulation run settings
RUN_DURATION = 120  # Run for 120 seconds (2 minutes) per test

FPS = 60

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def lighten(color, amount=80):
    """Return a lightened version of an RGB color by adding the given amount to each channel."""
    r, g, b = color
    return (min(r + amount, 255), min(g + amount, 255), min(b + amount, 255))


# ============================================================================
# TRAFFIC LIGHT CLASS
# ============================================================================

class TrafficLight:
    def __init__(self, direction):
        self.direction = direction
        self.state = 'RED'
        self.timer = 0
        self.state_changes = 0  # Track number of state changes
        
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2

        # Position lights near the intersection corners with consistent offsets
        # The lights are placed outside of the intersection square and offset along the axis.
        offset = ROAD_WIDTH // 2 + 30  # distance from center to place lights
        lane_offset = ROAD_WIDTH // 4   # slight lateral offset to align with lanes

        if direction == 'N':
            # Northbound light faces south; place above the intersection
            self.x = center_x - lane_offset
            self.y = center_y - offset
        elif direction == 'S':
            # Southbound light faces north; place below the intersection
            self.x = center_x + lane_offset
            self.y = center_y + offset
        elif direction == 'E':
            # Eastbound light faces west; place to the right of the intersection
            self.x = center_x + offset
            self.y = center_y - lane_offset
        elif direction == 'W':
            # Westbound light faces east; place to the left of the intersection
            self.x = center_x - offset
            self.y = center_y + lane_offset
    
    def change_state(self, new_state):
        """Change state and increment counter"""
        if new_state != self.state:
            self.state = new_state
            self.state_changes += 1
    
    def draw(self, screen):
        """Draw the traffic light with a subtle glow around the active lamp."""
        # Draw housing with subtle rounded corners
        bg_rect = pygame.Rect(
            self.x - LIGHT_RADIUS - 6,
            self.y - LIGHT_RADIUS - 6,
            LIGHT_RADIUS * 2 + 12,
            LIGHT_RADIUS * 6 + 24
        )
        pygame.draw.rect(screen, BLACK, bg_rect, border_radius=8)

        # Draw each light; add a glow by drawing a slightly larger circle with a lightened color
        # Top light (Red)
        red_active = (self.state == 'RED')
        red_color = RED if red_active else DARK_RED
        if red_active:
            glow_color = lighten(RED, 80)
            pygame.draw.circle(screen, glow_color, (self.x, self.y), LIGHT_RADIUS + 4)
        pygame.draw.circle(screen, red_color, (self.x, self.y), LIGHT_RADIUS)

        # Middle light (Yellow)
        yellow_active = (self.state == 'YELLOW')
        yellow_color = YELLOW if yellow_active else DARK_YELLOW
        middle_center = (self.x, self.y + LIGHT_RADIUS * 2)
        if yellow_active:
            glow_color = lighten(YELLOW, 80)
            pygame.draw.circle(screen, glow_color, middle_center, LIGHT_RADIUS + 4)
        pygame.draw.circle(screen, yellow_color, middle_center, LIGHT_RADIUS)

        # Bottom light (Green)
        green_active = (self.state == 'GREEN')
        green_color = GREEN if green_active else DARK_GREEN
        bottom_center = (self.x, self.y + LIGHT_RADIUS * 4)
        if green_active:
            glow_color = lighten(GREEN, 80)
            pygame.draw.circle(screen, glow_color, bottom_center, LIGHT_RADIUS + 4)
        pygame.draw.circle(screen, green_color, bottom_center, LIGHT_RADIUS)


# ============================================================================
# CAR CLASS (Enhanced with wait time tracking)
# ============================================================================

class Car:
    def __init__(self, x, y, direction, color, spawn_frame):
        self.x = x
        self.y = y
        self.direction = direction
        self.color = color
        self.speed = CAR_SPEED
        self.stopped = False
        
        # Tracking variables for metrics
        self.spawn_frame = spawn_frame
        self.wait_time = 0  # Total frames spent waiting
        self.crossed_intersection = False
        self.cross_frame = None
        
        if direction in ['N', 'S']:
            self.width = CAR_WIDTH
            self.height = CAR_HEIGHT
        else:
            self.width = CAR_HEIGHT
            self.height = CAR_WIDTH
    
    def get_stop_line_position(self):
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        if self.direction == 'N':
            return center_y - ROAD_WIDTH // 2 - self.height
        elif self.direction == 'S':
            return center_y + ROAD_WIDTH // 2
        elif self.direction == 'E':
            return center_x + ROAD_WIDTH // 2
        elif self.direction == 'W':
            return center_x - ROAD_WIDTH // 2 - self.width
    
    def has_crossed_intersection(self):
        """Check if car has passed through the intersection"""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        if self.direction == 'N':
            return self.y < center_y - ROAD_WIDTH // 2
        elif self.direction == 'S':
            return self.y > center_y + ROAD_WIDTH // 2
        elif self.direction == 'E':
            return self.x > center_x + ROAD_WIDTH // 2
        elif self.direction == 'W':
            return self.x < center_x - ROAD_WIDTH // 2
        
        return False
    
    def is_waiting_at_light(self, traffic_light):
        if traffic_light.state not in ['RED', 'YELLOW']:
            return False
        
        stop_line = self.get_stop_line_position()
        threshold = 50
        
        if self.direction == 'N':
            return stop_line - threshold < self.y < stop_line + 10
        elif self.direction == 'S':
            return stop_line - 10 < self.y < stop_line + threshold
        elif self.direction == 'E':
            return stop_line - 10 < self.x < stop_line + threshold
        elif self.direction == 'W':
            return stop_line - threshold < self.x < stop_line + 10
        
        return False
    
    def should_stop(self, traffic_light, cars_ahead):
        stop_line = self.get_stop_line_position()
        
        # Check distance to car ahead to maintain spacing
        for other_car in cars_ahead:
            if other_car.direction == self.direction:
                if self.direction == 'N':
                    distance = self.y - (other_car.y + other_car.height)
                    if 0 < distance < CAR_SPACING:
                        return True
                elif self.direction == 'S':
                    distance = other_car.y - (self.y + self.height)
                    if 0 < distance < CAR_SPACING:
                        return True
                elif self.direction == 'E':
                    distance = other_car.x - (self.x + self.width)
                    if 0 < distance < CAR_SPACING:
                        return True
                elif self.direction == 'W':
                    distance = self.x - (other_car.x + other_car.width)
                    if 0 < distance < CAR_SPACING:
                        return True
        
        # Obey traffic light
        if traffic_light.state in ['RED', 'YELLOW']:
            if self.direction == 'N':
                if self.y > stop_line:
                    return True
            elif self.direction == 'S':
                if self.y < stop_line:
                    return True
            elif self.direction == 'E':
                if self.x < stop_line:
                    return True
            elif self.direction == 'W':
                if self.x > stop_line:
                    return True
        
        return False
    
    def update(self, traffic_light, other_cars, current_frame):
        self.stopped = self.should_stop(traffic_light, other_cars)
        
        # Track wait time
        if self.stopped:
            self.wait_time += 1
        
        # Check if crossed intersection; record the frame when crossing occurs
        if not self.crossed_intersection and self.has_crossed_intersection():
            self.crossed_intersection = True
            self.cross_frame = current_frame
        
        # Move the car if not stopped
        if not self.stopped:
            if self.direction == 'N':
                self.y -= self.speed
            elif self.direction == 'S':
                self.y += self.speed
            elif self.direction == 'E':
                self.x += self.speed
            elif self.direction == 'W':
                self.x -= self.speed
    
    def draw(self, screen):
        """Draw the car with rounded corners and simple detailing to indicate direction."""
        car_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        # Base shape with rounded corners
        pygame.draw.rect(screen, self.color, car_rect, border_radius=6)
        # Outline
        pygame.draw.rect(screen, BLACK, car_rect, 2, border_radius=6)
        # Windshield / stripe for direction indication
        if self.direction in ['N', 'S']:
            # Horizontal windshield across the top quarter
            windshield_y = self.y + int(self.height * 0.25)
            start_pos = (self.x + 3, windshield_y)
            end_pos = (self.x + self.width - 3, windshield_y)
            pygame.draw.line(screen, lighten(self.color, 60), start_pos, end_pos, 2)
            # Vertical center stripe to show front/back orientation
            center_x = self.x + self.width // 2
            pygame.draw.line(screen, BLACK, (center_x, self.y + 2), (center_x, self.y + self.height - 2), 1)
        else:
            # Vertical windshield along the left quarter
            windshield_x = self.x + int(self.width * 0.25)
            start_pos = (windshield_x, self.y + 3)
            end_pos = (windshield_x, self.y + self.height - 3)
            pygame.draw.line(screen, lighten(self.color, 60), start_pos, end_pos, 2)
            # Horizontal center stripe to show front/back orientation
            center_y = self.y + self.height // 2
            pygame.draw.line(screen, BLACK, (self.x + 2, center_y), (self.x + self.width - 2, center_y), 1)
    
    def is_off_screen(self):
        if self.direction == 'N' and self.y < -self.height:
            return True
        elif self.direction == 'S' and self.y > SCREEN_HEIGHT:
            return True
        elif self.direction == 'E' and self.x > SCREEN_WIDTH:
            return True
        elif self.direction == 'W' and self.x < -self.width:
            return True
        return False


# ============================================================================
# METRICS TRACKING CLASS
# ============================================================================

class MetricsTracker:
    """Tracks all simulation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new run"""
        self.cars_crossed = 0
        self.total_wait_time = 0
        self.wait_times = []
        self.max_queue_N = 0
        self.max_queue_S = 0
        self.max_queue_E = 0
        self.max_queue_W = 0
        self.total_light_changes = 0
    
    def record_car_crossed(self, car):
        """Record when a car crosses the intersection"""
        self.cars_crossed += 1
        wait_seconds = car.wait_time / FPS
        self.wait_times.append(wait_seconds)
        self.total_wait_time += wait_seconds
    
    def update_queue_lengths(self, cars, traffic_lights):
        """Update maximum queue lengths"""
        queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        
        for car in cars:
            if car.is_waiting_at_light(traffic_lights[car.direction]):
                queues[car.direction] += 1
        
        self.max_queue_N = max(self.max_queue_N, queues['N'])
        self.max_queue_S = max(self.max_queue_S, queues['S'])
        self.max_queue_E = max(self.max_queue_E, queues['E'])
        self.max_queue_W = max(self.max_queue_W, queues['W'])
    
    def count_light_changes(self, traffic_lights):
        """Count total state changes across all lights"""
        total = sum(light.state_changes for light in traffic_lights.values())
        self.total_light_changes = total
    
    def get_average_wait_time(self):
        """Calculate average wait time"""
        if len(self.wait_times) == 0:
            return 0
        return sum(self.wait_times) / len(self.wait_times)
    
    def save_to_csv(self, mode, run_number, filename="traffic_data.csv"):
        """Save metrics to CSV file"""
        import os
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'mode', 'run_number', 
                'avg_wait_time', 'cars_cleared', 
                'max_queue_N', 'max_queue_S', 
                'max_queue_E', 'max_queue_W',
                'total_light_changes'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'mode': mode,
                'run_number': run_number,
                'avg_wait_time': round(self.get_average_wait_time(), 2),
                'cars_cleared': self.cars_crossed,
                'max_queue_N': self.max_queue_N,
                'max_queue_S': self.max_queue_S,
                'max_queue_E': self.max_queue_E,
                'max_queue_W': self.max_queue_W,
                'total_light_changes': self.total_light_changes
            })
        
        # Console output for quick review
        print(f"\n{'='*60}")
        print(f"Data saved to {filename}")
        print(f"Mode: {mode} | Run: {run_number}")
        print(f"Average Wait Time: {self.get_average_wait_time():.2f}s")
        print(f"Cars Cleared: {self.cars_crossed}")
        print(f"Max Queues - N:{self.max_queue_N} S:{self.max_queue_S} E:{self.max_queue_E} W:{self.max_queue_W}")
        print(f"Light Changes: {self.total_light_changes}")
        print(f"{'='*60}\n")


# ============================================================================
# TRAFFIC CONTROL FUNCTIONS
# ============================================================================

def update_traffic_lights_fixed(lights, frame_count):
    cycle_time = (frame_count // FPS) % (GREEN_DURATION * 2 + YELLOW_DURATION * 2)
    
    if cycle_time < GREEN_DURATION:
        # NS axis green, EW red
        for dir in ['N', 'S']:
            lights[dir].change_state('GREEN')
        for dir in ['E', 'W']:
            lights[dir].change_state('RED')
    elif cycle_time < GREEN_DURATION + YELLOW_DURATION:
        # NS axis yellow, EW red
        for dir in ['N', 'S']:
            lights[dir].change_state('YELLOW')
        for dir in ['E', 'W']:
            lights[dir].change_state('RED')
    elif cycle_time < GREEN_DURATION * 2 + YELLOW_DURATION:
        # EW axis green, NS red
        for dir in ['N', 'S']:
            lights[dir].change_state('RED')
        for dir in ['E', 'W']:
            lights[dir].change_state('GREEN')
    else:
        # EW axis yellow, NS red
        for dir in ['N', 'S']:
            lights[dir].change_state('RED')
        for dir in ['E', 'W']:
            lights[dir].change_state('YELLOW')

def count_waiting_cars(cars, direction, traffic_light):
    count = 0
    for car in cars:
        if car.direction == direction and car.is_waiting_at_light(traffic_light):
            count += 1
    return count


def update_traffic_lights_ai(lights, cars, ai_state):
    # Increment internal timers
    ai_state['decision_timer'] += 1
    ai_state['green_timer'] += 1
    
    # Count waiting cars on each direction and axis
    waiting_N = count_waiting_cars(cars, 'N', lights['N'])
    waiting_S = count_waiting_cars(cars, 'S', lights['S'])
    waiting_E = count_waiting_cars(cars, 'E', lights['E'])
    waiting_W = count_waiting_cars(cars, 'W', lights['W'])
    
    waiting_NS = waiting_N + waiting_S
    waiting_EW = waiting_E + waiting_W
    
    ai_state['waiting_counts'] = {
        'N': waiting_N, 'S': waiting_S,
        'E': waiting_E, 'W': waiting_W,
        'NS': waiting_NS, 'EW': waiting_EW
    }
    
    current_axis = ai_state['current_axis']
    
    # Decision to switch based on queue lengths and timers
    if ai_state['decision_timer'] >= AI_DECISION_INTERVAL:
        ai_state['decision_timer'] = 0
        # Only consider switching if minimum green time has elapsed
        if ai_state['green_timer'] >= AI_MIN_GREEN_TIME:
            if current_axis == 'NS' and waiting_EW > waiting_NS:
                ai_state['switching'] = True
                ai_state['switch_target'] = 'EW'
            elif current_axis == 'EW' and waiting_NS > waiting_EW:
                ai_state['switching'] = True
                ai_state['switch_target'] = 'NS'
    
    # Handle switching via a yellow phase
    if ai_state['switching']:
        ai_state['yellow_timer'] += 1
        
        # During yellow phase, set current axis lights to yellow, cross axis to red
        if ai_state['yellow_timer'] <= AI_YELLOW_TIME:
            if current_axis == 'NS':
                for dir in ['N', 'S']:
                    lights[dir].change_state('YELLOW')
                for dir in ['E', 'W']:
                    lights[dir].change_state('RED')
            else:
                for dir in ['N', 'S']:
                    lights[dir].change_state('RED')
                for dir in ['E', 'W']:
                    lights[dir].change_state('YELLOW')
        else:
            # Yellow phase ended, switch axis
            ai_state['current_axis'] = ai_state['switch_target']
            ai_state['switching'] = False
            ai_state['yellow_timer'] = 0
            ai_state['green_timer'] = 0
    else:
        # Maintain green on current axis, red on the other
        if current_axis == 'NS':
            for dir in ['N', 'S']:
                lights[dir].change_state('GREEN')
            for dir in ['E', 'W']:
                lights[dir].change_state('RED')
        else:
            for dir in ['N', 'S']:
                lights[dir].change_state('RED')
            for dir in ['E', 'W']:
                lights[dir].change_state('GREEN')


def spawn_car(direction, frame_count):
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    
    colors = [RED, BLUE, GREEN, ORANGE]
    color = random.choice(colors)
    
    if direction == 'N':
        x = center_x + LANE_WIDTH // 2 - CAR_WIDTH // 2
        y = SCREEN_HEIGHT + CAR_HEIGHT
    elif direction == 'S':
        x = center_x - LANE_WIDTH // 2 - CAR_WIDTH // 2
        y = -CAR_HEIGHT
    elif direction == 'E':
        x = -CAR_HEIGHT
        y = center_y - LANE_WIDTH // 2 - CAR_WIDTH // 2
    elif direction == 'W':
        x = SCREEN_WIDTH + CAR_HEIGHT
        y = center_y + LANE_WIDTH // 2 - CAR_WIDTH // 2
    
    return Car(x, y, direction, color, frame_count)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Traffic Simulator - Stage 5: Complete System")
    clock = pygame.time.Clock()
    
    # Create traffic lights for each approach
    traffic_lights = {
        'N': TrafficLight('N'),
        'S': TrafficLight('S'),
        'E': TrafficLight('E'),
        'W': TrafficLight('W')
    }
    
    # Default mode is FIXED
    mode = 'FIXED'
    
    # AI state for rule-based controller
    ai_state = {
        'current_axis': 'NS',
        'decision_timer': 0,
        'green_timer': 0,
        'switching': False,
        'yellow_timer': 0,
        'switch_target': None,
        'waiting_counts': {}
    }
    
    cars = []
    spawn_counter = 0
    total_cars_spawned = 0
    frame_count = 0
    run_number = 1
    
    metrics = MetricsTracker()
    
    running_simulation = False
    simulation_complete = False
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    # Switch to fixed-timer mode and reset simulation
                    mode = 'FIXED'
                    running_simulation = False
                    simulation_complete = False
                    frame_count = 0
                    cars.clear()
                    metrics.reset()
                    for light in traffic_lights.values():
                        light.state_changes = 0
                    print("Switched to FIXED-TIMER mode")
                elif event.key == pygame.K_a:
                    # Switch to AI rule-based mode and reset simulation
                    mode = 'AI'
                    running_simulation = False
                    simulation_complete = False
                    frame_count = 0
                    cars.clear()
                    metrics.reset()
                    ai_state = {
                        'current_axis': 'NS',
                        'decision_timer': 0,
                        'green_timer': 0,
                        'switching': False,
                        'yellow_timer': 0,
                        'switch_target': None,
                        'waiting_counts': {}
                    }
                    for light in traffic_lights.values():
                        light.state_changes = 0
                    print("Switched to AI RULE-BASED mode")
                elif event.key == pygame.K_SPACE:
                    # Start or proceed to next simulation run
                    if not running_simulation and not simulation_complete:
                        running_simulation = True
                        simulation_complete = False
                        frame_count = 0
                        cars.clear()
                        total_cars_spawned = 0
                        spawn_counter = 0
                        metrics.reset()
                        for light in traffic_lights.values():
                            light.state_changes = 0
                        print(f"\nStarting {mode} simulation run #{run_number}...")
                    elif simulation_complete:
                        # Prepare for next run
                        running_simulation = False
                        simulation_complete = False
                        run_number += 1
        
        # UPDATE
        if running_simulation:
            frame_count += 1
            elapsed_seconds = frame_count // FPS
            
            # Check if run duration reached
            if elapsed_seconds >= RUN_DURATION:
                running_simulation = False
                simulation_complete = True
                metrics.count_light_changes(traffic_lights)
                metrics.save_to_csv(mode, run_number)
            
            # Update traffic lights based on mode
            if mode == 'FIXED':
                update_traffic_lights_fixed(traffic_lights, frame_count)
            else:
                update_traffic_lights_ai(traffic_lights, cars, ai_state)
            
            # Spawn cars at a fixed rate
            spawn_counter += 1
            if spawn_counter >= SPAWN_RATE:
                direction = random.choice(['N', 'S', 'E', 'W'])
                new_car = spawn_car(direction, frame_count)
                cars.append(new_car)
                total_cars_spawned += 1
                spawn_counter = 0
            
            # Update car positions and record metrics when cars cross
            for car in cars:
                car.update(traffic_lights[car.direction], cars, frame_count)
                
                if car.crossed_intersection and car.cross_frame == frame_count:
                    metrics.record_car_crossed(car)
            
            metrics.update_queue_lengths(cars, traffic_lights)
            cars = [car for car in cars if not car.is_off_screen()]
        
        # DRAWING
        screen.fill(DARK_GRAY)
        draw_intersection(screen)
        
        for light in traffic_lights.values():
            light.draw(screen)
        
        for car in cars:
            car.draw(screen)
        
        draw_hud(screen, len(cars), total_cars_spawned, frame_count, mode, 
                 ai_state, running_simulation, simulation_complete, metrics, run_number)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


def draw_intersection(screen):
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    
    # Colors for road and intersection surfaces
    ROAD_COLOR = (90, 90, 90)
    INTERSECTION_COLOR = (120, 120, 120)

    # Draw vertical road strip
    vertical_road = pygame.Rect(center_x - ROAD_WIDTH // 2, 0, ROAD_WIDTH, SCREEN_HEIGHT)
    pygame.draw.rect(screen, ROAD_COLOR, vertical_road)
    # Draw horizontal road strip
    horizontal_road = pygame.Rect(0, center_y - ROAD_WIDTH // 2, SCREEN_WIDTH, ROAD_WIDTH)
    pygame.draw.rect(screen, ROAD_COLOR, horizontal_road)
    # Draw lighter intersection square on top to distinguish crossing
    intersection_rect = pygame.Rect(center_x - ROAD_WIDTH // 2, center_y - ROAD_WIDTH // 2, ROAD_WIDTH, ROAD_WIDTH)
    pygame.draw.rect(screen, INTERSECTION_COLOR, intersection_rect)
    
    # Draw dashed yellow lines down the center
    draw_dashed_line(screen, YELLOW, (center_x, 0), (center_x, SCREEN_HEIGHT), 20, 10)
    draw_dashed_line(screen, YELLOW, (0, center_y), (SCREEN_WIDTH, center_y), 20, 10)
    
    # Draw white stop lines around intersection
    line_thickness = 4
    pygame.draw.line(screen, WHITE,
                    (center_x - ROAD_WIDTH // 2, center_y - ROAD_WIDTH // 2),
                    (center_x, center_y - ROAD_WIDTH // 2), line_thickness)
    pygame.draw.line(screen, WHITE,
                    (center_x, center_y + ROAD_WIDTH // 2),
                    (center_x + ROAD_WIDTH // 2, center_y + ROAD_WIDTH // 2), line_thickness)
    pygame.draw.line(screen, WHITE,
                    (center_x + ROAD_WIDTH // 2, center_y - ROAD_WIDTH // 2),
                    (center_x + ROAD_WIDTH // 2, center_y), line_thickness)
    pygame.draw.line(screen, WHITE,
                    (center_x - ROAD_WIDTH // 2, center_y),
                    (center_x - ROAD_WIDTH // 2, center_y + ROAD_WIDTH // 2), line_thickness)

def draw_dashed_line(screen, color, start_pos, end_pos, dash_length=10, gap_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    if x1 == x2:
        # Vertical dashed line
        y = y1
        while y < y2:
            dash_end = min(y + dash_length, y2)
            pygame.draw.line(screen, color, (x1, y), (x1, dash_end), 2)
            y += dash_length + gap_length
    else:
        # Horizontal dashed line
        x = x1
        while x < x2:
            dash_end = min(x + dash_length, x2)
            pygame.draw.line(screen, color, (x, y1), (dash_end, y1), 2)
            x += dash_length + gap_length


def draw_hud(screen, active_cars, total_spawned, frame_count, mode, ai_state, 
            running_sim, sim_complete, metrics, run_number):
    font = pygame.font.Font(None, 26)
    elapsed_time = frame_count // FPS
    
    y_pos = 10
    
    # --- SYSTEM STATUS ---
    system_title = font.render("SYSTEM STATUS", True, CYAN)
    screen.blit(system_title, (10, y_pos))
    y_pos += 28

    # Mode line
    mode_color = CYAN if mode == 'AI' else WHITE
    mode_text = font.render(f"Mode: {mode}", True, mode_color)
    screen.blit(mode_text, (10, y_pos))
    y_pos += 22

    # Status line
    if running_sim:
        status_color = GREEN
        status = f"RUNNING ({elapsed_time}/{RUN_DURATION}s)"
    elif sim_complete:
        status_color = PURPLE
        status = "COMPLETE - Press SPACE for next run"
    else:
        status_color = YELLOW
        status = "Press SPACE to start run"
    status_text = font.render(f"Status: {status}", True, status_color)
    screen.blit(status_text, (10, y_pos))
    y_pos += 22

    # Run and car counts
    run_text = font.render(f"Run: {run_number}", True, WHITE)
    screen.blit(run_text, (10, y_pos))
    y_pos += 22
    cars_text = font.render(f"Active: {active_cars}  |  Spawned: {total_spawned}", True, WHITE)
    screen.blit(cars_text, (10, y_pos))
    y_pos += 28

    # --- PERFORMANCE METRICS ---
    if running_sim or sim_complete:
        perf_title = font.render("PERFORMANCE", True, CYAN)
        screen.blit(perf_title, (10, y_pos))
        y_pos += 28
        # Cars cleared
        cleared_text = font.render(f"Cars Cleared: {metrics.cars_crossed}", True, WHITE)
        screen.blit(cleared_text, (10, y_pos))
        y_pos += 22
        # Average wait
        avg_wait = metrics.get_average_wait_time()
        wait_text = font.render(f"Avg Wait: {avg_wait:.2f} s", True, WHITE)
        screen.blit(wait_text, (10, y_pos))
        y_pos += 22
        # Max queues across directions
        queue_text = font.render(
            f"Max Queue: N {metrics.max_queue_N}  |  S {metrics.max_queue_S}  |  E {metrics.max_queue_E}  |  W {metrics.max_queue_W}",
            True, WHITE
        )
        screen.blit(queue_text, (10, y_pos))
        y_pos += 26

    # --- AI DATA ---
    if mode == 'AI' and ai_state.get('waiting_counts'):
        ai_heading = font.render("AI DATA", True, CYAN)
        screen.blit(ai_heading, (10, y_pos))
        y_pos += 28
        counts = ai_state['waiting_counts']
        ns_text = font.render(f"N waiting: {counts['N']}  |  S waiting: {counts['S']}", True, WHITE)
        screen.blit(ns_text, (10, y_pos))
        y_pos += 20
        ew_text = font.render(f"E waiting: {counts['E']}  |  W waiting: {counts['W']}", True, WHITE)
        screen.blit(ew_text, (10, y_pos))
        y_pos += 20
        axis_text = font.render(
            f"Current Axis: {ai_state['current_axis']}  (NS: {counts['NS']}, EW: {counts['EW']})",
            True, WHITE
        )
        screen.blit(axis_text, (10, y_pos))
        y_pos += 20
        timers_text = font.render(
            f"Green Timer: {ai_state['green_timer']}  |  Decision: {ai_state['decision_timer']}",
            True, WHITE
        )
        screen.blit(timers_text, (10, y_pos))
        y_pos += 20
        if ai_state.get('switching', False):
            switching_text = font.render(
                f"Switching... Yellow: {ai_state['yellow_timer']} → Target: {ai_state['switch_target']}",
                True, YELLOW
            )
            screen.blit(switching_text, (10, y_pos))
            y_pos += 22


# Entry point
if __name__ == '__main__':
    main()
