import bisect
from math import sqrt
import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import copy
import functools
import matplotlib.pyplot as plt


# Parameters
screen_size = 800
num_agents = 50
agent_radius = 5
speed = 3
perception_radius = 400
signal_radius = 200
num_generations = 1
num_steps = 300
mutation_rate = 0.1
num_foods = 40
num_food_locations = 4
all_signals = []

# Initialize Pygame

class Colony:
    def __init__(self, x, y, radius=20):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), self.radius)

class FoodLocation:
    def __init__(self, x, y, num_foods=20):
        self.x = x
        self.y = y
        self.num_foods = num_foods
        self.radius = num_foods

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius)
        
    def signal_strength(self):
        return self.num_foods / (num_foods * 2)

    def update_food(self):
        self.num_foods -= 1
        self.radius -= 1
        # print(self.radius)

    def has_food(self):
        return self.num_foods > 0

# Simple Neural Network
class Brain(nn.Module):
    def __init__(self, num_inputs):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 12)
        self.fc2 = nn.Linear(12, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Agent class
class Agent:
    def __init__(self, x, y, radius, colony, signal_cooldown=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.colony = colony
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.brain = Brain(4)
        self.carrying_food = False
        self.signal_cooldown = signal_cooldown
        self.signal_counter = self.signal_cooldown


    def step(self, signals):
        perceptions = self.sense(signals)
        with torch.no_grad():
            input_data = [0.0] * 4
            if perceptions:
                dx, dy, signal_strength = perceptions[0]
                input_data[0] = dx / screen_size
                input_data[1] = dy / screen_size
                input_data[2] = min(1, signal_strength)
                input_data[3] = int(self.carrying_food)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            outputs = self.brain(input_tensor)
            signal_to_send, dx, dy = outputs.numpy()
            dx, dy = dx / (abs(dx) + abs(dy)), dy / (abs(dx) + abs(dy))
            self.x = (self.x + speed * dx) % screen_size
            self.y = (self.y + speed * dy) % screen_size
        self.signal_counter -= 1
        return signal_to_send
    
    def reset(self):
        self.signal_counter = self.signal_cooldown
        self.carrying_food = False


    def sense(self, signals):
        if not signals:
            return []
        perceptions = []
        min_distance = float('inf')
        min_index = -1
        for i, signal in enumerate(signals):
            dx = signal.x - self.x
            dy = signal.y - self.y
            distance_squared = dx ** 2 + dy ** 2
            if distance_squared <= signal_radius ** 2 and distance_squared < min_distance:
                min_distance = distance_squared
                min_index = i
                
        if min_index >= 0:
            x = 1 - ((signals[min_index].x - self.x) / screen_size)
            y = 1 - ((signals[min_index].y - self.y) / screen_size)
            perceptions.append((x, y, signals[min_index].frequency))
                    
        return perceptions
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        if self.carrying_food:
            pygame.draw.circle(screen, (0, 0, 255), (self.x, self.y), agent_radius // 2)

    def collect_food(self, food_locations):
        for location in food_locations:
            distance = np.sqrt((self.x - location.x) ** 2 + (self.y - location.y) ** 2)
            if distance < self.radius + location.radius and location.has_food():
                location.update_food()
                self.carrying_food = True
                return 0.5
        return 0

    def deposit_food(self):
        distance_to_colony = np.sqrt((self.x - self.colony.x) ** 2 + (self.y - self.colony.y) ** 2)
        if distance_to_colony < self.radius + self.colony.radius and self.carrying_food:
            self.carrying_food = False
            return 1
        return 0
    
class Signal:
    def __init__(self, frequency, x, y):
        self.frequency = max(0, min(1, frequency))
        self.strength = frequency * 50
        self.radius = self.strength
        self.x = x
        self.y = y
    
    def update(self): # need to keep food signals always (???)
        self.strength * 0.9
        return self.strength
    
    def draw(self, screen):
        # Set the color to a semi-transparent blue
        color = (0, 0, 255, 128)
        
        # Create a temporary Surface object to draw the circle on
        # temp_surface = pygame.Surface((screen_size, screen_size), pygame.SRCALPHA)

        # Draw the semi-transparent circle on the temporary surface
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(self.strength))

        # Blit the temporary surface onto the main screen
        # screen.blit(temp_surface, (0, 0))
         
        
            
class World:
    def __init__(self, agents, food_locations, colony, selection_rate, signal_cooldown=10):
        self.agents = agents
        self.food_locations = food_locations
        self.colony = colony
        self.signals = []
        self.steps = 0
        self.fitness_scores = [0] * len(agents)
        self.num_agents = len(agents)
        self.num_parents = int(self.num_agents * selection_rate)
        self.signal_cooldown = signal_cooldown
        for a in self.agents:
            a.colony = self.colony
    
    def step(self):
        self.update_signals()
        for i, agent in enumerate(self.agents):
            # print(i)
            if not agent.carrying_food:
                self.fitness_scores[i] += agent.collect_food(self.food_locations)
            else:
                self.fitness_scores[i] += agent.deposit_food()
            
            signal = agent.step(self.signals)
            while len(self.signals) >= 200:
                self.signals.pop()
            
            if agent.signal_counter > 0:
                continue
            
            agent.signal_counter = self.signal_cooldown
            
            signal = Signal(signal, agent.x, agent.y)
            
            if self.signals:
                bisect.insort_left(self.signals, signal, key=lambda x: -x.strength)
            else:
                # print("appending:", signal)
                self.signals.append(signal)
            
                
                
    def draw(self, screen):
        temp_surface = pygame.Surface((screen_size, screen_size), pygame.SRCALPHA)
        
        for signal in self.signals:
            signal.draw(temp_surface)
            
        screen.blit(temp_surface, (0, 0))
            
        for agent in self.agents:
            agent.draw(screen)
        
        for food in self.food_locations:
            food.draw(screen)
            
        self.colony.draw(screen)
    
    
    def select_parents(self):
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        best_agents_indices = sorted_indices[:self.num_parents]
        parents = [agents[i] for i in best_agents_indices]
        return parents
    
    
    def breed(self, parent, mutation_rate):
        child_brain = copy.deepcopy(parent.brain)
        for child_param in child_brain.parameters():
            mutation = torch.randn_like(child_param) * mutation_rate
            child_param.data.add_(mutation)

        child = Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, parent.colony)
        child.brain = child_brain
        return child


    def update_signals(self):
        for i, signal in enumerate(self.signals):
            if signal.update() <= 0:
                self.signals.pop(i)
    
    
    # run 1 generation and return fitness scores
    def evaluate_fitness(self, num_steps):
        self.signals = []
        self.steps = 0
        self.fitness_scores = [0] * len(agents)
        for agent in self.agents:
            agent.reset()
        for _ in range(num_steps):
            self.step()
        return self.fitness_scores
    
    def evolution(self, num_generations=num_generations, num_steps=num_steps):
        avg_fitness_top_25_percent = []
        for generation in range(num_generations):
            fitness_scores = self.evaluate_fitness(num_steps)
            parents = self.select_parents()
            offspring = [self.breed(random.choice(parents), mutation_rate) for _ in range(num_agents - self.num_parents)]
            agents = parents + offspring
            
            top_25_percent = sorted(fitness_scores)[-num_agents // 4:]
            mean_top_25_percent = np.mean(top_25_percent)
            avg_fitness_top_25_percent.append(mean_top_25_percent)
            if generation % 1 == 0:
                print(f"Generation: {generation}, Mean Fitness of Top 25%: {mean_top_25_percent}")
                
        return agents, avg_fitness_top_25_percent
    
    
def create_agents_and_colony(num_agents, agent_radius):
    colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, colony) for _ in range(num_agents)]
    return agents, colony

def create_food_locations(num_locations, num_foods):
    food_locations = [FoodLocation(random.randint(0, screen_size), random.randint(0, screen_size), num_foods) for _ in range(num_locations)]
    return food_locations


def plot_avg_fitness(avg_fitness_top_25_percent):
    plt.plot(avg_fitness_top_25_percent)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness of Top 25%')
    plt.title('Evolution of Fitness over Generations')
    plt.show()

def main(agents):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Random Population Movement with Senses')
    clock = pygame.time.Clock()

    new_colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    new_food_locations = create_food_locations(num_food_locations, num_foods)
    for a in agents:
        a.colony = new_colony
        
    world = World(agents, new_food_locations, new_colony, selection_rate=0.15) # top 15% multiply

    running = True
    while running:
        screen.fill((0, 0, 0))
        world.step()
        world.draw(screen)
        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == '__main__':
    agents, colony = create_agents_and_colony(num_agents, agent_radius)
    food_locations = create_food_locations(num_food_locations, num_foods)
    world = World(agents, food_locations, colony, selection_rate=0.15) # top 15% multiply
    trained_agents, avg_data = world.evolution(num_generations=5, num_steps=300)
    plot_avg_fitness(avg_data)
    print(avg_data)
    main(agents)