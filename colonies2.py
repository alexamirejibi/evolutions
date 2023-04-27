import bisect
from math import sqrt
import math
import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import copy
import functools
import matplotlib.pyplot as plt
import pickle


# Parameters
screen_size = 500
num_agents = 50
agent_radius = 5
speed = 5
perception_radius = 100
signal_radius = 50
num_generations = 1
num_steps = 400
mutation_rate = 0.05
num_foods = 40
num_food_locations = 4
all_signals = []

# device = torch.device('mps')
device = torch.device('cpu')


# Initialize Pygame

class Colony:
    def __init__(self, x, y, radius=20):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), self.radius)
    
    def randomize(self):
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)

class FoodLocation:
    def __init__(self, x, y, num_foods=20):
        self.x = x
        self.y = y
        self.num_foods = num_foods
        self.foods = num_foods
        self.radius = num_foods

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius)
        
    def signal_strength(self):
        return self.foods / (self.num_foods * 2)

    def update_food(self):
        self.foods -= 1
        self.radius -= 1
        # print(self.radius)

    def has_food(self):
        return self.foods > 0
    
    def reset_random(self):
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)
        self.foods = self.num_foods
        self.radius = self.foods

# Simple Neural Network
class Brain(nn.Module):
    def __init__(self, num_inputs):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc3 = nn.Linear(18, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Agent class
class Agent:
    def __init__(self, x, y, radius, colony, signal_cooldown=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.colony = colony
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.brain = Brain(9).to(device)
        self.carrying_food = False
        self.signal_cooldown = signal_cooldown
        self.signal_counter = self.signal_cooldown
        self.memory1 = 0
        self.memory2 = 0


    def step(self, signals):
        perceptions = self.sense(signals)
        # print(perceptions)
        with torch.no_grad():
            input_data = [0.0] * 9
            if perceptions:
                dx1, dy1, signal_strength = perceptions
                input_data[0] = dx1 / screen_size
                input_data[1] = dy1 / screen_size
                input_data[2] = min(1, signal_strength)
                dx2, dy2, signal_strength = perceptions
                input_data[3] = dx1 / screen_size
                input_data[4] = dy1 / screen_size
                input_data[5] = min(1, signal_strength)
            input_data[6] = int(self.carrying_food)
            input_data[7] = self.memory1
            input_data[8] = self.memory2
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
            outputs = self.brain(input_tensor).cpu()
            signal_to_send, dx, dy, memory1, memory2 = outputs.numpy()
            self.memory1 = memory1
            self.memory2 = memory2
            dx, dy = dx / (abs(dx) + abs(dy)), dy / (abs(dx) + abs(dy))
            self.x = (self.x + speed * dx) % screen_size
            self.y = (self.y + speed * dy) % screen_size
        self.signal_counter -= 1
        if self.carrying_food:
            signal_to_send = 1
        else:
            signal_to_send = signal_to_send * 0.7
        return signal_to_send
    
    def reset(self):
        self.signal_counter = self.signal_cooldown
        self.carrying_food = False
        self.memory1 = 0
        self.memory2 = 0
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)


    def sense(self, signals):
        perception_radius = signal_radius
        combined_signal_strength = 0
        combined_dx, combined_dy = 0, 0

        for signal in signals:
            distance = math.sqrt((self.x - signal.x)**2 + (self.y - signal.y)**2)
            if distance <= perception_radius:
                weight = 1 / (distance + 1e-6)  # Add a small epsilon to avoid division by zero
                combined_signal_strength += signal.strength * weight
                combined_dx += (signal.x - self.x) * weight
                combined_dy += (signal.y - self.y) * weight

        if combined_signal_strength > 0:
            combined_dx /= combined_signal_strength
            combined_dy /= combined_signal_strength
            return combined_dx, combined_dy, combined_signal_strength
        else:
            return None

    
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
        self.strength *= 0.97
        return self.strength
    
    def draw(self, screen):
        color = (0, 0, 255, min(128, max(10, self.strength * 10)))
        print(self.strength)
        
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), signal_radius)         
        
            
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
        
    def randomize_reset(self):
        for agent in agents:
            agent.reset()
        
        for food in self.food_locations:
            food.reset_random()
            
        colony.randomize()
        
        self.signals = []
        self.steps = 0
        self.fitness_scores = [0] * len(agents)
    
    
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

        child = Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, parent.colony, self.signal_cooldown)
        child.brain = child_brain
        return child


    def update_signals(self):
        for i, signal in enumerate(self.signals):
            if signal.update() < 0.05:
                self.signals.pop(i)
    
    
    # run 1 generation and return fitness scores
    def evaluate_fitness(self, num_steps, screen=None, clock=None):
        self.randomize_reset()
        for _ in range(num_steps):
            self.step()
            if screen != None and clock != None:
                screen.fill((0, 0, 0))
                self.draw(screen)
                pygame.display.flip()
                clock.tick(30)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        screen=None
                        clock=None
        return self.fitness_scores
    
    def evolution(self, num_generations=num_generations, num_steps=num_steps, display=False):
        if display:
                pygame.init()
                screen = pygame.display.set_mode((screen_size, screen_size))
                pygame.display.set_caption('Random Population Movement with Senses')
                clock = pygame.time.Clock()
        else:
            screen = None
            clock = None
                    
        avg_fitness_top_25_percent = []
        for generation in range(num_generations):
            fitness_scores = self.evaluate_fitness(num_steps, screen=screen, clock=clock)
            parents = self.select_parents()
            offspring = [self.breed(random.choice(parents), mutation_rate) for _ in range(self.num_agents - self.num_parents)]
            agents = parents + offspring
            top_25_percent = sorted(fitness_scores)[-num_agents // 4:]
            mean_top_25_percent = np.mean(top_25_percent)
            avg_fitness_top_25_percent.append(mean_top_25_percent)
            
            if generation % 5 == 0:
                print(f"Generation: {generation}, Mean Fitness of Top 25%: {mean_top_25_percent}")

        return agents, avg_fitness_top_25_percent
    
    
def create_agents_and_colony(num_agents, agent_radius):
    colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, colony) for _ in range(num_agents)]
    return agents, colony

def create_food_locations(num_locations, num_foods):
    food_locations = [FoodLocation(random.randint(0, screen_size), random.randint(0, screen_size), num_foods) for _ in range(num_locations)]
    return food_locations

def save_agents(agents, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(agents, f)
        
def load_agents(file_name):
    with open(file_name, 'rb') as f:
        agents = pickle.load(f)
    return agents



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
    
    epochs = 100000
    save_interval = 100
    avg_data = []

    # for epoch in range(epochs):
    #     print("Epoch:", epoch)
    #     trained_agents, avg_data_interval = world.evolution(num_generations=save_interval, num_steps=500, display=False)
    #     avg_data.extend(avg_data_interval)
    #     save_agents(trained_agents, f'trained_agents_epoch{epoch}.pickle')
    
    trained_agents = load_agents("trained_agents_epoch157.pickle")
    # plot_avg_fitness(avg_data)
    # print(avg_data)
    main(trained_agents)