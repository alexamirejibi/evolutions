import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import copy
import functools

# Parameters
screen_size = 800
num_agents = 50
agent_radius = 5
speed = 5
perception_radius = 200
num_generations = 1000
num_steps = 300
mutation_rate = 0.2
num_foods = 25

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption('Random Population Movement with Senses')
clock = pygame.time.Clock()

# Simple Neural Network
class Brain(nn.Module):
    def __init__(self, num_inputs):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 12)  # Increase the number of neurons in the first layer
        self.fc2 = nn.Linear(12, 2)  # Increase the number of neurons in the first layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Agent class
class Agent:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.max_objects = 2
        self.brain = Brain(2 + 2 * self.max_objects)  # without object type

    def sense(self, foods):
        perceptions = np.empty((0, 2))

        food_coords = np.array([[food.x, food.y] for food in foods])
        dx_dy = food_coords - np.array([self.x, self.y])
        distances_squared = np.sum(dx_dy**2, axis=1)
        mask = distances_squared <= perception_radius**2

        perceptions = dx_dy[mask]
        sorted_indices = np.argsort(np.sum(perceptions**2, axis=1))
        perceptions = perceptions[sorted_indices][:self.max_objects]

        return perceptions

    def move(self, foods):
        perceptions = self.sense(foods)
        with torch.no_grad():
            input_data = [self.x / screen_size, self.y / screen_size] + [0.0] * 2 * (self.max_objects)
            for i, (dx, dy) in enumerate(perceptions):
                input_data[2 + 2 * i] = dx / screen_size
                input_data[2 + 2 * i + 1] = dy / screen_size
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            direction = self.brain(input_tensor)
            dx, dy = direction.numpy()
            dx, dy = dx / (abs(dx) + abs(dy)), dy / (abs(dx) + abs(dy))
            self.x = (self.x + speed * dx) % screen_size
            self.y = (self.y + speed * dy) % screen_size
            
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Food:
    def __init__(self, x, y, radius=3):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (self.x, self.y), self.radius)

    def update_position(self):
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)

def collect_food(agent, foods):
    for food in foods:
        distance = np.sqrt((agent.x - food.x)**2 + (agent.y - food.y)**2)
        if distance < agent.radius + food.radius:
            # foods.remove(food)
            food.update_position()
            # foods.append(food)
            return 1
    return 0

def select_parents(agents, fitness_scores, num_parents):
    sorted_indices = np.argsort(fitness_scores)[::-1]
    best_agents_indices = sorted_indices[:num_parents]
    parents = [agents[i] for i in best_agents_indices]
    return parents

def evaluate_fitness(agents, num_steps, num_foods=100):
    foods = [Food(random.randint(0, screen_size), random.randint(0, screen_size)) for _ in range(num_foods)]
    fitness_scores = [0] * len(agents)
    for _ in range(num_steps):
        for i, agent in enumerate(agents):
            agent.move(foods)
            fitness_scores[i] += collect_food(agent, foods)
    return fitness_scores

def breed(parent, mutation_rate):
    child_brain = copy.deepcopy(parent.brain)
    for child_param in child_brain.parameters():
        if len(child_param.shape) == 2:
            mutation_mask = torch.tensor(
                [[random.random() < mutation_rate for _ in range(child_param.shape[1])]
                 for _ in range(child_param.shape[0])], dtype=torch.bool)
        elif len(child_param.shape) == 1:
            mutation_mask = torch.tensor(
                [random.random() < mutation_rate for _ in range(child_param.shape[0])], dtype=torch.bool)
        else:
            continue

        mutation = torch.randn_like(child_param) * mutation_rate
        child_param.data.add_(mutation * mutation_mask)

    child = Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius)
    child.brain = child_brain
    return child

import matplotlib.pyplot as plt

def run_genetic_algorithm(num_foods):
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius) for _ in range(num_agents)]

    num_parents = num_agents // 5
    avg_fitness_top_25_percent = []
    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(agents, num_steps, num_foods)
        parents = select_parents(agents, fitness_scores, num_parents)
        offspring = [breed(random.choice(parents), mutation_rate) for _ in range(num_agents - num_parents)]
        agents = parents + offspring

        top_25_percent = sorted(fitness_scores)[-num_agents // 4:]
        mean_top_25_percent = np.mean(top_25_percent)
        avg_fitness_top_25_percent.append(mean_top_25_percent)
        if generation % 1 == 0:
            print(f"Generation: {generation}, Mean Fitness of Top 25%: {mean_top_25_percent}")
            
    return agents, avg_fitness_top_25_percent

def plot_avg_fitness(avg_fitness_top_25_percent):
    plt.plot(avg_fitness_top_25_percent)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness of Top 25%')
    plt.title('Evolution of Fitness over Generations')
    plt.show()

trained_agents, avg_data = run_genetic_algorithm(100)
plot_avg_fitness(avg_data)

# Main loop
running = True
foods = [Food(random.randint(0, screen_size), random.randint(0, screen_size)) for _ in range(num_foods)]
while running:
    screen.fill((0, 0, 0))

    for agent in trained_agents:
        agent.move(foods)
        collect_food(agent, foods)
        agent.draw(screen)

    for food in foods:
        food.draw(screen)

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()


