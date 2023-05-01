import math
import pickle
import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import copy
import functools
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Parameters
screen_size = 800
num_agents = 50
agent_radius = 5
speed = 5
perception_radius = 400
num_generations = 1
num_steps = 300
mutation_rate = 0.1
num_foods = 25



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
        self.max_objects = 1
        self.brain = Brain(2 * self.max_objects)  # without object type

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
            input_data = [0.0] * 2 * (self.max_objects)
            for i, (dx, dy) in enumerate(perceptions):
                input_data[2 * i] = dx / screen_size
                input_data[2 * i + 1] = dy / screen_size
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
    return parents, best_agents_indices[0]

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
        mutation = torch.randn_like(child_param) * mutation_rate
        if random.random() < mutation_rate:
            child_param.data.add_(mutation)

    child = Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius)
    child.brain = child_brain
    return child


import matplotlib.pyplot as plt

def run_genetic_algorithm(num_foods, num_generations=num_generations, num_agents=num_agents, mutation_rate=mutation_rate, agents=None, steps=500):
    if agents == None:
        agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius) for _ in range(num_agents)]
    num_parents = num_agents // 5
    avg_fitness_top_25_percent = []
    tmp_mutation_rate = mutation_rate
    max_mean_top_25_percent = 30.0 * num_foods / num_agents
    best_agent = agents[0]
    best_score = 0
    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(agents, steps, num_foods)
        parents, best_agent_idx = select_parents(agents, fitness_scores, num_parents)
        
        if fitness_scores[best_agent_idx] > best_score:
            best_score = fitness_scores[best_agent_idx]
            best_agent = agents[best_agent_idx]
        
        offspring = [breed(random.choice(parents), tmp_mutation_rate) for _ in range(num_agents - num_parents)]
        agents = parents + offspring
        top_25_percent = sorted(fitness_scores)[-num_agents // 4:]
        mean_top_25_percent = np.mean(top_25_percent)
        avg_fitness_top_25_percent.append(mean_top_25_percent)
        if generation % 1 == 0:
            print(f"Generation: {generation}, Mean Fitness of Top 25%: {mean_top_25_percent}")
        tmp_mutation_rate = mutation_rate * (max_mean_top_25_percent - mean_top_25_percent) / max_mean_top_25_percent
        print(f"Dynamic mutation rate: {tmp_mutation_rate}")
            
    return agents, avg_fitness_top_25_percent, best_agent

def plot_avg_fitness(avg_fitness_top_25_percent):
    plt.plot(avg_fitness_top_25_percent)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness of Top 25%')
    plt.title('Evolution of Fitness over Generations')
    plt.show()
    
def save_model(agent, file_name):
    agent_nn = agent.brain.state_dict()
    torch.save(agent_nn, file_name)

def load_model_into_agents(file_name, num_agents=num_agents):
    agent_brain = torch.load(file_name)
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius) for _ in range(num_agents)]
    for agent in agents:
        agent.brain.load_state_dict(agent_brain)
    return agents

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Random Population Movement with Senses")
    parser.add_argument("--mode", choices=["train", "display"], required=True, help="Choose to train or display agents")
    parser.add_argument("--generations", type=int, default=100, help="Choose how many generations to train for")
    parser.add_argument("--num_agents", type=int, default=50, help="Choose the number of agents")
    parser.add_argument("--num_foods", type=int, default=25, help="Choose the number of foods")
    parser.add_argument("--mutation_rate", type=float, default=0.2, help="Rate of mutation")
    parser.add_argument("--steps", type=int, default=500, help="Steps per generation")
    # parser.add_argument("--lambda", type=int, default=0.9, help="Rate of mutation")
    parser.add_argument("--save_path", type=str, default="trained_agents_world.pt", help="Specify the path to save the best model")
    parser.add_argument("--model_path", type=str, default=None, help="Specify model path to load")
    return parser.parse_args()

def main(args):
    if args.mode == "train":
        if args.model_path != None:
            print("Loading model...")
            agents = load_model_into_agents(args.model_path, args.num_agents)
        else:
            agents = None
        trained_agents, avg_data, best_agent = run_genetic_algorithm(args.num_foods, args.generations, args.num_agents, args.mutation_rate, agents, args.steps)
        plot_avg_fitness(avg_data)
        save_model(best_agent, args.save_path)
        main_display(args, trained_agents)

    elif args.mode == "display":
        main_display(args)

def main_display(args, agents=None):
    if args.model_path == None:
        if agents == None:
            agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius) for _ in range(num_agents)]
    else:
        print("Loading model...")
        agents = load_model_into_agents(args.model_path)
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Random Population Movement with Senses')
    clock = pygame.time.Clock()
    foods = [Food(random.randint(0, screen_size), random.randint(0, screen_size)) for _ in range(args.num_foods)]
    running = True
    while running:
        screen.fill((0, 0, 0))
        for agent in agents:
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
            # Check for mouse click event
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the mouse click position
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Define the number of food items and the radius around the clicked position
                num_food_items = 10
                radius = 50

                # Create the new food items around the clicked position
                for _ in range(num_food_items):
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, radius)
                    new_x = mouse_x + distance * math.cos(angle)
                    new_y = mouse_y + distance * math.sin(angle)

                    # Add the new food item to the list of foods
                    new_food = Food(new_x, new_y)
                    foods.append(new_food)
        

    pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    main(args)


