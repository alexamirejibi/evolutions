import argparse
import bisect
from math import sqrt
import math
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import copy
import functools
import matplotlib.pyplot as plt
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Parameters
screen_size = 800
num_agents = 25
agent_radius = 5
speed = 10
perception_radius = screen_size
num_generations = 1
num_steps = 400
mutation_rate = 0.05
num_food_locations = 50
num_foods = num_food_locations
all_signals = []

# device = torch.device('mps')
device = torch.device('cpu')


# Initialize Pygame

class FoodLocation:
    def __init__(self, x, y, radius=3):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (self.x, self.y), self.radius)

    def reset_random(self):
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class Brain:
    def __init__(self, input_size, output_size, hidden_size=32, batch_size=32, memory_size=10000, gamma=0.99, lr=1e-3, device=device):
        self.device = device
        self.q_network = QNetwork(input_size, output_size, hidden_size).to(device)
        self.target_q_network = QNetwork(input_size, output_size, hidden_size).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.q_network.layers[-1].out_features - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()


    def memorize(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        next_q_values = self.target_q_network(next_states_tensor).max(1, keepdim=True)[0].detach()
        target_q_values = rewards_tensor + (self.gamma * next_q_values * (~dones_tensor))

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))


# Agent class
class Agent:
    def __init__(self, x, y, radius, brain):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.brain = brain
        self.state = None
        self.hit_wall = 0
        self.max_objects = 2
        self.input_len = 1 + (2 * self.max_objects)
        self.total_rewards = 0

    def step(self, signals):
        perceptions = self.sense(signals)
        input_data = [0.0] * self.input_len
        input_data[0] = self.hit_wall
        for i, (dx, dy) in enumerate(perceptions):
            input_data[1 + 2 * i] = dx / screen_size
            input_data[1 + 2 * i + 1] = dy / screen_size
        next_state = input_data

        if self.state is not None:
            reward = 0
            reward += self.collect_food(food_locations)
            reward -= self.hit_wall * 0.1
            self.total_rewards += reward
            self.brain.memorize(self.state, self.action, reward, next_state, False)
        self.state = next_state
        self.action = self.brain.act(self.state)
        dx, dy = self.action_to_movement(self.action)
        x = self.x + speed * dx
        y = self.y + speed * dy
            
        # ------------------------------

        # Keep the agent inside the screen boundaries
        self.hit_wall = False
        if 0 <= x <= screen_size:
            self.x = x
        else:
            self.hit_wall = True

        if 0 <= y <= screen_size:
            self.y = y
        else:
            self.hit_wall = True
            
        # return signal
    

    def action_to_movement(self, action):
        dx, dy = 0, 0
        if action == 0:
            dx, dy = 1, 0
        elif action == 1:
            dx, dy = -1, 0
        elif action == 2:
            dx, dy = 0, 1
        elif action == 3:
            dx, dy = 0, -1
        elif action == 4:
            dx, dy = 1 / math.sqrt(2), -1 / math.sqrt(2)
        elif action == 5:
            dx, dy = -1 / math.sqrt(2), -1 / math.sqrt(2)
        elif action == 6:
            dx, dy = 1 / math.sqrt(2), 1 / math.sqrt(2)
        elif action == 7:
            dx, dy = -1 / math.sqrt(2), 1 / math.sqrt(2)
        return dx, dy
    
    def reset(self):
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)
        self.hit_wall = 0
        self.total_rewards = 0
        

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
    
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def collect_food(self, food_locations):
        for location in food_locations:
            distance = np.sqrt((self.x - location.x) ** 2 + (self.y - location.y) ** 2)
            if distance < self.radius + location.radius:
                location.reset_random()
                # self.carrying_food = True
                return 1
        return 0

        
            
class World:
    def __init__(self, agents, food_locations, brain, selection_rate=0.15, signal_cooldown=10):
        self.agents = agents
        self.food_locations = food_locations
        self.num_agents = len(agents)
        self.brain = brain
        
    def step(self):        
        for i, agent in enumerate(self.agents):
            agent.step(self.food_locations)
    
            
    def update_agents_brains(self):
        for agent in self.agents:
            agent.brain.update()
            agent.brain.update_target_q_network()
                
                
    def draw(self, screen):
            
        for agent in self.agents:
            agent.draw(screen)
        
        for food in self.food_locations:
            food.draw(screen)
                    
    def randomize_reset(self):
        rewards = []
        for agent in agents:
            rewards.append(agent.total_rewards)
            agent.reset()
        for food in self.food_locations:
            food.reset_random()
        return rewards
    
    
def create_agents_and_brain(num_agents, agent_radius):
    brain = Brain(input_size=5, output_size=8, hidden_size=64, batch_size=32, memory_size=10000, gamma=0.99, lr=1e-3, device=device)
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, brain) for _ in range(num_agents)]
    # colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    return agents, brain

def create_food_locations(num_foods):
    food_locations = [FoodLocation(random.randint(0, screen_size), random.randint(0, screen_size)) for _ in range(num_foods)]
    return food_locations

        
def plot_avg_fitness(avg_fitness_top_25_percent):
    plt.plot(avg_fitness_top_25_percent)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness of Top 25%')
    plt.title('Evolution of Fitness over Generations')
    plt.show()

def main(world:World):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Random Population Movement with Senses')
    clock = pygame.time.Clock()

    running = True
    steps = 0
    while running:
        steps += 1
        if steps % 700 == 0:
            world.randomize_reset()
        screen.fill((0, 0, 0))
        world.step()
        world.draw(screen)
        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                new_food = FoodLocation(mouse_x, mouse_y)
                world.food_locations.append(new_food)

    pygame.quit()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Random Population Movement with Senses")
    parser.add_argument("--num_agents", type=int, default=num_agents, help="Number of agents in the simulation")
    parser.add_argument("--agent_radius", type=float, default=agent_radius, help="Radius of the agents")
    parser.add_argument("--num_foods", type=int, default=num_foods, help="Number of foods in each location")
    parser.add_argument("--screen_size", type=int, default=500, help="Size of the screen for the simulation display")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model you want to load")
    parser.add_argument("--save_path", type=str, default="./models/hive_easy_new.pt", help="Path for saving the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--display", action="store_true", help="Display the simulation using pygame")
    parser.add_argument("--epochs", type=int, default=10, help="How many epochs to train for")
    parser.add_argument("--steps", type=int, default=10000, help="Steps per epoch")
    args = parser.parse_args()
    return args


def run_simulation(world, display):
    if display:
        main(world)
    else:
        train(world)

def train(world):
    print(f"Training for {args.epochs} epochs")
    for epoch in range(epochs):
        print("Epoch:", epoch)
        for step in range(ep_steps):
            world.step()
            if step % update_interval == 0:
                world.update_agents_brains()
            if step % 700 == 0:
                rewards = world.randomize_reset()
                average = sum(rewards) / len(rewards)
                print(f"Average rewards at epoch {epoch} step {step}: {average}")
        world.brain.save_model(args.save_path)  # Save the model
        print(f"Saved at epoch {epoch} at filepath {args.save_path}")
    main(world)
    
        

if __name__ == "__main__":
    args = parse_args()

    agents, brain = create_agents_and_brain(args.num_agents, args.agent_radius)
    food_locations = create_food_locations(args.num_foods)
    world = World(agents, food_locations, brain=brain) # top 15% multiply
    
    if args.model_path != None:
        brain.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    
    epochs = args.epochs
    ep_steps = args.steps
    update_interval = 10

    if args.train:
        # world.brain.load_model(args.model_path)
        world.randomize_reset()
        for a in world.agents:
            a.brain = world.brain # just making sure...
        train(world)

    if args.display:
        # world.brain.load_model(args.model_path)
        world.randomize_reset()
        for a in world.agents:
            a.brain = world.brain

        run_simulation(world, args.display)