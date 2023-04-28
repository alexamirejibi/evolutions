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


# Parameters
screen_size = 500
num_agents = 25
agent_radius = 5
speed = 10
perception_radius = 400
num_generations = 1
num_steps = 400
mutation_rate = 0.05
num_food_locations = 50
num_foods = num_food_locations
all_signals = []

# device = torch.device('mps')
device = torch.device('cpu')


# Initialize Pygame

class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
        
class Projectile:
    def __init__(self, x, y, dx, dy, speed, team, radius=2):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.speed = speed
        self.team = team
        self.radius = radius

    def move(self):
        self.x += self.speed * self.dx
        self.y += self.speed * self.dy

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.radius)


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
    def __init__(self, x, y, radius, brain, team):
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
        
        self.team = team
        self.projectiles = []
        
    def shoot(self, dx, dy):
        self.projectiles.append(Projectile(self.x, self.y, dx, dy, 5, self.team))


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
            reward += self.collect_food(signals)
            reward -= self.hit_wall * 0.1
            self.total_rewards += reward
            self.brain.memorize(self.state, self.action, reward, next_state, False)
        self.state = next_state
        self.action = self.brain.act(self.state)
        
        if random.random() < 0.01:
            dx, dy = self.action_to_movement(random.randint(0, 7))
            self.shoot(dx, dy)
            
        new_projectiles = []
        for proj in self.projectiles:
            proj.move()
            if not any([obstacle.rect.collidepoint(proj.x, proj.y) for obstacle in obstacles]):
                new_projectiles.append(proj)
        self.projectiles = new_projectiles
            
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
    def hit_by_projectile(self, projectiles):
        for projectile in projectiles:
            if self.team != projectile.team:
                distance = np.sqrt((self.x - projectile.x) ** 2 + (self.y - projectile.y) ** 2)
                if distance < self.radius + projectile.radius:
                    return 1
        return 0
    
    def draw_projectiles(self, screen):
        for proj in self.projectiles:
            proj.draw(screen)

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
        

    def sense(self, signals):
        perceptions = np.empty((0, 2))
        print(perceptions)
        food_coords = np.array([[food.x, food.y] for food in signals])
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
    def __init__(self, agents, obstacles, brains,):
        self.agents = agents
        self.obstacles = obstacles
        self.num_agents = len(agents)
        self.brain = brains
        
    def step(self):        
        all_projectiles = []
        for agent in self.agents:
            all_projectiles += agent.projectiles

        for agent in self.agents:
            agent.step(all_projectiles)
            if agent.hit_by_projectile(all_projectiles):
                agent.total_rewards += 1

        
    
            
    def update_agents_brains(self):
        for agent in self.agents:
            agent.brain.update()
            agent.brain.update_target_q_network()
                
                
    def draw(self, screen):
            
        for agent in self.agents:
            agent.draw(screen)
        
        for obstacle in self.obstacles:
            obstacle.draw(screen)

        for agent in self.agents:
            agent.draw_projectiles(screen)
                    
    def randomize_reset(self):
        rewards = []
        for agent in agents:
            rewards.append(agent.total_rewards)
            agent.reset()
        return rewards

def create_obstacles(num_obstacles):
    obstacles = [Obstacle(random.randint(0, screen_size - 50), random.randint(0, screen_size - 50), random.randint(10, 50), random.randint(10, 50)) for _ in range(num_obstacles)]
    return obstacles
    
def create_agents_and_brain(num_agents, agent_radius):
    brain = Brain(input_size=5, output_size=8, hidden_size=64, batch_size=32, memory_size=10000, gamma=0.99, lr=1e-3, device=device)
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, brain, None) for _ in range(num_agents)]
    # colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    return agents, brain

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

def main(world:World):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Random Population Movement with Senses')
    clock = pygame.time.Clock()

    running = True
    steps = 0
    while running:
        steps += 1
        if steps % 1000 == 0:
            world.randomize_reset()
        screen.fill((0, 0, 0))
        world.step()
        world.draw(screen)
        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Random Population Movement with Senses")
    parser.add_argument("--num_agents", type=int, default=num_agents, help="Number of agents in the simulation")
    parser.add_argument("--agent_radius", type=float, default=agent_radius, help="Radius of the agents")
    parser.add_argument("--num_food_locations", type=int, default=num_food_locations, help="Number of food locations")
    parser.add_argument("--num_foods", type=int, default=num_foods, help="Number of foods in each location")
    parser.add_argument("--screen_size", type=int, default=500, help="Size of the screen for the simulation display")
    parser.add_argument("--model_path", type=str, default="./models/model.pt", help="Path to the saved model")
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
    print("Training...")
    for epoch in range(epochs):
        print("Epoch:", epoch)
        for step in range(ep_steps):
            world.step()
            if step % update_interval == 0:
                world.update_agents_brains()
            if step % 1000 == 0:
                rewards = world.randomize_reset()
                average = sum(rewards) / len(rewards)
                print(f"Average rewards at epoch {epoch} step {step}: {average}")
        # world.brain.save_model(f'./models/model.pt')  # Save the model
        print("Saved at epoch: ", epoch)
    main(world)
    
        

if __name__ == "__main__":
    args = parse_args()
    
    agents_team1, brain_team1 = create_agents_and_brain(args.num_agents // 2, args.agent_radius)
    agents_team2, brain_team2 = create_agents_and_brain(args.num_agents // 2, args.agent_radius)

    
    for agent in agents_team1:
        agent.team = 1
    for agent in agents_team2:
        agent.team = 2
    agents = agents_team1 + agents_team2
    obstacles = create_obstacles(10)
    
    world = World(agents, obstacles, brains=[brain_team1, brain_team2]) # top 15% multiply

    epochs = args.epochs
    ep_steps = args.steps
    save_interval = 500000000000
    avg_data = []
    update_interval = 10

    if args.train:
        # world.brain.load_model(args.model_path)
        world.randomize_reset()
        for a in world.agents:
            a.brain = world.brain
        train(world)

    if args.display:
        # world.brain.load_model(args.model_path)
        world.randomize_reset()
        for a in world.agents:
            a.brain = world.brain

        run_simulation(world, args.display)