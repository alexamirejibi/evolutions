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
signal_radius = 200
num_generations = 1
num_steps = 400
mutation_rate = 0.05
num_foods = 200
num_food_locations = 1
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
    def __init__(self, x, y, num_foods=40):
        self.x = x
        self.y = y
        self.num_foods = num_foods
        self.foods = num_foods
        self.radius = self.get_radius()

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius)
        
    def signal_strength(self): # doesn't do anyting currently
        return 2

    def update_food(self):
        self.foods -= 1
        self.radius = self.get_radius()
        # print(self.radius)

    def has_food(self):
        return self.foods > 0
    
    def reset_random(self):
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)
        self.foods = self.num_foods
        self.radius = self.get_radius()
        
    def get_radius(self):
        return self.foods * 0.3



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
    def __init__(self, x, y, radius, colony, brain, signal_cooldown=500, id=None):
        self.id = id
        self.x = x
        self.y = y
        self.radius = radius
        self.colony = colony
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.brain = brain
        self.carrying_food = False
        self.signal_cooldown = signal_cooldown
        self.signal_counter = self.signal_cooldown
        self.state = None
        self.at_food_loc = 0
        self.perceived_signal = None
        self.hit_wall = 0

    def step(self, signals):
        perceptions = self.sense(signals)
        
        input_data = [0.0] * 8
        if perceptions:
            dx1, dy1, signal_strength = perceptions
            self.perceived_signal = self.x + dx1, self.y + dy1, signal_strength
            input_data[0] = dx1 / screen_size
            input_data[1] = dy1 / screen_size
            input_data[2] = min(1, signal_strength)
        input_data[3] = int(self.carrying_food)
        input_data[4] = self.at_food_loc
        input_data[5] = (self.x - self.colony.x) / screen_size
        input_data[6] = (self.y - self.colony.y) / screen_size 
        input_data[7] = self.hit_wall
        next_state = input_data

        if self.state is not None:
            reward = 0
            if self.carrying_food:
                reward += self.deposit_food()
            else:
                reward += self.collect_food(food_locations)
            reward -= self.hit_wall * 0.2
            # print(reward)
            self.brain.memorize(self.state, self.action, reward, next_state, False)

        self.state = next_state
        self.action = self.brain.act(self.state)

        dx, dy, signal = self.action_to_movement(self.action)
        x = self.x + speed * dx
        y = self.y + speed * dy

        # Keep the agent inside the screen boundaries

        self.signal_counter -= 1
        if self.at_food_loc:
            self.signal_counter = 0
            signal = 1
        
        self.hit_wall = False
        if 0 <= x <= screen_size:
            self.x = x
        else:
            self.hit_wall = True

        if 0 <= y <= screen_size:
            self.y = y
        else:
            self.hit_wall = True
        
            
        return signal
    

    def action_to_movement(self, action):
        dx, dy, signal = 0, 0, 0
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
        elif action == 8:
            signal = 1
        elif action == 9:
            signal = 0.5
        return dx, dy, signal    
    
    def reset(self):
        self.signal_counter = self.signal_cooldown
        self.carrying_food = False
        self.x = random.randint(0, screen_size)
        self.y = random.randint(0, screen_size)
        self.hit_wall = 0
        print(self.colony.x, self.colony.y)
        

    def sense(self, signals):
        perception_radius = signal_radius
        combined_signal_strength = 0
        combined_dx, combined_dy = 0, 0
        
        for signal in signals:
            if signal.agent_id == self.id:
                continue
            distance = math.sqrt((self.x - signal.x)**2 + (self.y - signal.y)**2)
            # if distance <= perception_radius:
            # if True: # for debugging
            weight = gaussian(distance, 0.99)
            # print(weight)
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
        
        if self.perceived_signal != None:
            dy, dx, st = self.perceived_signal
            line_color = (255, 255, 255)
            # print(dy, dx)
            self.perceived_signal = None
            pygame.draw.line(screen, line_color, (self.x, self.y), (dx, dy))


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

def gaussian(distance, sigma):
    return math.exp(-distance**2 / (2 * sigma**2))

class Signal:
    def __init__(self, frequency, x, y, agent_id=None):
        self.frequency = max(0, min(1, frequency))
        self.strength = frequency * 50
        self.radius = self.strength
        self.x = x
        self.y = y
        self.agent_id = agent_id
    
    def update(self): # need to keep food signals always (???)
        self.strength *= 0.95
        return self.strength
    
    def draw(self, screen):
        # color = (0, 0, 255, min(int(self.strength), 128))
        color = (0, 0, 255, 128)
        pygame.draw.circle(screen, color, (self.x, self.y), int(self.strength))

        
            
class World:
    def __init__(self, agents, food_locations, colony, brain, selection_rate=0.15, signal_cooldown=10):
        self.agents = agents
        self.food_locations = food_locations
        self.colony = colony
        self.signals = []
        self.steps = 0
        self.fitness_scores = [0] * len(agents)
        self.rewards = 0
        self.num_agents = len(agents)
        self.num_parents = int(self.num_agents * selection_rate)
        self.signal_cooldown = signal_cooldown
        self.brain = brain
        for a in self.agents:
            a.colony = self.colony
            
        food_signal_strength = self.food_locations[0].signal_strength()
        for f in self.food_locations:
            self.signals.append(Signal(food_signal_strength, f.x, f.y, agent_id=-1))
    
    def step(self):
        self.update_signals()
        food_signal_strength = self.food_locations[0].signal_strength()
        for f in self.food_locations:
            self.signals.append(Signal(food_signal_strength, f.x, f.y, agent_id=-1))
        
        for i, agent in enumerate(self.agents):
            
            agent.at_food_loc = self.at_food_location(agent)
            
            if not agent.carrying_food:
                self.fitness_scores[i] += agent.collect_food(self.food_locations)
            else:
                self.fitness_scores[i] += agent.deposit_food()
            
            signal = agent.step(self.signals)
            while len(self.signals) >= 200:
                self.signals.pop()
            
            if agent.signal_counter > 0 and agent.at_food_loc == 0:
                continue
            
            agent.signal_counter = self.signal_cooldown
            
            signal = Signal(signal, agent.x, agent.y, agent.id)
            
            if self.signals:
                bisect.insort_left(self.signals, signal, key=lambda x: -x.strength)
            else:
                # print("appending:", signal)
                self.signals.append(signal)
                
    def at_food_location(self, agent):
        for location in food_locations:
            distance = np.sqrt((agent.x - location.x) ** 2 + (agent.y - location.y) ** 2)
            if distance < agent.radius + location.radius and location.has_food():
                return 1
        return 0
    
            
    def update_agents_brains(self): #TODO
        for agent in self.agents:
            agent.brain.update()
            agent.brain.update_target_q_network()
                
                
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


    def update_signals(self):
        for i, signal in enumerate(self.signals):
            if signal.agent_id==-1:
                # print(signal.x, signal.y)
                continue
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
    
    
def create_agents_and_colony(num_agents, agent_radius):
    brain = Brain(input_size=8, output_size=10, hidden_size=64, batch_size=32, memory_size=10000, gamma=0.99, lr=1e-3, device=device)
    agents = [Agent(random.randint(0, screen_size), random.randint(0, screen_size), agent_radius, None, brain, id=i) for i in range(num_agents)]
    colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    return agents, colony, brain

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

def main(world:World):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Random Population Movement with Senses')
    clock = pygame.time.Clock()

    # new_colony = Colony(random.randint(0, screen_size), random.randint(0, screen_size))
    # new_food_locations = create_food_locations(num_food_locations, num_foods)
    # for a in agents:
    #     a.colony = new_colony
        
    # world = World(agents, new_food_locations, new_colony, selection_rate=0.15) # top 15% multiply

    running = True
    steps = 0
    while running:
        steps += 1
        if steps % 100 == 0:
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
        for step in range(save_interval):
            world.step()
            if step % update_interval == 0:
                world.update_agents_brains()
            if step % 1000 == 0:
                world.randomize_reset()
        world.brain.save_model(f'./models/model.pt')  # Save the model
        print("Saved at epoch: ", epoch)
        

if __name__ == "__main__":
    args = parse_args()

    agents, colony, brain = create_agents_and_colony(args.num_agents, args.agent_radius)
    food_locations = create_food_locations(args.num_food_locations, args.num_foods)
    world = World(agents, food_locations, colony, brain=brain) # top 15% multiply

    epochs = 1000000
    steps = 100000
    save_interval = 50000
    avg_data = []
    update_interval = 10

    if args.train:
        world.brain.load_model(args.model_path)
        world.randomize_reset()
        for a in world.agents:
            a.brain = world.brain

        train(world)

    if args.display:
        world.brain.load_model(args.model_path)
        world.randomize_reset()
        for a in world.agents:
            a.brain = world.brain

        run_simulation(world, args.display)