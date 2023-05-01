# Experiments in the Terrarium
Simulating populations of primitive agents scavenging for resources.


```python
! pip install -r requirements.txt
```


```python
! pip install pygame
```

# Experiment #1 - World.py
First task: simulate actual evolution. Each agent has a neural network with a single hidden layer of size 12. Inputs: dy, dx of nearest food; outputs: direction to walk in. Yes, they're just evolving to output their inputs... not so easy for them as it sounds though.

At the end of each generation, the top 25% agents stay into the next (this is referred to as "elitism.") The rest of the agents are mutated versions of the top agents.

We can specify the mutation rate with --mutation_rate. I use a dynamic mutation rate: if the agents aren't performing well, the mutation rate will be close to the specified rate. If they are performing well, the mutation rate will be lower. The rate adjusts at every generation. Thus, it's okay to go for a slightly high mutation rate to speed up evolution. I'd start with 0.2.


```python
! python world.py --help
```

### Random agents
To see how agents start off at generation 0, run this cell. The little green items are food.


```python
! python world.py --mode display
```

### All initialized as best agent from my training (200 generations)
**By the way, you can click on the screen to make food appear.**
These have a continuous action space so they are very smooth compared to the next experiments.


```python
! python world.py --mode display --model_path "best_agent_world_demo.pt" --num_foods 30
```

### Training test
If you want to try training them, execute this cell. It takes about 400 generations for them to start getting decently good. That takes about 20 minutes(?) Change it to a smaller number if you just want to test it.

Note: this cell failed on me when running in this notebook. If it doesn't work, just paste it into a command line. It should work.


```python
! python world.py --mode train --save_path "training_test_demo.pt" --generations 400 --mutation_rate 0.2
```

# Experiment 2: Hive Brain Reinforcement Learning
It's not really a hive brain, but kinda! All agents have copies of the same model. Every 10 steps, the model is updated with the experiences of all of the agents. They all learn collectively.

## Easy task - just eat food
This trains WAY faster than their genetic counterparts. I did this as a proof of concept to see if I could make the collective learning work. They are also very fun to watch.

The demo model was trained for 10 epochs. Performance plateaus after.


```python
! python hivebrain_easy.py --model_path "models/hive_easy_trained_10_ep.pt" --display --num_foods 100
```

### Train it yourself
From my trials it actually gets close to peak performance in about 5 epochs, which only takes a few minutes to train.


```python
! python hivebrain_easy.py --train --epochs 10 --steps 10000 --save_path "hive_easy_test_demo.pt"
```

# Difficult task - Cooperate to bring food to colony
Now, instead of having food appear randomly on the map, there is only one big food location with infinite food. Agents can pass food to each other. They get rewarded for getting food to their colony.

The goal here was to have them learn cooperative behavior: the food passing radius is fairly large, so the optimal policy would be cooperative. Every agent that participated in a given piece of food travelling to the colony gets rewarded.

The real question is whether sharing the same brain here is a good idea.

```python
! python hivebrain_coop_dumb.py --display --model_path "models/hive_e30_s6.pt"
```
