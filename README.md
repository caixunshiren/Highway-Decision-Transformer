# Highway-Decision-Transformer
Decision Transformer for offline single-agent autonomous highway driving

## Code Structure
```
.
├── README.md
├── modules # all model modules contained here
│   ├── __init__.py
├── pipelines # all training, testing, preprocessing, and data gathering pipelines contained here
│   ├── __init__.py
├── expert_scripts # all expert data collection contained here
├── example-notebooks # scartch/example jupyternotebooks
├── experiments # all training, testing, and demo experiment files for various models
```

## Data Collection
Three online RL methods were used to collect expert data: Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and Monte Carlo Tree Search (MCTS). The scripts to collect data can be found in ```/expert_scripts```.

Below is a demonstration of the performance of the various experts. PPO and DQN are highly popular, state-of-the-art online RL methods while MCTS completely searches the game tree at each iteration and thus always finds the maximum reward/best move.

### Expert Online RL Demo
1. Proximal Policy Optimization (PPO)
<p align="center">
<img
     src="figures/PPO.gif"
     width="500"
     >
</p>

2. Deep Q-Network (DQN)
<p align="center">
<img
     src="figures/DQN.gif"
     width="500"
     >
</p>

3. Monte Carlo Tree Search (MCTS)
<p align="center">
<img
     src="figures/MCTS.gif"
     width="500"
     >
</p>

## Models
1. Benchmark: Behaviour Cloning
This is one of two benchmark models used by the original DT paper. By following an imitation-learning approach, we planned to develop an agent to mimic the behaviours of the expert on which it is trained on.

We implemented this using a multi-layer perceptron. The model is defined in ```/modules/behaviour_cloning.py```.

3. Benchmark: Conservative Q-Learning
The is the state-of-the-art offline RL method. It uses a temporal difference learning approach.

The model is yet to be created.

5. Baseline Decision Transformer
Several experiments with various configurations of training datasets and parameters have been conducted in ```/experiments/```.

The DT model is defined in ```/modules/decision_transformer.py``` which is based on GPT-2 defined in ```/modules/trajectory_gpt2.py```.

7. Decision Transformer with Different Encoders


9. LSTM


##
