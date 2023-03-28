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

## Expert Online RL Demo
1. Proximal Policy Optimization (PPO)
![PPO](figures/PPO.gif)

2. Deep Q-Network (DQN)

3. Monte Carlo Tree Search (MCTS)
