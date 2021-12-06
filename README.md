Reproduce the experiment for GridWorld. Two policies are simply Q-tables. Experiments show that only action replay is needed. HER transitions and subgoal testing transitions seem to hurt the performance.

### Training
```python3 run_hac.py --n_layers 2 --env hierq-grid-world-v0 --retrain --timesteps 2000000 --seed 0 --group 2-level```

### Testing
```python3 run_hac.py --n_layers 2 --env hierq-grid-world-v0 --test --show --timesteps 2000000 --seed 0 --group 2-level```
