# multitask-rl

Multitask RL experiments.

## Dependencies

- rlkit, gym, mujoco, multiworld

## Todo

- [x] Train a point mass to go from any starting state to any goal.
  - SAC/PPO (OpenAI baselines, softlearning repo).
  - [x] Make a point mass environment
    - Box, state = current position, action = vector (ex: 2d vector), new state = current position + vector
  - [x] Train a policy to solve point mass env with contextual SAC
- [x] Policy distillation on point mass environment
  - [x] Using optimal straight-line policy
  - [x] Using SAC trained point mass policy of 15 individual tasks
- [ ] Testing with known difficult environments
  - [ ] Replicate Kevin/Deirdre 9 task setup, 3 task case should work with contextual SAC
  - [ ] Policy distillation
  - [ ] Gradient conflicts
  - [ ] Trying to address moving target problem by adding to replay buffer experiences from single-task train
