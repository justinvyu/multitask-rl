
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from multitask.run_policy import run_distilled_policy
import joblib

import pickle
from rlkit.torch.networks import TanhMlpPolicy
from gym_pointmass.envs import PointMassEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
import matplotlib.pyplot as plt
import seaborn as sns

def get_batch(env, batch_size, use_optimal_policy, policies):
    """
    Returns an array with sampled "inputs", which are sampled points in the state space,
    and ground truth labels for the optimal action at that point.
    :param batch_size: Number of data points to generate.
    """
    batch_obs, batch_acts = [], []
    # We can use env.reset() to generate random states/goals.
    for _ in range(batch_size):
        sample = np.array(env.reset())
        point = np.array(sample[:env.dimension])
        goal_idx = np.argmax(sample[env.dimension:])
        goal = np.array(env.goals[goal_idx])
        if use_optimal_policy:
            act = goal - point
            if (np.linalg.norm(act) > 1):
                act /= np.linalg.norm(act) # Limit to unit vector.
        else:
            expert = policies[goal_idx]
            single_task_sample = np.concatenate([point, [1]])
            act, _ = expert.get_action(single_task_sample)
            act = np.array(act)
        batch_obs.append(sample)
        batch_acts.append(act)

    return np.array(batch_obs), np.array(batch_acts)

def get_expert_policy(path, task_index):
    """
    Returns the trained policy stored in the directory of `task_index`.
    """
    data = joblib.load(path + "/" + str(task_index) + "/params.pkl")
    return data["evaluation/policy"]

def train_supervised(num_tasks,
                     use_optimal_policy=True,
                     policies=None,
                     epochs_per_task=500,
                     batch_size=100, lr=1e-3):
    base_env = PointMassEnv(n=num_tasks)
    env = FlatGoalEnv(base_env, append_goal_to_obs=True)
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size

    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=act_dim,
        hidden_sizes=[64, 64, 64]
    )

    criterion = nn.MSELoss()
    optim = Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs_per_task * num_tasks):
        if not use_optimal_policy:
            assert len(policies) == num_tasks, "Number of expert policies needs " \
                                               "to be equal to the number of tasks"
        obs, act_labels = get_batch(env, batch_size, use_optimal_policy, policies)

        obs_var, act_labels_var = Variable(torch.from_numpy(obs)).float(), \
                                  Variable(torch.from_numpy(act_labels)).float()
        acts = policy(obs_var)

        optim.zero_grad()
        loss = criterion(acts, act_labels_var)
        loss.backward()
        optim.step()
        if epoch % 50 == 0:
            print("epoch: {0} \t loss: {1}".format(epoch, loss.data.item()))

    print("FINAL loss: {1}".format(epoch, loss.data.item()))

    out = dict(
        policy=policy,
        env=env
    )
    with open("./logs/policy-distillation/model-{0}-from_trained_experts.pkl".format(num_tasks), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return policy, env

if __name__ == "__main__":
    train = False
    if train:
        num_tasks = 15
        policies_path = "./logs/policy-distillation/expert-singletask-policies-{}".format(num_tasks)
        policies = [get_expert_policy(policies_path, i) for i in range(num_tasks)]
        train_supervised(num_tasks,
                         use_optimal_policy=False,
                         policies=policies)
    else:
        # TODO: Make a generic plot trajectory helper function.
        num_rollouts = 200
        results = run_distilled_policy("./logs/policy-distillation/model-15-from_trained_experts.pkl", num_rollouts)
        paths = results["paths"]
        env = results["env"]
        sns.set()
        plt.figure(figsize=(8, 8))

        goal_to_show = 14
        for path in paths:
            obs = path["observations"]
            acts = path["actions"]
            goal_idx = np.argmax(obs[0, 2:])
            if goal_idx != goal_to_show:
                continue

            plot_row, plot_col = goal_idx // 5, goal_idx % 5

            start_x = obs[0, 0]
            start_y = obs[0, 1]

            plt.scatter(start_x, start_y, color="g")
            plt.scatter(obs[1:, 0], obs[1:, 1], color="b", s=10)

            acts_x = acts[:, 0]
            acts_y = acts[:, 1]

            plt.quiver(obs[:, 0], obs[:, 1], acts_x, acts_y,
                       angles='xy', scale_units='xy', scale=1, width=.002, headwidth=2, alpha=.9)

            goal = env.goals[goal_idx]
            goal_x, goal_y = goal[0], goal[1]

            plt.scatter(goal[0], goal[1], color="r")  # Goal

        plt.xlim(-env.bound, env.bound)
        plt.ylim(-env.bound, env.bound)

        # Legend
        plt.legend(["Initial State (s_0)", "States (s_t)", "Actions (a_t)", "Goal Point"])

        # Add unit circle
        circle = plt.Circle((0, 0), env.goal_distance, color='black', alpha=.5, fill=False)
        plt.gcf().gca().add_artist(circle)

        circle2 = plt.Circle(env.goals[goal_to_show], 0.25, color='red', alpha=.75, fill=False)
        plt.gcf().gca().add_artist(circle2)

        # for goal in env.goals:
        #     circle = plt.Circle(goal, 0.25, color='red', alpha=.75, fill=False)
        #     plt.gcf().gca().add_artist(circle)

        final_states = np.array(results["final_states"])
        goals = np.array(results["goal_states"])
        diff = final_states - goals
        # completion = np.sum((np.linalg.norm(diff, axis=1) < 0.25).astype(int)) / num_rollouts
        # plt.title("% task completion: {0:.0%}, {1} rollouts".format(completion, num_rollouts))

        plt.show()
