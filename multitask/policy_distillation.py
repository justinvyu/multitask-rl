
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

def get_batch(env, batch_size, policies):
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
        if not policies:
            # If no policies are being loaded, use the optimal straight-line policy.
            act = goal - point
            if (np.linalg.norm(act) > 1):
                act /= np.linalg.norm(act) # Limit to unit vector.
        else:
            # Load expert model.
            expert = policies[goal_idx]
            # Modify observation to only contain a "single-task" one-hot encoding
            # appended to the end, since the expert only has one goal. Ex: [0.3, 0.5, 1]
            single_task_sample = np.concatenate([point, [1]])
            # Get the optimal action from this state.
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

def train_distilled_policy(num_tasks,
                           policies=None,
                           epochs_per_task=500,
                           batch_size=100, lr=1e-3):
    """
    Trains a distilled policy (using an optimal expert or a trained expert).
    Saves the policy in a .pkl file along with the env and the loss history.

    :param num_tasks: Number of tasks/policies to distill.
    :param policies: A list of length `num_tasks` containing all the individual experts.
    :param epochs_per_task: Number of training epochs per task.
    :param batch_size: Batch sample size per update step.
    :param lr: Learning rate of the optimizer.
    :return: The trained policy and the environment.
    """
    base_env = PointMassEnv(n=num_tasks)
    env = FlatGoalEnv(base_env, append_goal_to_obs=True)
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size

    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=act_dim,
        hidden_sizes=[64, 64]
        # hidden_sizes=[64, 64, 64]
    )

    loss_history = []
    criterion = nn.MSELoss()
    optim = Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs_per_task * num_tasks):
        if policies:
            assert len(policies) == num_tasks, "Number of expert policies needs " \
                                               "to be equal to the number of tasks"
        obs, act_labels = get_batch(env, batch_size, policies)

        obs_var, act_labels_var = Variable(torch.from_numpy(obs)).float(), \
                                  Variable(torch.from_numpy(act_labels)).float()
        acts = policy(obs_var)

        optim.zero_grad()
        loss = criterion(acts, act_labels_var)
        loss.backward()
        optim.step()

        loss_val = loss.data.item()
        loss_history.append(loss_val)
        if epoch % 50 == 0:
            print("epoch: {0} \t loss: {1}".format(epoch, loss_val))

    print("FINAL loss: {1}".format(epoch, loss.data.item()))
    out = dict(
        policy=policy,
        env=env,
        loss_history=loss_history
    )
    appended_path = "-from_expert_policies" if policies else ""
    path = "./logs/policy-distillation/model-{0}{1}.pkl".format(num_tasks, appended_path)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return policy, env

if __name__ == "__main__":
    train = False
    if train:
        num_tasks = 5
        policies_path = "./logs/policy-distillation/expert-singletask-policies-{}".format(num_tasks)
        policies = [get_expert_policy(policies_path, i) for i in range(num_tasks)]
        train_distilled_policy(num_tasks, policies=policies)
    else:
        # TODO: Make a generic plot trajectory helper function.
        num_rollouts = 200
        results = run_distilled_policy("./logs/policy-distillation/model-25.pkl", num_rollouts)
        paths = results["paths"]
        env = results["env"]
        sns.set()
        plt.figure(figsize=(8, 8))

        for path in paths:
            obs = path["observations"]
            acts = path["actions"]
            goal_idx = np.argmax(obs[0, 2:])

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

        for goal in env.goals:
            circle = plt.Circle(goal, 0.25, color='red', alpha=.75, fill=False)
            plt.gcf().gca().add_artist(circle)

        final_states = np.array(results["final_states"])
        goals = np.array(results["goal_states"])
        diff = final_states - goals
        completion = np.sum((np.linalg.norm(diff, axis=1) < 0.25).astype(int)) / num_rollouts
        plt.title("% task completion: {0:.0%}, {1} rollouts".format(completion, num_rollouts))

        plt.show()
