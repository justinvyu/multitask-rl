
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import joblib
import pickle

from rlkit.torch.networks import TanhMlpPolicy
from gym_pointmass.envs import PointMassEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
from multitask.run_policy import run_distilled_policy
from multitask.visualize_rollouts import visualize_rollouts

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
        num_rollouts = 200
        results = run_distilled_policy("./logs/policy-distillation/model-25.pkl", num_rollouts)
        visualize_rollouts(results)