
from multitask.run_policy import run_policy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt
import joblib
import torch
import itertools

def plot_task_completion(num_rollouts_per_task=50, completion_threshold=0.1):
    sns.set()
    common_path = "./logs/sac-multitask-"
    num_tasks = [1, 2, 3, 4, 5, 6, 7, 10, 15]
    task_completion = []

    for n in num_tasks:
        path = common_path + str(n) + "/params.pkl"
        num_rollouts = n * num_rollouts_per_task
        results = run_policy(path, num_rollouts)

        final_states = results["final_states"]
        goals = results["goal_states"]
        diff = final_states - goals

        percent = np.sum((np.linalg.norm(diff, axis=1) < completion_threshold).astype(int)) / num_rollouts
        task_completion.append(percent)

    print(task_completion)
    plt.title("Task completion vs. number of tasks")
    plt.xlabel("Number of tasks (n)")
    plt.ylabel("% task completion ({0} samples per task)".format(num_rollouts_per_task))
    plt.ylim(0, 1.1)
    plt.scatter(num_tasks, task_completion)
    plt.plot(num_tasks, task_completion)
    plt.show()

def plot_learning_curve():
    sns.set()
    common_path = "./logs/sac-multitask-"
    num_tasks = [1, 2, 3, 4, 5, 6, 7, 10, 15]

    for n in num_tasks:
        path = common_path + str(n) + "/progress.csv"
        print(path)
        data = genfromtxt(path, dtype=None, delimiter=",", names=True)
        print(data.dtype.names)
        avg_rets = data["evaluationAverage_Returns"]
        min_rets = data["evaluationRewards_Min"]
        max_rets = data["evaluationRewards_Max"]
        idxs = np.arange(0, n * 50, n)

        plt.plot(np.array(range(n * 50)) / n, avg_rets[:n*50])
        # plt.errorbar(data["Epoch"][:n*50], avg_rets[:n * 50], yerr=np.array([min_rets[:n*50], max_rets[:50*n]]), alpha=0.9)

    plt.legend(num_tasks)
    plt.xticks(np.arange(0, 51, 10))
    plt.ylabel("Average Return")
    plt.xlabel("Epoch (200 env steps/epoch)")
    plt.title("Learning curves with different number of tasks")
    plt.show()

def plot_values():
    path = "./logs/sac-multitask-3/params.pkl"
    data = joblib.load(path)
    print(data.keys())
    target_qf1 = data["trainer/target_qf1"]
    policy = data["evaluation/policy"]
    env = data["evaluation/env"]
    bound = env.bound
    print(target_qf1)

    x_vals = np.arange(-bound, bound + 0.5, 0.25)
    y_vals = np.arange(-bound, bound + 0.5, 0.25)
    vals = np.zeros((len(x_vals), len(y_vals)))

    plt.figure(figsize=(8, 8))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            obs = np.array([[x, y, 0, 0, 1]])
            act, _ = policy.get_action(obs)
            act = np.array(act)
            value = target_qf1(torch.Tensor(obs), torch.Tensor(act))
            value = int(value.detach().numpy().flatten()[0])
            vals[i][j] = value
            # plt.annotate("{}".format(value), (x, y), fontsize=10)

    xy = np.array(list(itertools.product(x_vals, y_vals)))
    print(xy)

    plt.scatter(xy[:, 0], xy[:, 1], c=vals.flatten(), s=30, cmap="gray")

    # for x in x_vals:
    #     for y in y_vals:
    #         obs = np.array([[x, y, 1.]])
    #         act, _ = policy.get_action(obs)
    #         act = np.array(act)
    #         value = target_qf1(torch.Tensor(obs), torch.Tensor(act))
    #         value = int(value.detach().numpy().flatten()[0])
    #         # print(np.abs(max_val - value))
    #         # print(np.abs(max_val) - np.abs(min_val))
    #         normalized = np.abs(max_val - value) / np.abs(max_val - min_val)
    #         plt.scatter(x, y, c=np.random.random(), s=10, cmap='rainbow')
    #         plt.annotate("{}".format(value), (x, y))

    # plt.scatter(5, 0, c="green")
    # plt.scatter(-5, 0, c="green")

    plt.xlim(-6.25, 6.25)
    plt.ylim(-6.25, 6.25)
    plt.show()

if __name__ == "__main__":
    plot_task_completion(completion_threshold=0.5)
    # plot_learning_curve()
    # plot_values()