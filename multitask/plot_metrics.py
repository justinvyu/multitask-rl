
from multitask.run_policy import run_policy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt

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

if __name__ == "__main__":
    # plot_task_completion()
    plot_learning_curve()