
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_separated_by_task(file):
    file = "./logs/sac-pointmass-multitask-5/sac-pointmass-multitask-5_2019_04_20_18_57_19_0000--s-0/params.pkl"

    data = joblib.load(file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']

    # plt.figure(figsize=(8, 8))
    num_goals = len(env.goals)
    has_circle = np.zeros(num_goals).astype(bool)

    fig, ax = plt.subplots(nrows=5, ncols=num_goals // 5)
    fig.set_size_inches(8, 8)
    print("Number of goals:", num_goals)
    for i in range(200):
        path = rollout(
            env,
            policy,
            max_path_length=100,
            animated=False,
        )

        # print(path)
        obs = path["observations"]
        acts = path["actions"]
        goal_idx = np.argmax(obs[0, 2:])
        plot_row, plot_col = goal_idx // 5, goal_idx % 5
        goal_plot = ax[plot_row, plot_col]

        # Turn off
        goal_plot.set_yticklabels([])
        goal_plot.set_xticklabels([])

        start_x = obs[0, 0]
        start_y = obs[0, 1]

        goal_plot.scatter(start_x, start_y, color="green")
        goal_plot.scatter(obs[1:, 0], obs[1:, 1], color="b")
        goal_plot.quiver(obs[:, 0], obs[:, 1], acts[:, 0], acts[:, 1],
                         angles='xy', scale_units='xy', scale=1, width=.005, headwidth=3, alpha=.9)
        # plt.annotate("start=({0}, {1})".format(start_x.round(4), start_y.round(4)), (start_x, start_y), xytext=(start_x-.5, start_y+.2))

        final_x, final_y = obs[len(obs) - 1, 0], obs[len(obs) - 1, 1]
        # plt.annotate("end=({0}, {1})".format(final_x.round(4), final_y.round(4)), (final_x, final_y), xytext=(final_x-.5, final_y-.2))

        goal = env.goals[goal_idx]
        goal_x, goal_y = goal[0], goal[1]
        # plt.annotate("goal=({0}, {1})".format(goal_x.round(4), goal_y.round(4)), (goal_x, goal_y), xytext=(goal_x-.5, goal_y+.1))
        goal_plot.scatter(goal[0], goal[1], color="r")  # Goal
        goal_plot.set_xlim(-1.5, 1.5)
        goal_plot.set_ylim(-1.5, 1.5)

        if not has_circle[goal_idx]:
            circle = plt.Circle((0, 0), 1, color='black', alpha=.5, fill=False)
            goal_plot.add_artist(circle)
            has_circle[goal_idx] = True

if __name__ == "__main__":
    sns.set()

    # 50 epochs new learning params -> 1 task
    file = "./logs/sac-multitask-1/params.pkl"
    # 100
    # file = "./logs/past-experiments/sac-pointmass-multitask-1/sac-multitask-1/params.pkl"
    # 200 epochs -> 2 tasks
    # file = "./logs/sac-pointmass-multitask-2/sac-multitask-2/params.pkl"
    # 300 epochs -> 3 tasks
    # file = "./logs/sac-pointmass-multitask-3/sac-multitask-3/params.pkl"
    # 200 epochs -> 4 tasks
    # file = "./logs/sac-pointmass-multitask-4/sac-multitask-4/params.pkl"
    # 250 epochs -> 5 tasks
    # file = "./logs/sac-pointmass-multitask-5/sac-multitask-5/params.pkl"
    # 300 epochs -> 6 tasks
    # file = "./logs/sac-pointmass-multitask-6/sac-multitask-6/params.pkl"
    # 350 epochs -> 7 tasks
    # file = "./logs/sac-pointmass-multitask-7/sac-multitask-7/params.pkl"
    # 500 epochs -> 10 tasks
    # file = "./logs/sac-pointmass-multitask-10/sac-multitask-10/params.pkl"
    file = "./logs/sac-multitask-15/params.pkl"

    # Goal point at origin
    # file = "./logs/sac-pointmass-multitask-1/sac-pointmass-multitask-1_2019_04_22_20_10_57_0000--s-0/params.pkl"

    # Semi sparse experiment
    # radius of 0.25 * goal_distance
    # file = "./logs/sac-pointmass-multitask-1/sac-pointmass-multitask-1_2019_04_22_20_48_30_0000--s-0/params.pkl"
    # radius of 0.5 * goal_distance
    # file = "./logs/sac-pointmass-multitask-1/sac-pointmass-multitask-1_2019_04_22_20_52_36_0000--s-0/params.pkl"

    data = joblib.load(file)

    # Load in deterministic evaluation policy
    policy = data['evaluation/policy']
    env = data['evaluation/env']

    plt.figure(figsize=(8, 8))
    num_goals = len(env.goals)

    final_states = []
    goals = []

    print("Number of goals:", num_goals)
    num_plotted = 0
    # for i in range(10):
    while num_plotted < 10:
        path = rollout(
            env,
            policy,
            max_path_length=100,
            animated=False,
        )

        # print(path)
        obs = path["observations"]
        acts = path["actions"]
        goal_idx = np.argmax(obs[0, 2:])
        if goal_idx != 0:
            continue
        num_plotted += 1

        plot_row, plot_col = goal_idx // 5, goal_idx % 5

        start_x = obs[0, 0]
        start_y = obs[0, 1]

        plt.scatter(start_x, start_y, color="green")
        plt.scatter(obs[1:, 0], obs[1:, 1], color="b", s=10)

        acts_x = acts[:, 0]
        acts_y = acts[:, 1]

        # norms = np.linalg.norm(acts, axis=1)
        # acts_x = acts[:, 0] / norms * 0.1
        # acts_y = acts[:, 1] / norms * 0.1

        plt.quiver(obs[:, 0], obs[:, 1], acts_x, acts_y,
                   angles='xy', scale_units='xy', scale=1, width=.002, headwidth=2, alpha=.9)
        # plt.quiver(obs[:, 0], obs[:, 1], acts[:, 0], acts[:, 1],
        #            angles='xy', scale_units='xy', scale=1, width=.005, headwidth=3, alpha=.9)
        # plt.annotate("start=({0}, {1}), {2}".format(start_x.round(4), start_y.round(4), i), (start_x, start_y), xytext=(start_x-.5, start_y+.2))

        final_x, final_y = obs[len(obs) - 1, 0], obs[len(obs) - 1, 1]
        final_states.append(np.array([final_x, final_y]))

        # plt.scatter(final_x, final_y)
        # plt.annotate("end=({0}, {1})".format(final_x.round(4), final_y.round(4)), (final_x, final_y),
        #              xytext=(final_x - .5, final_y - 0.5))

        goal = env.goals[goal_idx]
        goal_x, goal_y = goal[0], goal[1]
        goals.append(np.array([goal_x, goal_y]))

        # plt.annotate("goal=({0}, {1})".format(goal_x.round(4), goal_y.round(4)), (goal_x, goal_y), xytext=(goal_x-.5, goal_y+.1))
        plt.scatter(goal[0], goal[1], color="r") # Goal

        plt.xlim(-env.bound, env.bound)
        plt.ylim(-env.bound, env.bound)

    # Legend
    plt.legend(["Initial State (s_0)", "States (s_t)", "Actions (a_t)", "Goal Point"])

    # Add unit circle
    circle = plt.Circle((0, 0), env.goal_distance, color='black', alpha=.5, fill=False)
    plt.gcf().gca().add_artist(circle)

    final_states = np.array(final_states)
    goals = np.array(goals)
    diff = final_states - goals
    print(np.sum((np.linalg.norm(diff, axis=1) < 0.1).astype(int)))

    plt.show()

    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics([path])
    logger.dump_tabular()