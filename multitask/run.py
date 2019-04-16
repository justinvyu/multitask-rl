
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # file = "./logs/sac-pointmass-multitask/sac-pointmass-multitask_2019_04_13_23_32_53_0000--s-0/params.pkl"
    # file = "./logs/sac-pointmass-multitask-3/sac-pointmass-multitask-3_2019_04_14_00_30_43_0000--s-0/params.pkl"
    # file = "./logs/sac-pointmass-multitask-5/sac-pointmass-multitask-5_2019_04_14_01_40_18_0000--s-0/params.pkl"
    # file = "./logs/sac-pointmass-multitask-10/sac-pointmass-multitask-10_2019_04_14_13_54_28_0000--s-0/params.pkl"
    file = "./logs/sac-pointmass-multitask-25/sac-pointmass-multitask-25_2019_04_15_23_30_06_0000--s-0/params.pkl"

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
        goal_plot.scatter(goal[0], goal[1], color="r") # Goal
        goal_plot.set_xlim(-1.5, 1.5)
        goal_plot.set_ylim(-1.5, 1.5)


        if not has_circle[goal_idx]:
            circle = plt.Circle((0, 0), 1, color='black', alpha=.5, fill=False)
            goal_plot.add_artist(circle)
            has_circle[goal_idx] = True

        # rew = path["rewards"]
        # plt.plot(rew)


    # Legend
    # plt.legend(["Initial State (s_0)", "States (s_t)", "Actions (a_t)", "Goal Point"])

    # Add unit circle
    # circle = plt.Circle((0, 0), 1, color='black', alpha=.5, fill=False)
    # plt.gcf().gca().add_artist(circle)

    plt.show()

    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics([path])
    logger.dump_tabular()