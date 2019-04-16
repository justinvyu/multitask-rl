
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
    print("Policy loaded")
    # while True:
    plt.figure(figsize=(8, 8))
    goals = env.goals
    print(env.goals)
    for i in range(150):
        path = rollout(
            env,
            policy,
            max_path_length=100,
            animated=False,
        )
        # print(path)
        obs = path["observations"]
        acts = path["actions"]
        start_x = obs[0, 0]
        start_y = obs[0, 1]
        # for i, (o, a) in enumerate(zip(obs, acts)):
        #     start_x, start_y = o[0], o[1]
        #     act_x, act_y = a[0], a[1]
        #     plt.scatter(start_x, start_y)  # Start point
        #     plt.quiver(start_x, start_y, act_x, act_y, angles='xy', scale_units='xy', scale=1)
        # plt.scatter(start_x, start_y, color="b")
        # plt.annotate("start=({0}, {1})".format(start_x.round(4), start_y.round(4)), (start_x, start_y), xytext=(start_x-.5, start_y+.2))
        plt.scatter(start_x, start_y, color="green")
        plt.scatter(obs[1:, 0], obs[1:, 1], color="b")
        plt.quiver(obs[:, 0], obs[:, 1], acts[:, 0], acts[:, 1],
                   angles='xy', scale_units='xy', scale=1, width=.005, headwidth=3, alpha=.9)
        final_x, final_y = obs[len(obs) - 1, 0], obs[len(obs) - 1, 1]
        # plt.annotate("end=({0}, {1})".format(final_x.round(4), final_y.round(4)), (final_x, final_y), xytext=(final_x-.5, final_y-.2))

        goal_idx = np.argmax(obs[0, 2:])
        goal = env.goals[goal_idx]
        goal_x, goal_y = goal[0], goal[1]
        # plt.annotate("goal=({0}, {1})".format(goal_x.round(4), goal_y.round(4)), (goal_x, goal_y), xytext=(goal_x-.5, goal_y+.1))
        plt.scatter(goal[0], goal[1], color="r") # Goal
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        # rew = path["rewards"]
        # plt.plot(rew)
        # plt.show()
    plt.legend(["Initial State (s_0)", "States (s_t)", "Actions (a_t)", "Goal Point"])
    circle = plt.Circle((0, 0), 1, color='black', alpha=.5, fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.show()

    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics([path])
    logger.dump_tabular()