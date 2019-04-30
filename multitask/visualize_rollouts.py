
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_rollouts(results,
                       show_threshold_circles=True,
                       completion_threshold=0.25):
    sns.set()
    plt.figure(figsize=(8, 8))

    paths = results["paths"]
    env = results["env"]

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

        plt.scatter(goal_x, goal_y, color="r")  # Goal

    plt.xlim(-env.bound, env.bound)
    plt.ylim(-env.bound, env.bound)

    # Legend
    plt.legend(["Initial State (s_0)", "States (s_t)", "Actions (a_t)", "Goal Point"])

    # Add unit circle
    circle = plt.Circle((0, 0), env.goal_distance, color='black', alpha=.5, fill=False)
    plt.gcf().gca().add_artist(circle)

    # Draw circles around the goal points with radius completion threshold.
    if show_threshold_circles:
        for goal in env.goals:
            circle = plt.Circle(goal, 0.25, color='red', alpha=.75, fill=False)
            plt.gcf().gca().add_artist(circle)

    final_states = np.array(results["final_states"])
    goals = np.array(results["goal_states"])
    diff = final_states - goals
    completed_trials = (np.linalg.norm(diff, axis=1) < completion_threshold).astype(int)
    completion = np.sum(completed_trials) / len(paths)
    plt.title("% task completion: {0:.0%}, {1} rollouts".format(completion, len(paths)))

    plt.show()