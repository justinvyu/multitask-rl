
from rlkit.samplers.util import rollout
import joblib
import numpy as np

def run_policy(fn, num_rollouts):
    data = joblib.load(fn)

    # Load deterministic evaluation policy.
    policy = data["evaluation/policy"]
    env = data["evaluation/env"]
    return _run_policy(env, policy, num_rollouts)

def run_distilled_policy(fn, num_rollouts):
    data = joblib.load(fn)
    policy = data["policy"]
    env = data["env"]
    return _run_policy(env, policy, num_rollouts)

def _run_policy(env, policy, num_rollouts):
    """
    Takes in a trained policy, runs the policy for a specified number of
    rollouts, and returns the results of the experiment.
    :param fn: The path to the `params.pkl` file containing the trained policy.
    :param num_rollouts: The number of rollouts to experience.
    :return: A list of the `num_rollouts` recorded paths to be used for
             further analysis.
    """
    start_states, final_states, goal_states, actions, paths = [], [], [], [], []

    for i in range(num_rollouts):
        path = rollout(
            env,
            policy,
            max_path_length=100,
            animated=False,
        )
        obs = path["observations"]
        acts = path["actions"]

        goal_idx = np.argmax(obs[0, 2:])
        start_x, start_y = obs[0, 0], obs[0, 1]
        acts_x, acts_y = acts[:, 0], acts[:, 1]
        final_x, final_y = obs[len(obs) - 1, 0], obs[len(obs) - 1, 1]
        goal = env.goals[goal_idx]
        goal_x, goal_y = goal[0], goal[1]

        start_states.append(np.array([start_x, start_y]))
        final_states.append(np.array([final_x, final_y]))
        goal_states.append(np.array([goal_x, goal_y]))
        actions.append(np.array([acts_x, acts_y]))
        paths.append(path)

    return dict(
        start_states=np.array(start_states),
        final_states=np.array(final_states),
        goal_states=np.array(goal_states),
        actions=np.array(actions),
        paths=paths,
        env=env
    )