
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    file = "./logs/td3-pointmass-singletask/td3-pointmass-singletask_2019_04_13_16_58_33_0000--s-0/params.pkl"
    data = joblib.load(file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    # while True:
    for i in range(10):
        path = rollout(
            env,
            policy,
            max_path_length=100,
            animated=False,
        )
        print(path)
        obs = path["observations"]
        plt.scatter(obs[0, 0], obs[0, 1]) # Start point
        plt.scatter(obs[1:, 0], obs[1:, 1])
        plt.scatter(1, 0) # Goal
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.show()

        rew = path["rewards"]
        plt.plot(rew)
        plt.show()

    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics([path])
    logger.dump_tabular()