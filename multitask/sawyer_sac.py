from gym_pointmass.envs import PointMassEnv
from gym_pointmass.envs import PointMassEnvRewardType
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv
import numpy as np

import joblib
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

def run_sac(base_expl_env, base_eval_env, variant):
    expl_env = base_expl_env
    eval_env = base_eval_env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant["layer_size"]
    num_hidden = variant["num_hidden_layers"]
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_hidden
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant["trainer_kwargs"]
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.train()

def train_policy():
    base_expl_env = SawyerReachXYZEnv()
    base_eval_env = SawyerReachXYZEnv()
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=64,
        num_hidden_layers=2,
        replay_buffer_size=int(1e5),
        algorithm_kwargs=dict(
            num_epochs=50,
            num_eval_steps_per_epoch=1500,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=200,
            max_path_length=200,
            batch_size=100,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        )
    )
    setup_logger("sac-reachxyz", variant=variant)
    run_sac(base_expl_env, base_eval_env, variant)

if __name__ == "__main__":
    # train_policy()

    # file = "./logs/sac-hammer_2019_05_02_11_14_48_0000--s-0/params.pkl"
    # data = joblib.load(file)
    # policy = data["evaluation/policy"]
    # env = data["evaluation/env"]

    env = SawyerReachPushPickPlace6DOFEnv()

    while True:
        obs = env.reset()
        done = False
        while not done:
            action = (np.random.rand(3) * 2 - 1) * 0.5
            print(action)
            obs, rew, done,  _ = env.step(action)
            print(obs)
            env.render()
