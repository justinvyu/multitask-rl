from gym_pointmass.envs import PointMassEnv, PointMassEnvRewardType

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


from multiworld.core.flat_goal_env import FlatGoalEnv
import numpy as np

def experiment(variant):
    base_expl_env = PointMassEnv(n=2, reward_type=variant["reward_type"])
    expl_env = FlatGoalEnv(
        base_expl_env,
        append_goal_to_obs=True
    )
    base_eval_env = PointMassEnv(n=2, reward_type=variant["reward_type"])
    eval_env = FlatGoalEnv(
        base_eval_env,
        append_goal_to_obs=True
    )
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    print(expl_env.observation_space, expl_env.action_space)
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.train()


if __name__ == "__main__":

    # base_env = PointMassEnv(n=1)
    # env = FlatGoalEnv(
    #     base_env,
    #     append_goal_to_obs=True
    # )
    # obs = env.reset()
    # print(env.observation_space, env.wrapped_env.observation_space)
    # print("obs: ", obs)
    # for i in range(100):
    #     # next_obs, reward, done, _ = env.step(np.array([0.2, 0.3]))
    #     next_obs, reward, done, _  = env.step(2 * np.random.rand(2) - 1)
    #     print("next_obs: {0}, reward: {1}, done: {2}".format(next_obs, reward, done))

    variant = dict(
        reward_type=PointMassEnvRewardType.DISTANCE,
        algorithm_kwargs=dict(
            num_epochs=500,
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=250,
            num_expl_steps_per_train_loop=250,
            min_num_steps_before_training=100,
            max_path_length=100,
            batch_size=50,
        ),
        trainer_kwargs=dict(
            discount=0.99,
        ),
        qf_kwargs=dict(
            hidden_sizes=[64, 32],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 32],
        ),
        replay_buffer_size=int(1E6),
    )
    setup_logger("td3-pointmass-multitask", variant=variant)
    experiment(variant)