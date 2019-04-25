from gym_pointmass.envs import PointMassEnv
from gym_pointmass.envs import PointMassEnvRewardType
from multiworld.core.flat_goal_env import FlatGoalEnv

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import TanhMlpPolicy
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy

from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

def experiment(variant):
    base_expl_env = PointMassEnv(
        n=variant["num_tasks"],
        reward_type=PointMassEnvRewardType.DISTANCE
    )
    expl_env = FlatGoalEnv(
        base_expl_env,
        append_goal_to_obs=True
    )

    base_eval_env = PointMassEnv(
        n=variant["num_tasks"],
        reward_type=PointMassEnvRewardType.DISTANCE
    )
    eval_env = FlatGoalEnv(
        base_eval_env,
        append_goal_to_obs=True
    )
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        std=0.1
    )
    # es = GaussianStrategy(
    #     action_space=expl_env.action_space,
    #     max_sigma=0.1,
    #     min_sigma=0.1
    # )
    # policy =TanhMlpPolicy(
    #     input_size=obs_dim,
    #     output_size=action_dim,
    #     hidden_sizes=[M, M]
    # )
    # expl_policy = PolicyWrappedWithExplorationStrategy(
    #     exploration_strategy=es,
    #     policy=policy
    # )
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
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
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
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=64,
        num_tasks=1,
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
        ),
    )
    setup_logger('sac-pointmass-multitask-' + str(variant["num_tasks"]), variant=variant)
    experiment(variant)