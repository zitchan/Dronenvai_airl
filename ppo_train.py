from gymnasium.envs.registration import register
import gymnasium as gym
import torch
import random
from gymnasium.wrappers import RecordEpisodeStatistics
import subprocess
import threading
import numpy as np
import wandb
import pickle
import os
from tqdm import tqdm
import argparse
from stable_baselines3.common.utils import get_device
from customnetwork import feature_extractor as rnetwork
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor


def start_external_program(exe_path):
    subprocess.Popen([exe_path, '-windowed', '-ResX=1280', '-ResY=720'])


class EpisodeRewardCallback(BaseCallback):
    """
    自定义 Callback 来记录训练期间每个 episode 的平均奖励，并使用 episode 作为 TensorBoard 横坐标。
    """

    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0  # 记录 episode 数量

    def _on_step(self) -> bool:
        # 检查每一步的 infos 字段是否包含 episode 信息 (表示 episode 已经结束)
        infos = self.locals['infos']
        for info in infos:
            if 'episode' in info.keys():
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # 更新 episode 计数
                self.episode_count += 1

                # 计算并打印最近 100 个 episode 的平均奖励和长度
                mean_ep_reward = np.mean(self.episode_rewards[-100:])
                mean_ep_length = np.mean(self.episode_lengths[-100:])

                if self.verbose > 0:
                    print(
                        f"Episode: {self.episode_count}, Mean Episode Reward: {mean_ep_reward}, Mean Episode Length: {mean_ep_length}")

                # 记录到 TensorBoard，使用 episode_count 作为横坐标
                if self.logger is not None:
                    self.logger.record('episode/mean_reward', mean_ep_reward)
                    self.logger.record('episode/mean_length', mean_ep_length)
                    self.logger.record('episode/episode_count', self.episode_count)

                    # 调用 dump() 来传入 step (即 episode_count)
                    self.logger.dump(step=self.episode_count)

        return True


class LogActionRewardCallback(BaseCallback):
    def __init__(self, action_space, verbose=0):
        super(LogActionRewardCallback, self).__init__(verbose)
        self.action_space = action_space

    def _on_step(self) -> bool:
        # 获取当前动作和奖励
        action = self.locals["actions"]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = self.locals["rewards"]

        # 打印动作和奖励
        if self.verbose > 0:
            print(
                f"Step: {self.num_timesteps}, Action: {action}, Reward: {reward}, std:{self.model.policy.log_std.data}")
        return True  # 继续训练


class CustomActorCriticPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        # 提取特征
        latent_pi, latent_vf = self.mlp_extractor(self.features_extractor(obs))

        # 通过动作分布获取动作
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        # 计算动作的对数概率
        log_probs = distribution.log_prob(actions)

        # 获取值函数输出
        values = self.value_net(latent_vf)

        # 打印信息（可选）
        # print(f"mean: {distribution.distribution.mean} std: {distribution.distribution.stddev}")

        return actions, values, log_probs


def main():
    # arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe_path', type=str, default='D:\\zhechen\\baseline\\Easy\\Blocks.exe',
                        require=True)
    parser.add_argument('--seed', type=float, default=3)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ent_coef', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=float, default=512)
    parser.add_argument('--n_steps', type=float, default=2048)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--clip_range_vf', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=float, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=0.15)
    parser.add_argument('--train_step', type=float, default=260000)
    parser.add_argument('--mode', type=str, default='reward3', help='reward1 or reward2 or reward3 or airl',
                        require=True)
    args = parser.parse_args()

    thread = threading.Thread(target=start_external_program, args=(args.exe_path,))
    thread.start()
    mode = args.mode
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    if mode == "airl":
        print(f'starting training mode {args.mode}')
        # config = {
        #     "policy_type": "MlpPolicy",
        #     "total_timesteps": 25000,
        #     "env_id": "airsim-airl",
        # }
        # run = wandb.init(
        #     project="airl_ppo",
        #     config=config,
        #     sync_tensorboard=True,
        # )
        register(
            id='AirSim/AirSimEnv-v3',
            entry_point='airsim_env_airl:AirSimEnv_v3',
            max_episode_steps=300,  # 每个episode的最大步数
        )
        env = gym.make('AirSim/AirSimEnv-v3')
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=5.0)
        checkpoint_saving_path = './model/ppo_checkpoints/v3'
    elif mode == "reward1":
        print("train in the Sparse reward")
        register(
            id='AirSim/AirSimEnv-v1',
            entry_point='airsim_env_v1:AirSimEnv_v1',
            max_episode_steps=300,
        )
        # config = {
        #     "policy_type": "MlpPolicy",
        #     "total_timesteps": 25000,
        #     "env_id": "airsim-v1",
        # }
        # run = wandb.init(
        #     project="SparseR_ppo",
        #     config=config,
        #     sync_tensorboard=True,
        # )
        env = gym.make('AirSim/AirSimEnv-v1')
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_reward=1000)
        checkpoint_saving_path = './model/ppo_checkpoints/v1'
        print(f"checkpoint will save in {checkpoint_saving_path}")
    elif mode == "reward2":
        print("train in the human-designed reward")
        register(
            id='AirSim/AirSimEnv-v2',
            entry_point='airsim_env_v1:AirSimEnv_v2',
            max_episode_steps=300,
        )
        # config = {
        #     "policy_type": "MlpPolicy",
        #     "total_timesteps": 500000,
        #     "env_id": "airsim-v2",
        # }
        # run = wandb.init(
        #     project="humanR_ppo",
        #     config=config,
        #     sync_tensorboard=True,
        # )
        env = gym.make('AirSim/AirSimEnv-v2')
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_reward=100.0)
        checkpoint_saving_path = './model/ppo_checkpoints/v2'
        print(f"checkpoint will save in {checkpoint_saving_path}")
    elif mode == "reward3":
        print("train in the simple dense reward 2")
        register(
            id='AirSim/AirSimEnv-v4',
            entry_point='airsim_env_v1:AirSimEnv_v4',
            max_episode_steps=300,
        )
        # config = {
        #     "policy_type": "MlpPolicy",
        #     "total_timesteps": 500000,
        #     "env_id": "airsim-v4",
        # }
        # run = wandb.init(
        #     project="SparseR_ppo2",
        #     config=config,
        #     sync_tensorboard=True,
        # )
        env = gym.make('AirSim/AirSimEnv-v4')
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_reward=100.0)
        checkpoint_saving_path = './model/ppo_checkpoints/v4'
        print(f"checkpoint will save in {checkpoint_saving_path}")
    env = VecMonitor(env, filename='monitor_log.txt')

    device = get_device()
    policy_kwargs = dict(
        features_extractor_class=rnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], qf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-1.8
    )

    print(device)
    checkpoint_path = "./model/bc_pretrain/pretrain_model.zip1"
    if os.path.exists(checkpoint_path):
        print(f"==== 发现已有checkpoint，加载并继续训练: {checkpoint_path} ====")
        model = PPO.load(checkpoint_path, env=env, device="cuda")
        model.policy.log_std.data += 6.6
        print(model.policy.log_std)
    else:
        model = PPO(
            env=env,
            policy=ActorCriticPolicy,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.9,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            normalize_advantage=True,
            vf_coef=0.15,
            max_grad_norm=args.max_grad_norm,
            n_epochs=10,
            seed=random_seed,
            device="cuda",
            verbose=2,
        )
        model.policy.log_std.data[1] += 0.1
        print(model.policy.log_std)

    model.set_random_seed(random_seed)

    # CheckpointCallback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_saving_path,
                                             name_prefix='ppo_model')

    # action_callback = LogActionRewardCallback(env.action_space)
    # 开始训练
    model.learn(total_timesteps=args.train_step, callback=[checkpoint_callback,
        # WandbCallback(model_save_path=f"models/ppo/{run.id}",verbose=2,)
    ])
    model.save("./model/ppo_checkpoints/ppo_airsim_model")


if __name__ == '__main__':
    main()
