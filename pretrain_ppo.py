import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.envs.registration import register
import random
from stable_baselines3.common.utils import set_random_seed
import subprocess
import threading
from stable_baselines3 import PPO
from customnetwork import feature_extractor as fnetwork
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.pyplot as plt
import argparse


def start_external_program(exe_path):
    subprocess.Popen([exe_path, '-windowed', '-ResX=1280', '-ResY=720'])


def main():
    # arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe_path', type=str, default='D:\\zhechen\\baseline\\Easy\\Blocks.exe')
    parser.add_argument('--seed', type=float, default=345)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ent_coef', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=float, default=512)
    parser.add_argument('--n_steps', type=float, default=2048)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--clip_range_vf', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=float, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=0.15)
    parser.add_argument('--train_step', type=float, default=50000)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--testing_episode', type=float, default=10)
    args = parser.parse_args()

    thread = threading.Thread(target=start_external_program, args=(args.exe_path,))
    thread.start()
    random_seed = args.seed
    random.seed(random_seed)
    # np.random.seed(random_seed)
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    register(
        id='AirSim/AirSimEnv-v0',
        entry_point='airsim_env_v1:AirSimEnv_v1',
        max_episode_steps=600,
    )

    env = gym.make('AirSim/AirSimEnv-v0')

    device = get_device()
    print(device)

    policy_kwargs = dict(
        features_extractor_class=fnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], qf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-1
    )

    learner = PPO(
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
        n_epochs=args.n_epochs,
        seed=random_seed,
        device="cuda",
        verbose=2,
    )
    learner.set_random_seed(random_seed)

    # LOAD
    expert_data = torch.load("./expert/pretrain.pth", weights_only=False)
    batch_size = 512
    obs = expert_data['obs']
    acts = expert_data['acts']
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    acts = torch.tensor(acts, dtype=torch.float32).to(device)

    dataset = TensorDataset(obs, acts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = learner.policy
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    losses = []

    for epoch in range(args.train_step):
        for batch_obs, batch_acts in dataloader:
            latent_pi, _ = policy.mlp_extractor(policy.features_extractor(batch_obs))
            dist = policy._get_action_dist_from_latent(latent_pi)

            # 计算克隆损失（负对数似然）
            loss = -dist.log_prob(batch_acts).mean()

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    learner.save("./model/bc_pretrain/pretrain_model")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss", color="blue")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve During Training")
    plt.legend()
    plt.grid()
    plt.show()

    if args.testing:
        log_std = learner.policy.log_std
        print("Original log_std:", log_std)
        for episode in range(args.testing_episode):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                # 使用模型选择动作
                action, _states = learner.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()


if __name__ == '__main__':
    main()
