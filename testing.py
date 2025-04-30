import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import random
from stable_baselines3.common.utils import set_random_seed
import subprocess
import threading
from stable_baselines3 import PPO
from customnetwork import feature_extractor as fnetwork
from stable_baselines3.common.utils import get_device
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle
import argparse


def start_external_program(exe_path):
    subprocess.Popen([exe_path, '-windowed', '-ResX=1280', '-ResY=720'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe_path', type=str, default='D:\\zhechen\\baseline\\Easy\\Blocks.exe', required=True)
    parser.add_argument('--seed', type=float, default=345)
    parser.add_argument('--episode', type=float, default=30)
    parser.add_argument('--mode', type=str, default='reward4', help='reward1 or reward2 or reward3 or airl'
                        , required=True)
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
        id='AirSim_testing/AirSimEnv-testing',  # 环境的ID
        entry_point='airsim_testing:AirSimEnv_testing',  # 指向自定义环境的模块和类
        max_episode_steps=300,  # 每个episode的最大步数
    )
    env = gym.make('AirSim_testing/AirSimEnv-testing')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_reward=10)

    device = get_device()
    print(device)

    policy_kwargs = dict(
        features_extractor_class=fnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], qf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-1
    )
    mean_distance_array = []
    success_array = []
    numbers = []
    step = 0
    if args.mode == 'airl':
        version = "v3"
    elif args.mode == 'reward1':
        version = "v1"
    elif args.mode == 'reward2':
        version = "v2"
    elif args.mode == 'reward3':
        version = "v4"
    else:
        print("default running")
        version = "v3"
    for n in range(1, 26):
        model_path = f'./model/ppo_checkpoints/{version}/ppo_model_{n}0000_steps.zip'
        print(f"loading model step {n}0000")
        if os.path.exists(model_path):
            print(f"Loading pretrained PPO model from {model_path}...")
            model = PPO.load(model_path, load_optimizer=False, device="cuda")
        episodes = args.episode
        success = 0
        for episode in range(episodes):
            obs = env.reset()
            done = False
            distance_array = []
            distance = 0

            while not done:
                # 使用模型选择动作
                step += 1
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated = env.step(action)
                done = terminated
                distance = reward
            else:
                distance = reward
                distance_array.append(distance)
                if distance > 54.9:
                    success += 1
                print(distance, success)
        numbers.append(n)
        mean_distance = np.mean(distance_array)
        mean_distance_array.append(mean_distance)
        success_rate = success / episodes
        success_array.append(success_rate)
        print(mean_distance, success_rate)
        data = {
            'numbers': numbers,
            'success_rates': success_array,
            'distances': mean_distance_array
        }
        with open(f'{args.mode}.pkl', 'wb') as f:
            pickle.dump(data, f)

    env.close()


if __name__ == '__main__':
    main()
