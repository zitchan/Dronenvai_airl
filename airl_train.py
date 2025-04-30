from gymnasium.envs.registration import register
import torch
import argparse
from imitation.util import logger
from torch.optim import Adam
from imitation.algorithms.adversarial.common import AdversarialTrainer
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import get_device
from imitation.algorithms.adversarial.airl import AIRL
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import DDPG, PPO
from customnetwork import feature_extractor as fnetwork
from customnetwork import CustomShapedRewardNet as rewardnetwork
from customnetwork import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import rollout
import subprocess
import threading
import json
import numpy as np
from imitation.data import serialize
import os
from stable_baselines3.common.policies import ActorCriticPolicy
import gzip
import pickle


def start_external_program(exe_path):
    subprocess.Popen([exe_path, '-windowed', '-ResX=1280', '-ResY=720'])


def convert_trajectories_to_expert_data(trajectories):
    expert_data = {
        "obs": [],
        "acts": [],
        "next_obs": [],
        "dones": [],
    }
    for traj in trajectories:
        expert_data["obs"].append(traj.obs[:-1])  # 当前状态
        expert_data["acts"].append(traj.acts)  # 动作
        expert_data["next_obs"].append(traj.obs[1:])  # 下一状态
        expert_data["dones"].append([False] * (len(traj.acts) - 1) + [True])  # dones 标志

    # 转换为 numpy 格式并返回
    expert_data = {k: np.concatenate(v) for k, v in expert_data.items()}
    return expert_data


def save_rollout_buffer_to_json(round_num: int, gen_algo):
    """
    保存每一回合的 rollout_buffer 数据到单独的 JSON 文件
    """
    if hasattr(gen_algo, "rollout_buffer"):
        buffer = gen_algo.rollout_buffer

        # 提取观测、动作和奖励
        data = {
            "round": round_num,
            "actions": buffer.actions.tolist(),
            "rewards": buffer.rewards.tolist()
        }

        # 每个回合一个单独的文件
        file_name = f"./rollout/rollout_buffer_round_{round_num}.json"
        with open(file_name, "w") as file:
            json.dump(data, file, indent=3)

        print(f"Round {round_num} data saved to {file_name}")


class CustomActorCriticPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        # 提取特征
        latent_pi, latent_vf = self.mlp_extractor(self.features_extractor(obs))

        # 通过动作分布获取动作
        distribution = self._get_action_dist_from_latent(latent_pi)
        if hasattr(distribution.distribution, 'stddev'):
            with torch.no_grad():
                max_std = 0.5  # 目标标准差上限
                distribution.distribution.stddev.clamp_(max=max_std)
        actions = distribution.get_actions(deterministic=deterministic)

        # 计算动作的对数概率
        log_probs = distribution.log_prob(actions)

        # 获取值函数输出
        values = self.value_net(latent_vf)

        # 打印信息（可选）
        print(f"mean: {distribution.distribution.mean} std: {distribution.distribution.stddev}")

        return actions, values, log_probs


def main():
    # arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe_path', type=str, default='D:\\zhechen\\baseline\\Easy\\Blocks.exe')
    parser.add_argument('--seed', type=float, default=123)
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_path', type=str, default='./model/bc_pretrain/pretrain_model.zip')
    args = parser.parse_args()

    thread = threading.Thread(target=start_external_program, args=(args.exe_path,))
    thread.start()

    random_seed = args.seed
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    register(
        id='AirSim/AirSimEnv-v0',  # 环境的ID
        entry_point='airsim_env:AirSimEnv_v2',  # 指向自定义环境的模块和类
        max_episode_steps=600,  # 每个episode的最大步数
    )

    env = make_vec_env(
        'AirSim/AirSimEnv-v0',
        rng=np.random.default_rng(random_seed),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    # env = RecordEpisodeStatistics(gym.make('AirSim/AirSimEnv-v0'), deque_size=100)
    device = get_device()

    file_path = os.path.join("expert", "expert_data_all.pkl")
    if os.path.exists(file_path):
        print("Loading expert rollouts...")
        rollouts = serialize.load(file_path)
        print("success loading expert rollouts...")
    else:
        model_path = "./model/bc_pretrain/pretrain_model.zip"
        expert = PPO.load(model_path)
        print("Sampling expert rollouts...")
        rollouts = rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_episodes=24),
            rng=np.random.default_rng(random_seed),
        )
        # serialize.save(file_path, rollouts)

    policy_kwargs = dict(
        features_extractor_class=fnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], qf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-2.7
    )

    learner = PPO(
        env=env,
        policy=ActorCriticPolicy,
        gae_lambda=0.95,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.9,
        clip_range=0.2,
        clip_range_vf=0.1,
        normalize_advantage=True,
        vf_coef=0.15,
        max_grad_norm=2,
        n_epochs=10,
        seed=random_seed,
        device="cuda",
        verbose=2,
    )

    # PPO
    model_path = args.pretrain_path
    if os.path.exists(model_path) and args.pretrain:
        del learner
        print(f"Loading pretrained PPO model from {model_path}...")
        learner = PPO.load(model_path, env=env, load_optimizer=False, print_system_info=True,
                           device="cuda")  # 加载模型并绑定环境
        # print(learner.policy.optimizer)
        learner.policy.optimizer = Adam(learner.policy.parameters(), lr=5e-5)
        print(learner.policy.optimizer)
        print(learner.policy.log_std)
        learner.policy.log_std.data += 6
        print(learner.policy.log_std)

    # reward_path = "./model/airl/reward/reward_net39.pth"
    # if os.path.exists(reward_path):
    #     reward_net = torch.load(reward_path, weights_only=False)
    # else:
    reward_net = rewardnetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize=RunningNorm,
        feature_extractor=fnetwork,
        use_state=True,
        use_action=True,
        use_next_state=True,
        use_done=True,
        discount_factor=0.99,
        feature_output_dim=96,
        if_share_feature_extractor=False
    )

    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=512,
        gen_replay_buffer_capacity=2048,
        gen_train_timesteps=500,
        n_disc_updates_per_round=4,
        disc_opt_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        init_tensorboard=True,
        log_dir="./airl_tensorboard/",
    )

    airl_trainer.allow_variable_horizon = True
    for n in range(1, 50):
        print(f"This is the No. {n}")
        airl_trainer.train(15000)
        torch.save(reward_net, f"./model/airl/reward/reward_net{n}.pth")
        learner.save(f"./model/airl/policy/ppo_policy_{n}.zip")


if __name__ == '__main__':
    main()
