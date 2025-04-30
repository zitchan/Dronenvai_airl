import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from imitation.util.networks import BaseNorm, RunningNorm
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
from imitation.rewards.reward_nets import BasicShapedRewardNet
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast
from imitation.rewards.reward_nets import ShapedRewardNet, BasicPotentialMLP, BasicRewardNet


def obs_split(combined):
    batch_size = combined.shape[0]
    img = combined[:, :, :, :128]
    vel = combined[:, :, :, 128:]
    vel = vel.reshape(batch_size, 1, 72, 3)
    vel = vel[:, 0, 0, :]
    return img, vel


class feature_extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, normalize_velocity_layer=None, normalize_image_layer=None,
                 features_dim=96):
        super().__init__(observation_space, features_dim=features_dim)
        self.normalize_velocity_layer = normalize_velocity_layer
        self.normalize_image_layer = normalize_image_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3[0].weight.data.normal_(0, 0.5)  # initialization
        self.image_fc1 = nn.Linear(256, 64, bias=False)
        self.image_fc2 = nn.Linear(64, 48, bias=False)
        self.image_fc1.weight.data.normal_(0, 0.5)
        self.image_fc2.weight.data.normal_(0, 0.5)

        self.vel_fc1 = nn.Linear(3, 48, bias=False)
        self.vel_fc1.weight.data.normal_(0, 0.5)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, obs):
        image, vel = obs_split(obs)

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.image_fc1(x)
        x = self.relu(x)
        x = self.image_fc2(x)

        vel = self.vel_fc1(vel)

        state_process = torch.cat((x, vel), dim=1)
        state_process = self.tanh(state_process)

        return state_process


class CustomShapedRewardNet(ShapedRewardNet):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            *,
            normalize: Optional[Type[RunningNorm]] = None,
            reward_hid_sizes: Sequence[int] = (64, 32),
            potential_hid_sizes: Sequence[int] = (64, 32),
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = False,
            use_done: bool = False,
            if_share_feature_extractor: bool = False,
            discount_factor: float = 0.99,
            feature_extractor: Type[BaseFeaturesExtractor],
            feature_output_dim: int,
            **kwargs,
    ):
        # 保存传入参数为实例属性
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.discount_factor = discount_factor
        self.if_share_feature_extractor = if_share_feature_extractor
        self.normalize = normalize

        # Initialize the base reward network
        base_reward_net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            hid_sizes=reward_hid_sizes,
            **kwargs,
        )

        # Initialize the potential network
        potential_net = BasicPotentialMLP(
            observation_space=gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(feature_output_dim,)
            ),
            hid_sizes=potential_hid_sizes,
            **kwargs,
        )

        super().__init__(
            base=base_reward_net,
            potential=potential_net,
            discount_factor=discount_factor,
        )

        # Feature extractor initialization

        self.feature_extractor = feature_extractor(observation_space, **kwargs)
        if not self.if_share_feature_extractor:
            self.feature_extractor_potential = feature_extractor(observation_space, **kwargs)
        # Linear transform to match reward network input
        self.feature_transform = nn.Linear(
            feature_output_dim, reward_hid_sizes[0]  # Match input size of the first layer
        )
        self.action_transform = nn.Linear(
            3, reward_hid_sizes[0]  # Match input size of the first layer
        )
        # Normalize layer

        self.mlp = nn.Sequential(
            nn.Linear(reward_hid_sizes[0]*2, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.potential = nn.Sequential(
            nn.Linear(reward_hid_sizes[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs, action, next_obs, done, **kwargs):
        # share feature extractor will be realized in the future
        if not self.if_share_feature_extractor:
            obs_feature = self.feature_extractor(obs)
            obs_feature = self.feature_transform(obs_feature)
            if self.use_next_state:
                next_obs_feature = self.feature_extractor_potential(next_obs)
                next_obs_feature = self.feature_transform(next_obs_feature)
            action = self.action_transform(action)
            inputs = []
            if self.use_state:
                inputs.append(torch.flatten(obs_feature, 1))
            if self.use_action:
                inputs.append(torch.flatten(action, 1))
            combined_inputs = torch.cat(inputs, dim=-1)
            reward = self.mlp(combined_inputs)
            if self.use_next_state:
                new_shaping_output = self.potential(next_obs_feature).flatten()
                old_shaping_output = self.potential(obs_feature).flatten()
                reward = reward.squeeze(-1)
                old_shaping_output = old_shaping_output.squeeze(-1)
                new_shaping_output = new_shaping_output.squeeze(-1)
                new_shapping = (1-done.float()) * new_shaping_output
                final_reward = (reward + self.discount_factor * new_shapping - old_shaping_output)
            else:
                final_reward = reward.squeeze(-1)

        # print(final_reward.shape)
        assert final_reward.shape == obs.shape[:1]
        return final_reward





