import time
import numpy as np
import torch
import airsim
from PIL import Image
import gymnasium as gym
from gymnasium import spaces

clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -0.5
floorZ = 1.18
goals = [7, 17, 27.5, 45, goalY]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']


def obs_combine(img, vel):
    vel = np.tile(vel[np.newaxis, np.newaxis, :], (1, 72, 1))
    combined = np.concatenate((img, vel), axis=-1)
    return combined


class AirSimEnv_v3(gym.Env):
    def __init__(self):
        super(AirSimEnv_v3, self).__init__()
        self.episode = 0
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0
        self.img_seq = None
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 72, 131),
                                            dtype=np.float32)

        self.reward_net = torch.load('./model/airl/reward/reward_net65.pth', weights_only=False, map_location=torch.device("cuda:0"))
        self.reward_net.eval()

        self.distance = []
        self.max_distance = 0
        self.episode_distance = 0

    def transform_input(self, osb):
        responses = osb[0]
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        image = Image.fromarray(img2d)
        image = np.array(image.resize((72, 128)).convert('L'))
        image = np.float32(image.reshape(1, 72, 128))
        image /= 255.
        vel = np.array(osb[1], dtype=np.float32)

        return [image, vel]

    def reset(self, seed=None, options=None):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * timeslice).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * timeslice).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        observation = self.get_state()
        image = observation[0]
        vel = observation[1]
        # LSTM
        # self.img_seq = np.repeat(image[np.newaxis, :], 5, axis=0)
        # histroy = self.img_seq.copy()
        self.img_seq = image
        history = self.img_seq.copy()
        obs = obs_combine(history, vel)
        info = {}
        return obs, info

    def get_state(self):
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        observation = self.transform_input(observation)
        return observation

    def step(self, action):
        # print("action: ", quad_offset)
        # move with given velocity
        observation = self.get_state()
        image = observation[0]
        vel = observation[1]
        obs = obs_combine(image, vel)

        self.client.simPause(False)
        quad_offset = [float(i) for i in action]
        x, y, z = quad_offset
        # print(x, y, z)
        has_collided = False
        landed = False
        self.client.moveByVelocityAsync(1.5 * x, 1.5 * y + 0.5, 1.5 * z, timeslice)
        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided
            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = landed or quad_pos.z_val > floorZ
            collision = collided or landed
            if collision:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break
        self.client.simPause(True)

        try:
            observation = self.get_state()
        except Exception as e:
            time.sleep(10)
            observation = self.get_state()

        quad_pos = self.client.getMultirotorState().kinematics_estimated.position

        # decide whether done
        dead = has_collided or quad_pos.y_val <= outY
        done = dead or quad_pos.y_val >= goalY
        if quad_pos.y_val > self.max_distance:
            self.max_distance = quad_pos.y_val

        if quad_pos.y_val > self.episode_distance:
            self.episode_distance = quad_pos.y_val

        self.img_seq = observation[0]
        history_ = self.img_seq.copy()
        obs_ = obs_combine(history_, observation[1])
        reward = self.compute_reward(obs, action, obs_, done)
        # log info
        info = {}
        info['Y'] = quad_pos.y_val
        info['level'] = self.level
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= outY:
            info['status'] = 'out'
        elif quad_pos.y_val >= goalY:
            info['status'] = 'goal'
        else:
            info['status'] = 'going'
        # if self.episode % 5 == 0:

        # print(f'action: {quad_offset}')
        print(f'reward: {reward}')
        if done:
            self.episode += 1
            self.distance.append(self.episode_distance)
            if self.episode % 100 == 0:
                print(f'episode distance: {self.episode_distance} maxdistance: {self.max_distance} meandistance: {np.mean(self.distance)}')
            self.episode_distance = 0

        return obs_, reward, done, False, info

    def render(self):
        pass

    def compute_reward(self, observation, action, observation_, done):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(torch.device("cuda:0"))
        obs_ = torch.tensor(observation_, dtype=torch.float32).unsqueeze(0).to(torch.device("cuda:0"))
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(torch.device("cuda:0"))
        with torch.no_grad():
            reward = self.reward_net(obs, action, obs_, done)
        output = reward.cpu().numpy().item()

        return output

    def close(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')
