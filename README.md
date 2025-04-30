# DroneNav-AIRL

A PyTorch-based implementation of Adversarial Inverse Reinforcement Learning (AIRL) for vision-based continuous-control drone navigation. This repository provides training, evaluation, and reward visualization tools for UAV navigation tasks in AirSim environments.

## 🚀 Requirement
unreal engine = 4.27<br>
airsim = 1.5.0<br>
gymnasium >= 0.29.1<br>
stablebaseline3 = 2.2.1<br>
torch >= 2.6.0<br>
imitation = 1.0.1<br>

## 💻 Equirement
Windows 10<br>
Python 3.10.13<br>

## 🔌 Prepare training
Download：https://drive.google.com/drive/folders/1fHg3iDCTcPSZ6J9T8eIQB5l8gxJmqQEL?usp=sharing <br>
Unzip all file & Extract the **model** into the project root <br>
Unzip the env to whatever u like

## 🔥 Training

### 1. Pretrain

```bash
python pretrain_ppo.py \
  --exe_path "path_to\Blocks.exe" \
  --seed 345 \
  --gae_lambda 0.95 \
  --ent_coef 0 \
  --learning_rate 1e-4 \
  --batch_size 512 \
  --n_steps 2048 \
  --clip_range 0.2 \
  --clip_range_vf 0.1 \
  --n_epochs 10 \
  --max_grad_norm 0.15 \
  --train_step 50000 \
  --testing False \
  --testing_episode 10

### 2. Extract reward funcion

```bash
python airl_train.py \
  --exe_path "path_to\Blocks.exe" \
  --seed 345 \
  --pretrain True \
  --pretrain_path "./model/bc_pretrain/pretrain_model.zip"

### 3. Evaluate reward function

```bash
python ppo_train.py \
  --exe_path "path_to\Blocks.exe" \
  --seed 3 \
  --gae_lambda 0.95 \
  --ent_coef 0 \
  --learning_rate 1e-4 \
  --batch_size 512 \
  --n_steps 2048 \
  --clip_range 0.2 \
  --clip_range_vf 0.1 \
  --n_epochs 10 \
  --max_grad_norm 0.15 \
  --train_step 260000 \
  --mode "airl" or "rewar1"  or "rewar2" or "rewar3"

## 😄 Reference

Reference environment and part of reward function idea for my baseline training **https://github.com/sunghoonhong/AirsimDRL?tab=readme-ov-file#check-1-min-madness **
