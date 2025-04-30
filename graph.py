import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')

# 文件路径与标签
files = ['reward1.pkl', 'reward2.pkl', 'reward3.pkl', 'reward4.pkl']
labels = ['Our method', 'Human-designed reward', 'Sparse reward', 'Simple dense reward']
colors = ['#2878B5', '#9AC9DB', '#F8AC8C', '#FF8884']  # 蓝、橙、绿

# 读取数据
all_success = []
all_distances = []
training_steps = None

for file in files:
    path = os.path.join('output', file)
    with open(path, 'rb') as f:
        data = pickle.load(f)
        all_success.append(np.array(data['success_rates'])*100)
        all_distances.append(np.array(data['distances']))
        if training_steps is None:
            training_steps = np.array(data['numbers']) * 10000

# --- 设置分组柱状图参数 ---
n_groups = len(training_steps)
bar_width = 0.25
x = np.arange(n_groups)  # 每个 step 一组

# --- 柱状图 ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# 每组柱子偏移位置
for i in range(4):
    ax1.bar(x + (i - 1) * bar_width, all_success[i],
            width=bar_width, label=labels[i], color=colors[i])

    # 平均线
    avg = np.mean(all_success[i])
    ax1.axhline(y=avg, color=colors[i], linestyle='--', linewidth=1.5, label=f'Avg. {labels[i]}')

# 设置轴
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Success Rate (%)')
ax1.set_ylim(0, 150)
ax1.set_xticks(x)
ax1.set_xticklabels([str(int(step)) for step in training_steps], rotation=45)
ax1.set_title('Comparison of Success Rates')
ax1.legend()
fig1.tight_layout()

# --- 折线图 ---

fig2, ax2 = plt.subplots(figsize=(10, 5))

line_styles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', '+']
face_colors = ['#2878B5', '#9AC9DB', '#F8AC8C', '#FF8884']
edge_colors = ['#2878B5', '#9AC9DB', '#F8AC8C', '#FF8884']  # 和曲线颜色一致

for i in range(4):
    ax2.plot(training_steps, all_distances[i],
             label=labels[i],
             color=edge_colors[i],
             linestyle=line_styles[i],
             marker=markers[i],
             markersize=7,
             markerfacecolor=face_colors[i],
             markeredgecolor=edge_colors[i],
             markeredgewidth=1.5,
             linewidth=2)

ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Mean Distance')
ax2.set_ylim(10, 60)
ax2.set_title('Comparison of Mean Distances')
ax2.legend()
fig2.tight_layout()



plt.show()
