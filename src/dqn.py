"""
DQN (Deep Q-Network) 算法实现 - CartPole-v1 环境
理论基础：使用深度神经网络近似 Q 函数，结合经验回放和目标网络实现稳定训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

# ==================== 1. Q 网络 ====================
class QNetwork(nn.Module):
    """
    Q 网络：使用神经网络来近似 Q 函数 Q(s, a)
    输入：状态 s (4维: 小车位置、速度、杆子角度、角速度)
    输出：每个动作的 Q 值 (2维: 左、右)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化网络结构
        Args:
            state_dim: 状态空间的维度 (CartPole 中是 4)
            action_dim: 动作空间的维度 (CartPole 中是 2)
            hidden_dim: 隐藏层神经元数量
        """
        super().__init__()
        # 使用 Sequential 构建三层全连接网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 输入层 → 隐藏层1
            nn.ReLU(),                          # ReLU 激活函数，引入非线性
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层1 → 隐藏层2
            nn.ReLU(),                          # ReLU 激活函数
            nn.Linear(hidden_dim, action_dim)   # 隐藏层2 → 输出层 (Q 值)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 状态张量，形状可以是 (batch_size, state_dim) 或 (state_dim,)
        Returns:
            每个动作的 Q 值，形状 (batch_size, action_dim)
        """
        return self.net(x)


# ==================== 2. 经验回放池 ====================
class ReplayBuffer:
    """
    经验回放池 (Experience Replay)
    作用：
    1. 打破数据之间的时序相关性，稳定训练
    2. 提高样本利用率，可以重复使用历史经验
    3. 减少更新方差
    """
    def __init__(self, capacity=100000):
        """
        Args:
            capacity: 缓冲区最大容量，超出时自动删除最旧的经验
        """
        # deque 是双端队列，maxlen 限制最大长度
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一个经验元组
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止 (episode 结束)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批经验
        Args:
            batch_size: 批量大小
        Returns:
            分别返回 states, actions, rewards, next_states, dones 的 numpy 数组
        """
        batch = random.sample(self.buffer, batch_size)
        # zip(*batch) 将 [(s1,a1,r1,s1',d1), ...] 转置为 (s列表, a列表, ...)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer)


# ==================== 3. DQN Agent ====================
class DQNAgent:
    """
    DQN 智能体
    包含：
    1. Q 网络 (q_net): 用于选择动作和计算当前 Q 值
    2. 目标网络 (target_net): 用于计算稳定的目标 Q 值
    3. 经验回放池: 存储和采样经验
    4. ε-贪婪策略: 平衡探索和利用
    """
    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
                 buffer_capacity=100000, batch_size=64,
                 target_update_freq=100):
        """
        初始化 DQN Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率 (learning rate)
            gamma: 折扣因子，控制未来奖励的重要性
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减速度
            buffer_capacity: 经验池容量
            batch_size: 批大小
            target_update_freq: 目标网络更新频率 (步数)
        """
        self.action_dim = action_dim
        self.gamma = gamma                    # 折扣因子
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0                   # 全局步数计数器
        
        # ε-贪婪参数：用于探索-利用平衡
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start          # 当前探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 创建 Q 网络和目标网络
        # Q 网络：用于动作选择和训练更新
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim=256)
        # 目标网络：用于计算稳定的目标 Q 值，参数更新较慢
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim=256)
        # 初始化时目标网络参数与 Q 网络相同
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # 优化器：使用 Adam 优化算法
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 经验回放池
        self.memory = ReplayBuffer(buffer_capacity)
    
    def act(self, state, eval_mode=False):
        """
        根据当前状态选择动作 (ε-贪婪策略)
        
        Args:
            state: 当前状态
            eval_mode: 是否评估模式 (True: 不探索，直接选择最优动作)
        Returns:
            选择的动作 (0 或 1)
        """
        # 评估模式或随机数大于 epsilon 时，利用网络选择动作
        if not eval_mode and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.action_dim - 1)
        
        # 利用：使用 Q 网络选择最优动作
        with torch.no_grad():  # 不计算梯度，提高效率
            # 将状态转换为张量并增加 batch 维度
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # 前向传播得到 Q 值
            q_values = self.q_net(state_tensor)
            # 返回最大 Q 值对应的动作索引
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """
        更新探索率 epsilon (指数衰减)
        公式: ε = ε_min + (ε_max - ε_min) * exp(-step / decay)
        
        训练初期 epsilon 较大 → 更多探索
        训练后期 epsilon 较小 → 更多利用
        """
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-self.step_count / self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """
        核心学习函数：从经验池采样，计算损失，更新 Q 网络
        
        DQN 损失函数:
        L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))^2]
        """
        # 经验池不够一个 batch 时，不更新
        if len(self.memory) < self.batch_size:
            return None
        
        # 1. 从经验池随机采样一个 batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 2. 转换为 PyTorch 张量
        states = torch.FloatTensor(states)           # (batch_size, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1)  # (batch_size, 1)
        rewards = torch.FloatTensor(rewards)         # (batch_size,)
        next_states = torch.FloatTensor(next_states) # (batch_size, state_dim)
        dones = torch.FloatTensor(dones)             # (batch_size,)
        
        # 3. 计算当前 Q 值 Q(s, a)
        # q_net(states) 输出: (batch_size, action_dim)
        # gather(1, actions) 提取每个样本对应动作的 Q 值
        # squeeze(1) 去掉多余维度，得到 (batch_size,)
        current_q = self.q_net(states).gather(1, actions).squeeze(1)
        
        # 4. 计算目标 Q 值 y = r + γ * max_a' Q_target(s', a') * (1 - done)
        # 使用 torch.no_grad() 确保目标网络不参与梯度计算
        with torch.no_grad():
            # 获取下一状态的最大 Q 值
            # max(dim=1) 返回 (values, indices)，[0] 取 values
            next_q = self.target_net(next_states).max(dim=1)[0]  # (batch_size,)
            
            # 目标 Q 值
            # (1 - dones): 如果 done=1 (终止状态)，则下一状态没有奖励
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 5. 计算损失 (均方误差)
        loss = F.mse_loss(current_q, target_q)
        
        # 6. 反向传播更新 Q 网络
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()              # 计算梯度
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()        # 更新参数
        
        # 7. 更新目标网络 (硬更新)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            # 将 Q 网络的参数复制到目标网络
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # 8. 衰减探索率 epsilon
        self.update_epsilon()
        
        return loss.item()
    
    def save(self, path):
        """保存模型参数"""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        print(f"模型已从 {path} 加载")


# ==================== 4. 训练 DQN ====================
def train_dqn(env_name='CartPole-v1', num_episodes=500, render=False):
    """
    训练 DQN 智能体
    
    Args:
        env_name: 环境名称
        num_episodes: 训练的 episode 数量
        render: 是否实时渲染画面
    Returns:
        agent: 训练好的智能体
        episode_rewards: 每个 episode 的总奖励
        episode_lengths: 每个 episode 的步数
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2
    
    # 创建 DQN 智能体
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []  # 记录每个 episode 的总奖励
    episode_lengths = []  # 记录每个 episode 的步数
    
    print(f"\n开始训练 DQN on {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print("-" * 60)
    
    # 训练循环
    for episode in range(num_episodes):
        state, _ = env.reset()  # 重置环境，获取初始状态
        total_reward = 0
        length = 0
        done = False
        
        # 一个 episode 的交互循环
        while not done:
            # 根据当前状态选择动作
            action = agent.act(state)
            
            # 执行动作，获取结果
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # episode 是否结束
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 学习更新 (从经验池采样并更新网络)
            agent.learn()
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            length += 1
        
        # 记录本 episode 的结果
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        
        # 每 50 个 episode 打印一次统计信息
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {length}")
    
    env.close()
    return agent, episode_rewards, episode_lengths


# ==================== 5. 测试 DQN ====================
def test_dqn(agent, env_name='CartPole-v1', num_episodes=10, render=True):
    """
    测试训练好的 DQN 智能体 (无探索，纯利用)
    
    Args:
        agent: 训练好的智能体
        env_name: 环境名称
        num_episodes: 测试的 episode 数量
        render: 是否显示画面
    Returns:
        rewards: 每个 episode 的奖励列表
        lengths: 每个 episode 的步数列表
    """
    # 创建环境，如果 render=True 则显示画面
    env = gym.make(env_name, render_mode='human' if render else None)
    rewards = []
    lengths = []
    
    print(f"\n开始测试 DQN on {env_name}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        length = 0
        done = False
        
        while not done:
            # 评估模式: eval_mode=True，不进行探索
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            length += 1
        
        rewards.append(total_reward)
        lengths.append(length)
        print(f"Test Episode {episode+1:2d}: Reward = {total_reward:6.2f}, Steps = {length}")
    
    env.close()
    print("-" * 60)
    print(f"平均测试奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    return rewards, lengths


# ==================== 6. 绘图函数 ====================
import matplotlib
matplotlib.use('TkAgg')  # 设置后端，避免交互模式问题

def setup_chinese_font():
    """配置 matplotlib 支持中文显示"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        font_name = 'SimHei'      # Windows 黑体
    elif system == 'Darwin':       # macOS
        font_name = 'Arial Unicode MS'
    else:                          # Linux
        font_name = 'WenQuanYi Zen Hei'
    
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试设置中文字体
try:
    setup_chinese_font()
    USE_CHINESE = True
except:
    USE_CHINESE = False
    print("Warning: Chinese font not available, using English labels")

def plot_results(episode_rewards, episode_lengths=None, save_path='dqn_training_results.png'):
    """
    绘制训练结果曲线
    
    Args:
        episode_rewards: 每个 episode 的奖励列表
        episode_lengths: 每个 episode 的步数列表 (可选)
        save_path: 图片保存路径
    """
    # 创建子图
    fig, axes = plt.subplots(1, 2 if episode_lengths else 1, figsize=(14, 5))
    
    if episode_lengths is None:
        axes = [axes]
    
    # 根据字体可用性选择语言
    if USE_CHINESE:
        reward_title = '训练奖励曲线'
        steps_title = '每回合步数曲线'
        reward_label = '原始奖励'
        steps_label = '原始步数'
        smooth_label = '平滑曲线'
        x_label = '回合数'
        y_reward = '总奖励'
        y_steps = '步数'
    else:
        reward_title = 'Training Reward Curve'
        steps_title = 'Steps per Episode Curve'
        reward_label = 'Raw Reward'
        steps_label = 'Raw Steps'
        smooth_label = 'Moving Average'
        x_label = 'Episode'
        y_reward = 'Total Reward'
        y_steps = 'Steps'
    
    # 绘制奖励曲线
    ax = axes[0]
    ax.plot(episode_rewards, alpha=0.4, label=reward_label, color='blue', linewidth=1)
    
    # 计算滑动平均 (平滑曲线)
    window = min(21, len(episode_rewards) // 5)
    if window >= 3:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='same')
        smoothed[:window//2] = episode_rewards[:window//2]
        smoothed[-(window//2):] = episode_rewards[-(window//2):]
        ax.plot(smoothed, 'r-', linewidth=2, label=f'{smooth_label} (window={window})')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_reward)
    ax.set_title(reward_title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 绘制步数曲线
    if episode_lengths and len(episode_lengths) > 0:
        ax = axes[1]
        ax.plot(episode_lengths, alpha=0.4, label=steps_label, color='green', linewidth=1)
        
        if window >= 3:
            smoothed_len = np.convolve(episode_lengths, np.ones(window)/window, mode='same')
            smoothed_len[:window//2] = episode_lengths[:window//2]
            smoothed_len[-(window//2):] = episode_lengths[-(window//2):]
            ax.plot(smoothed_len, 'orange', linewidth=2, label=f'{smooth_label} (window={window})')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_steps)
        ax.set_title(steps_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Figure saved to {save_path}")


# ==================== 7. 主程序入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("DQN Training Example - CartPole-v1")
    print("=" * 60)
    
    # 1. 训练 DQN 智能体
    # num_episodes=300: 训练 300 个 episode
    agent, rewards, lengths = train_dqn(
        env_name='CartPole-v1',
        num_episodes=300
    )
    
    # 2. 打印训练统计信息
    print("\n" + "-" * 60)
    print("Training Statistics:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Last 100 episodes avg reward: {np.mean(rewards[-100:]):.2f}")
    print(f"  Best episode reward: {np.max(rewards):.2f}")
    print(f"  Worst episode reward: {np.min(rewards):.2f}")
    
    # 3. 绘制训练曲线
    plot_results(rewards, lengths, save_path='dqn_training_results.png')
    
    # 4. 保存训练好的模型
    agent.save('dqn_cartpole.pth')
    
    # 5. 测试训练好的智能体 (显示画面)
    test_rewards, test_lengths = test_dqn(
        agent=agent,
        env_name='CartPole-v1',
        num_episodes=10,
        render=True
    )
    
    # 6. 打印测试结果
    print("\n" + "-" * 60)
    print(f"Test Results - Avg Reward: {np.mean(test_rewards):.2f} +/- {np.std(test_rewards):.2f}")