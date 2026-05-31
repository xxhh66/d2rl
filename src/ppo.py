"""
PPO算法核心公式：

1. 重要性采样比率：
   r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

2. Clipped Surrogate目标函数：
   L^CLIP(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

3. GAE优势估计：
   A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
   δ_t = r_t + γV(s_{t+1}) - V(s_t)

4. 价值函数损失：
   L^VF(θ) = (V_θ(s_t) - V_target)^2

5. 熵正则化：
   S(π_θ)(s_t) = -Σ_a π_θ(a|s_t) log π_θ(a|s_t)

6. 总损失：
   L_total = L^CLIP + c1 * L^VF - c2 * S

Actor-Critic (基础框架)->A2C / A3C (同步/异步版本)->
TRPO (信任区域策略优化)->PPO (近端策略优化) ← 简化版TRPO   
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 1. 神经网络模型 ====================

class PolicyNetwork(nn.Module):
    """策略网络（Actor）"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        
        # 返回动作概率分布
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    def get_action(self, state, action=None):
        """选择动作并计算对数概率和熵"""
        probs = self.forward(state)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class ValueNetwork(nn.Module):
    """价值网络（Critic）"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# ==================== 2. PPO Agent ====================

class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent
    
    核心思想：
    1. 使用重要性采样重用旧策略的数据
    2. 通过裁剪限制策略更新幅度
    3. 多轮更新提高样本效率
    """
    
    def __init__(self, 
                 state_dim,
                 action_dim,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,      # GAE参数
                 clip_epsilon=0.2,      # 裁剪范围 ε
                 ppo_epochs=10,         # 每批数据更新次数
                 batch_size=64,
                 entropy_coef=0.01,     # 熵正则化系数 c2
                 value_coef=0.5,        # 价值损失系数 c1
                 max_grad_norm=0.5,     # 梯度裁剪
                 device='cpu'):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # 创建网络
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # 存储轨迹数据
        self.clear_memory()
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
    
    def clear_memory(self):
        """清空轨迹缓存"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state):
        """
        选择动作（用于环境交互）
        
        Args:
            state: 当前状态
            
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy = self.policy_net.get_action(state_tensor)
            value = self.value_net(state_tensor)
        
        # 存储轨迹
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_transition(self, reward, done):
        """存储转移"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae_advantages(self, rewards, values, dones, next_value):
        """
        计算GAE (Generalized Advantage Estimation)
        
        公式：A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        gae = 0
        
        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD误差
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # 优势归一化（提高稳定性）
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self, next_state, done):
        """
        更新PPO策略
        
        PPO-Clip目标函数：
        L^CLIP(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        """
        # 计算下一个状态的价值
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.value_net(next_state_tensor).item()
        
        # 转换数据为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        
        # 计算优势函数
        advantages = self.compute_gae_advantages(self.rewards, self.values, self.dones, next_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 计算回报（价值目标）
        returns = advantages + values
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        for epoch in range(self.ppo_epochs):
            # 随机采样小批量
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 1. 策略损失 (PPO-Clip)
                probs = self.policy_net(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 重要性采样比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 裁剪的目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                   1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. 价值损失
                new_values = self.value_net(batch_states).squeeze()
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # 3. 总损失（加入熵正则化）
                loss = (policy_loss + 
                       self.value_coef * value_loss - 
                       self.entropy_coef * entropy)
                
                # 反向传播 - 策略网络
                self.policy_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # 反向传播 - 价值网络
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        # 清空轨迹缓存
        self.clear_memory()
        
        return (total_policy_loss / update_count,
                total_value_loss / update_count,
                total_entropy / update_count)
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        print(f"模型已从 {path} 加载")


# ==================== 3. PPO训练函数 ====================

def train_ppo(env_name='CartPole-v1', 
              num_episodes=1000,
              render=False,
              **agent_kwargs):
    """
    训练PPO Agent
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建Agent
    agent = PPOAgent(state_dim, action_dim, **agent_kwargs)
    
    print(f"\n开始训练 PPO on {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"超参数: γ={agent.gamma}, λ={agent.gae_lambda}, ε={agent.clip_epsilon}")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储转移
            agent.store_transition(reward, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # 如果回合结束，更新策略
            if done:
                policy_loss, value_loss, entropy = agent.update(next_state, done)
                agent.training_history['policy_losses'].append(policy_loss)
                agent.training_history['value_losses'].append(value_loss)
                agent.training_history['entropies'].append(entropy)
                break
        
        # 记录结果
        agent.training_history['episode_rewards'].append(episode_reward)
        agent.training_history['episode_lengths'].append(episode_length)
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(agent.training_history['episode_rewards'][-50:])
            avg_length = np.mean(agent.training_history['episode_lengths'][-50:])
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Policy Loss: {policy_loss:.4f} | "
                  f"Value Loss: {value_loss:.4f}")
    
    env.close()
    return agent


# ==================== 4. PPO测试函数 ====================

def test_ppo(agent, env_name='CartPole-v1', num_episodes=10, render=True):
    """测试训练好的PPO Agent"""
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    print(f"\n开始测试 PPO on {env_name}")
    print("-" * 70)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _, _ = agent.policy_net.get_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step_count += 1
        
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode+1:2d}: Reward = {episode_reward:.0f}, Steps = {step_count}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    print("-" * 70)
    print(f"平均奖励: {avg_reward:.2f}")
    
    return avg_reward


# ==================== 5. 可视化函数 ====================

def plot_training_results(agent):
    """绘制训练结果"""
    history = agent.training_history
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 奖励曲线
    ax1 = axes[0, 0]
    rewards = history['episode_rewards']
    ax1.plot(rewards, alpha=0.4, label='Raw Reward', color='blue')
    
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training Reward Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 步数曲线
    ax2 = axes[0, 1]
    lengths = history['episode_lengths']
    ax2.plot(lengths, alpha=0.4, label='Steps per Episode', color='green')
    
    if len(lengths) >= window:
        smoothed_steps = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(lengths)), smoothed_steps,
                'orange', linewidth=2, label=f'Moving Average (window={window})')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('PPO Training Steps Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失曲线
    ax3 = axes[1, 0]
    policy_losses = history['policy_losses']
    value_losses = history['value_losses']
    
    # 滑动平均
    if len(policy_losses) >= window:
        policy_smoothed = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
        value_smoothed = np.convolve(value_losses, np.ones(window)/window, mode='valid')
        ax3.plot(policy_smoothed, label='Policy Loss', color='red')
        ax3.plot(value_smoothed, label='Value Loss', color='blue')
    
    ax3.set_xlabel('Update Step')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 熵曲线
    ax4 = axes[1, 1]
    entropies = history['entropies']
    
    if len(entropies) >= window:
        entropy_smoothed = np.convolve(entropies, np.ones(window)/window, mode='valid')
        ax4.plot(entropy_smoothed, label='Entropy', color='purple')
    
    ax4.set_xlabel('Update Step')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy (Exploration)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 6. PPO超参数对比 ====================

def compare_ppo_hyperparameters():
    """比较不同超参数对PPO性能的影响"""
    print("\n" + "=" * 70)
    print("PPO超参数对比实验")
    print("=" * 70)
    
    env_name = 'CartPole-v1'
    num_episodes = 300
    
    configs = [
        {'name': 'Standard', 'clip_epsilon': 0.2, 'ppo_epochs': 10},
        {'name': 'Small Clip', 'clip_epsilon': 0.1, 'ppo_epochs': 10},
        {'name': 'Large Clip', 'clip_epsilon': 0.3, 'ppo_epochs': 10},
        {'name': 'More Epochs', 'clip_epsilon': 0.2, 'ppo_epochs': 20},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n训练: {config['name']}")
        print(f"  clip_epsilon={config['clip_epsilon']}, ppo_epochs={config['ppo_epochs']}")
        
        agent = train_ppo(
            env_name=env_name,
            num_episodes=num_episodes,
            render=False,
            clip_epsilon=config['clip_epsilon'],
            ppo_epochs=config['ppo_epochs'],
            verbose=False
        )
        
        results[config['name']] = agent.training_history['episode_rewards']
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    
    for name, rewards in results.items():
        window = 20
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=name, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (window=20)')
    plt.title('PPO超参数对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 7. PPO算法演示 ====================

def demonstrate_ppo():
    """演示PPO算法的核心步骤"""
    print("\n" + "=" * 70)
    print("PPO算法核心步骤演示")
    print("=" * 70)
    
    # 模拟数据
    print("\n1. 重要性采样比率计算:")
    print("-" * 40)
    
    old_probs = [0.3, 0.5, 0.2]
    new_probs = [0.4, 0.4, 0.2]
    action = 1
    
    old_log_prob = np.log(old_probs[action])
    new_log_prob = np.log(new_probs[action])
    ratio = np.exp(new_log_prob - old_log_prob)
    
    print(f"  旧策略概率: {old_probs}")
    print(f"  新策略概率: {new_probs}")
    print(f"  采取动作 {action}")
    print(f"  旧log概率: {old_log_prob:.3f}")
    print(f"  新log概率: {new_log_prob:.3f}")
    print(f"  重要性比率 r(θ) = {ratio:.3f}")
    
    print("\n2. PPO-Clip目标函数:")
    print("-" * 40)
    
    advantage = 0.5
    epsilon = 0.2
    
    surr1 = ratio * advantage
    surr2 = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
    loss = -min(surr1, surr2)
    
    print(f"  Advantage A = {advantage}")
    print(f"  r(θ) = {ratio:.3f}")
    print(f"  r(θ)A = {surr1:.3f}")
    print(f"  clip(r(θ), {1-epsilon}, {1+epsilon})A = {surr2:.3f}")
    print(f"  min = {min(surr1, surr2):.3f}")
    print(f"  Loss = -min = {loss:.3f}")
    
    print("\n3. GAE优势估计:")
    print("-" * 40)
    
    rewards = [1, 0, 1, 1]
    values = [0.5, 0.6, 0.7, 0.8]
    gamma = 0.9
    gae_lambda = 0.95
    
    print(f"  rewards: {rewards}")
    print(f"  values: {values}")
    print(f"  γ={gamma}, λ={gae_lambda}")
    
    gae = 0
    for t in range(len(rewards)-1, -1, -1):
        if t == len(rewards) - 1:
            next_val = 0
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        print(f"  t={t}: δ={delta:.3f}, GAE={gae:.3f}")


# ==================== 8. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("PPO (Proximal Policy Optimization) Algorithm Implementation")
    print("=" * 70)
    
    # PPO参数说明
    print("\n" + "-" * 70)
    print("PPO 核心参数说明:")
    print("-" * 70)
    print("""
    ┌─────────────────┬──────────┬────────────────────────────────┐
    │     参数        │   值     │            说明                 │
    ├─────────────────┼──────────┼────────────────────────────────┤
    │ clip_epsilon    │   0.2    │ 裁剪范围，限制策略更新幅度      │
    │ gamma           │   0.99   │ 折扣因子                       │
    │ gae_lambda      │   0.95   │ GAE参数，平衡偏差与方差        │
    │ ppo_epochs      │   10     │ 每批数据的更新次数             │
    │ entropy_coef    │   0.01   │ 熵正则化系数，鼓励探索         │
    │ value_coef      │   0.5    │ 价值损失系数                   │
    │ lr_actor        │   3e-4   │ 策略网络学习率                 │
    │ lr_critic       │   1e-3   │ 价值网络学习率                 │
    └─────────────────┴──────────┴────────────────────────────────┘
    """)
    
    # 演示核心概念
    demonstrate_ppo()
    
    # 训练PPO
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    agent = train_ppo(
        env_name='CartPole-v1',
        num_episodes=500,
        render=False,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=10,
        batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 打印训练统计
    print("\n" + "-" * 70)
    print("Training Statistics:")
    print(f"  Total episodes: {len(agent.training_history['episode_rewards'])}")
    print(f"  Last 100 episodes avg reward: {np.mean(agent.training_history['episode_rewards'][-100:]):.2f}")
    print(f"  Max reward: {np.max(agent.training_history['episode_rewards']):.0f}")
    
    # 绘制训练曲线
    plot_training_results(agent)
    
    # 保存模型
    # agent.save_model('ppo_cartpole.pth')
    
    # 测试智能体
    test_ppo(agent, num_episodes=5)