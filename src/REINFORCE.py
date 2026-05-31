"""
REINFORCE算法核心公式：

1. 策略梯度定理：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * Q^π(s,a)]

2. REINFORCE更新公式（使用蒙特卡洛回报）：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * G_t]
   其中 G_t = Σ_{k=t}^{T} γ^{k-t} r_k

3. 梯度上升更新：
   θ ← θ + α * ∇_θ J(θ)

4. 损失函数（最小化）：
   L(θ) = -E[log π_θ(a|s) * G_t]

5. 带基线的REINFORCE：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * (G_t - b(s))]
   b(s) 通常是状态价值函数 V(s)
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
# ==================== 1. 策略网络 ====================

class PolicyNetwork(nn.Module):
    """策略网络 - 输出动作概率分布"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 输出概率分布
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """前向传播，输出动作概率"""
        return self.network(x)
    
    def get_action(self, state, deterministic=False):
        """
        根据策略选择动作
        
        Args:
            state: 状态张量
            deterministic: 是否确定性选择（测试时使用）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            entropy: 策略的熵（用于探索）
        """
        probs = self.forward(state)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class ValueNetwork(nn.Module):
    """价值网络 - 用作基线（Baseline）"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """前向传播，输出状态价值"""
        return self.network(x).squeeze(-1)


# ==================== 2. 基础REINFORCE算法 ====================

class REINFORCE:
    """
    基础REINFORCE算法
    
    更新公式：θ ← θ + α * ∇_θ log π_θ(a|s) * G_t
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.01,
                 gamma=0.99,
                 hidden_dim=128):
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # 存储episode数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': []
        }
    
    def select_action(self, state):
        """
        选择动作
        
        Args:
            state: 当前状态（numpy数组）
        
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state_tensor)
        
        # 存储轨迹
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        
        return action.item()
    
    def store_reward(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
    
    def finish_episode(self):
        """
        在一个episode结束后更新策略
        
        计算每个时间步的折扣回报G_t，然后更新策略
        """
        # 计算折扣回报
        returns = []
        G = 0
        
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # 标准化回报（降低方差，提高稳定性）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略损失
        policy_loss = 0
        for log_prob, G_t in zip(self.log_probs, returns):
            policy_loss -= log_prob * G_t
        
        policy_loss = policy_loss / len(self.rewards)
        
        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 清空轨迹
        self.clear_memory()
        
        return policy_loss.item()
    
    def clear_memory(self):
        """清空episode轨迹"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def train(self, env, num_episodes=1000, max_steps=500, render=False):
        """
        训练REINFORCE算法
        
        Args:
            env: Gym环境
            num_episodes: 训练的episode数量
            max_steps: 每个episode的最大步数
            render: 是否渲染环境
        """
        print("\n" + "=" * 70)
        print("REINFORCE算法训练")
        print("=" * 70)
        print(f"状态维度: {env.observation_space.shape[0]}")
        print(f"动作维度: {env.action_space.n}")
        print(f"学习率: {self.optimizer.param_groups[0]['lr']}")
        print(f"折扣因子 γ: {self.gamma}")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # 可选渲染
                if render:
                    env.render()
                
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存储奖励
                self.store_reward(reward)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Episode结束，更新策略
            loss = self.finish_episode()
            
            # 记录历史
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['losses'].append(loss)
            
            # 打印进度
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                avg_length = np.mean(self.training_history['episode_lengths'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:6.1f} | "
                      f"Loss: {loss:.4f}")
        
        if render:
            env.close()
        
        return self.policy


# ==================== 3. REINFORCE with Baseline ====================

class REINFORCEWithBaseline:
    """
    REINFORCE with Baseline算法
    
    更新公式：θ ← θ + α * ∇_θ log π_θ(a|s) * (G_t - V(s))
    
    使用价值网络作为基线，降低方差
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 policy_lr=0.01,
                 value_lr=0.01,
                 gamma=0.99,
                 hidden_dim=128):
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        self.gamma = gamma
        
        # 存储episode数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state_tensor)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        
        return action.item()
    
    def store_reward(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
    
    def finish_episode(self):
        """Episode结束时更新"""
        # 计算折扣回报
        returns = []
        G = 0
        
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # 计算状态价值（作为基线）
        states_tensor = torch.FloatTensor(np.array(self.states))
        values = self.value_net(states_tensor).detach()
        
        # 计算优势函数 A = G_t - V(s)
        advantages = returns - values
        
        # 标准化优势（降低方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 策略损失
        policy_loss = 0
        for log_prob, adv in zip(self.log_probs, advantages):
            policy_loss -= log_prob * adv
        policy_loss = policy_loss / len(self.rewards)
        
        # 价值损失（MSE）
        value_loss = nn.MSELoss()(self.value_net(states_tensor), returns)
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        
        # 清空轨迹
        self.clear_memory()
        
        return policy_loss.item(), value_loss.item()
    
    def clear_memory(self):
        """清空轨迹"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def train(self, env, num_episodes=1000, max_steps=500):
        """训练REINFORCE with Baseline"""
        print("\n" + "=" * 70)
        print("REINFORCE with Baseline算法训练")
        print("=" * 70)
        print(f"状态维度: {env.observation_space.shape[0]}")
        print(f"动作维度: {env.action_space.n}")
        print(f"策略学习率: {self.policy_optimizer.param_groups[0]['lr']}")
        print(f"价值学习率: {self.value_optimizer.param_groups[0]['lr']}")
        print(f"折扣因子 γ: {self.gamma}")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.store_reward(reward)
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            policy_loss, value_loss = self.finish_episode()
            
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['policy_losses'].append(policy_loss)
            self.training_history['value_losses'].append(value_loss)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Policy Loss: {policy_loss:.4f} | "
                      f"Value Loss: {value_loss:.4f}")
        
        return self.policy


# ==================== 4. 可视化函数 ====================

def plot_reinforce_results(agent, title="REINFORCE Training Results"):
    """绘制REINFORCE训练结果"""
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
    ax1.set_title(f'{title} - Reward Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 步数曲线
    ax2 = axes[0, 1]
    lengths = history['episode_lengths']
    ax2.plot(lengths, alpha=0.4, label='Steps per Episode', color='green')
    
    if len(lengths) >= window:
        smoothed_steps = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(lengths)), smoothed_steps,
                'orange', linewidth=2, label=f'Moving Average')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失曲线
    if 'losses' in history:
        ax3 = axes[1, 0]
        losses = history['losses']
        if len(losses) >= window:
            smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax3.plot(smoothed_loss, label='Loss', color='red')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    elif 'policy_losses' in history:
        ax3 = axes[1, 0]
        policy_losses = history['policy_losses']
        value_losses = history['value_losses']
        
        if len(policy_losses) >= window:
            policy_smoothed = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
            value_smoothed = np.convolve(value_losses, np.ones(window)/window, mode='valid')
            ax3.plot(policy_smoothed, label='Policy Loss', color='red')
            ax3.plot(value_smoothed, label='Value Loss', color='blue')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Losses')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 成功率曲线
    ax4 = axes[1, 1]
    if len(rewards) >= window:
        # 假设CartPole成功的阈值是195
        success_threshold = 195
        success_rate = []
        for i in range(window, len(rewards) + 1):
            rate = np.sum(np.array(rewards[i-window:i]) >= success_threshold) / window
            success_rate.append(rate)
        ax4.plot(range(window, len(rewards) + 1), success_rate, 'g-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate')
        ax4.set_title(f'Success Rate (threshold={success_threshold})')
        ax4.axhline(y=0.8, color='r', linestyle='--', label='80% target')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 5. REINFORCE算法演示 ====================

def demonstrate_reinforce():
    """演示REINFORCE算法的核心计算"""
    print("\n" + "=" * 70)
    print("REINFORCE算法核心计算演示")
    print("=" * 70)
    
    # 模拟数据
    print("\n1. 策略梯度计算:")
    print("-" * 40)
    
    # 模拟策略网络输出
    logits = np.array([1.0, 2.0, 0.5])
    probs = np.exp(logits) / np.sum(np.exp(logits))
    action_taken = 1
    
    print(f"  策略输出概率: {probs}")
    print(f"  实际采取的动作: {action_taken}")
    
    # 计算对数概率
    log_prob = np.log(probs[action_taken])
    print(f"  log π(a|s) = ln({probs[action_taken]:.3f}) = {log_prob:.4f}")
    
    # 模拟episode回报
    episode_rewards = [1, 0, 1, 1, 0]
    gamma = 0.9
    
    print(f"\n2. 折扣回报G_t计算:")
    print("-" * 40)
    print(f"  Episode奖励序列: {episode_rewards}")
    print(f"  折扣因子 γ = {gamma}")
    
    returns = []
    G = 0
    for t, reward in enumerate(reversed(episode_rewards)):
        G = reward + gamma * G
        returns.insert(0, G)
        print(f"  G_{len(episode_rewards)-t-1} = {G:.3f}")
    
    print(f"\n  折扣回报: {[f'{r:.3f}' for r in returns]}")
    
    print("\n3. 策略梯度更新:")
    print("-" * 40)
    
    log_prob_value = -0.5
    G_t = 2.5
    
    gradient = log_prob_value * G_t
    print(f"  ∇_θ J = log_prob * G_t = {log_prob_value} * {G_t} = {gradient:.3f}")
    
    learning_rate = 0.01
    param_update = learning_rate * gradient
    print(f"  参数更新 = α * ∇_θ J = {learning_rate} * {gradient:.3f} = {param_update:.4f}")
    
    print("\n4. 损失函数:")
    print("-" * 40)
    
    loss = -log_prob_value * G_t
    print(f"  L(θ) = -log_prob * G_t = -({log_prob_value}) * {G_t} = {loss:.3f}")
    
    print("\n5. 梯度上升 vs 梯度下降:")
    print("-" * 40)
    print("  目标: 最大化 J(θ) (期望回报)")
    print("  实现: 最小化 L(θ) = -J(θ)")
    print(f"  梯度上升: θ ← θ + α * ∇J")
    print(f"  梯度下降: θ ← θ - α * ∇L")
    print("  两者等价: ∇L = -∇J")


# ==================== 6. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("REINFORCE算法完整实现")
    print("=" * 70)
    
    # 核心公式说明
    print("\n" + "-" * 70)
    print("核心公式:")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. 策略梯度定理:                                               │
    │     ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * Q^π(s,a)]               │
    │                                                                 │
    │  2. REINFORCE更新:                                              │
    │     θ ← θ + α * ∇_θ log π_θ(a|s) * G_t                        │
    │                                                                 │
    │  3. 折扣回报:                                                   │
    │     G_t = r_{t+1} + γr_{t+2} + γ^2r_{t+3} + ...               │
    │                                                                 │
    │  4. 损失函数:                                                   │
    │     L(θ) = -E[log π_θ(a|s) * G_t]                             │
    │                                                                 │
    │  5. REINFORCE with Baseline:                                   │
    │     θ ← θ + α * ∇_θ log π_θ(a|s) * (G_t - V(s))               │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 演示核心概念
    demonstrate_reinforce()
    
    # 创建环境
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 选择算法
    use_baseline = True  # 切换是否使用baseline
    
    if use_baseline:
        print("\n使用 REINFORCE with Baseline 算法")
        agent = REINFORCEWithBaseline(
            state_dim, action_dim,
            policy_lr=0.01,
            value_lr=0.01,
            gamma=0.99,
            hidden_dim=128
        )
    else:
        print("\n使用基础 REINFORCE 算法")
        agent = REINFORCE(
            state_dim, action_dim,
            learning_rate=0.01,
            gamma=0.99,
            hidden_dim=128
        )
    
    # 训练
    agent.train(env, num_episodes=500, max_steps=500)
    
    # 绘制结果
    plot_reinforce_results(agent, "REINFORCE" + (" with Baseline" if use_baseline else ""))
    
    # 测试训练好的策略
    print("\n" + "=" * 70)
    print("测试训练好的策略")
    print("=" * 70)
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(5):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.policy.get_action(state_tensor, deterministic=True)
            
            state, reward, terminated, truncated, _ = test_env.step(action.item())
            done = terminated or truncated
            total_reward += reward
        
        print(f"Test Episode {episode+1}: Reward = {total_reward:.0f}")
    
    test_env.close()
    
    print("\n" + "=" * 70)
    print("REINFORCE算法总结")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  REINFORCE算法特点:                                             │
    │  - 蒙特卡洛方法: 需要完整episode                                │
    │  - 无偏估计: 梯度估计是真实梯度的无偏估计                       │
    │  - 高方差: 由于使用蒙特卡洛采样，方差较大                       │
    │  - 收敛性: 理论上保证收敛到局部最优                             │
    │                                                                 │
    │  改进方法:                                                      │
    │  - 使用Baseline: 降低方差但不引入偏差                          │
    │  - 回报标准化: 提高数值稳定性                                  │
    │  - 梯度裁剪: 防止梯度爆炸                                      │
    │                                                                 │
    │  适用场景:                                                      │
    │  - 回合制任务（有明确的终止状态）                               │
    │  - 动作空间连续或离散均可                                      │
    │  - 需要完整轨迹才能评估的环境                                  │
    └─────────────────────────────────────────────────────────────────┘
    """)