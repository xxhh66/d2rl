"""
策略梯度算法核心公式：

1. 策略梯度定理：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * Q^π(s,a)]

2. REINFORCE算法（Monte Carlo策略梯度）：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * G_t]
   其中 G_t = Σ_{k=t}^{T} γ^{k-t} r_k

3. REINFORCE with Baseline：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * (G_t - b(s))]
   b(s) 通常是状态价值函数 V(s)

4. Actor-Critic：
   ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) * A(s,a)]
   其中 A(s,a) = Q(s,a) - V(s) 是优势函数

5. 策略梯度更新（梯度上升）：
   θ ← θ + α * ∇_θ J(θ)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 1. 策略网络 ====================

class PolicyNetwork(nn.Module):
    """策略网络（Actor）"""
    def __init__(self, state_dim, action_dim, hidden_dim=64, discrete=True):
        super(PolicyNetwork, self).__init__()
        
        self.discrete = discrete
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if discrete:
            # 离散动作空间
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            # 连续动作空间 - 均值和标准差
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        features = self.shared_net(x)
        
        if self.discrete:
            logits = self.action_head(features)
            return torch.softmax(logits, dim=-1)
        else:
            mean = self.mean_head(features)
            std = torch.exp(self.log_std)
            return mean, std
    
    def get_action(self, state, deterministic=False):
        """选择动作并计算对数概率"""
        if self.discrete:
            probs = self.forward(state)
            dist = Categorical(probs)
            
            if deterministic:
                action = torch.argmax(probs)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return action, log_prob, entropy
        else:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            return action, log_prob, entropy


class ValueNetwork(nn.Module):
    """价值网络（Critic）- 用作baseline"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==================== 2. REINFORCE算法 ====================

class REINFORCE:
    """
    REINFORCE算法 (Monte Carlo Policy Gradient)
    
    更新公式：∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
    """
    
    def __init__(self, 
                 state_dim,
                 action_dim,
                 lr=0.01,
                 gamma=0.99,
                 hidden_dim=64):
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, discrete=True)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # 存储轨迹
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
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state_tensor)
        
        # 存储
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        
        return action.item()
    
    def store_transition(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
    
    def finish_episode(self):
        """Episode结束时计算梯度并更新"""
        # 计算折扣回报
        returns = []
        G = 0
        
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # 标准化回报（降低方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度
        policy_loss = 0
        for log_prob, G_t in zip(self.log_probs, returns):
            policy_loss -= log_prob * G_t
        
        policy_loss = policy_loss / len(self.rewards)
        
        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        return policy_loss.item()
    
    def train(self, env, num_episodes=1000, max_steps=200, render=False):
        """训练REINFORCE"""
        print("\nREINFORCE算法训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.store_transition(reward)
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # 更新策略
            loss = self.finish_episode()
            
            # 记录
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['losses'].append(loss)
            
            # 打印进度
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Loss: {loss:.4f}")
        
        return self.policy


# ==================== 3. REINFORCE with Baseline ====================

class REINFORCEWithBaseline:
    """
    REINFORCE with Baseline
    
    更新公式：∇_θ J(θ) = E[∇_θ log π_θ(a|s) * (G_t - V(s))]
    
    使用状态价值函数作为baseline降低方差
    """
    
    def __init__(self, 
                 state_dim,
                 action_dim,
                 lr_policy=0.01,
                 lr_value=0.01,
                 gamma=0.99,
                 hidden_dim=64):
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, discrete=True)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        self.gamma = gamma
        
        # 存储轨迹
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
    
    def store_transition(self, reward):
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
        
        # 计算状态价值作为baseline
        states_tensor = torch.FloatTensor(np.array(self.states))
        values = self.value_net(states_tensor).detach()
        
        # 优势函数 A = G_t - V(s)
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
        self.policy_optimizer.step()
        
        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        return policy_loss.item(), value_loss.item()
    
    def train(self, env, num_episodes=1000, max_steps=200):
        """训练REINFORCE with Baseline"""
        print("\nREINFORCE with Baseline训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.store_transition(reward)
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
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Policy Loss: {policy_loss:.4f}")
        
        return self.policy


# ==================== 4. Actor-Critic算法 ====================

class ActorCritic:
    """
    Actor-Critic算法
    
    更新公式：
    - Actor: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]
    - Critic: 更新价值函数 V(s)
    
    A(s,a) = r + γV(s') - V(s) (TD误差)
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor=0.001,
                 lr_critic=0.01,
                 gamma=0.99,
                 hidden_dim=64):
        
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, discrete=True)
        self.critic = ValueNetwork(state_dim, hidden_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': []
        }
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, state, action, reward, next_state, done, log_prob, value):
        """单步更新"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        with torch.no_grad():
            if done:
                td_target = reward
            else:
                next_value = self.critic(next_state_tensor)
                td_target = reward + self.gamma * next_value
            
            advantage = td_target - value
        
        # Actor损失
        action_tensor = torch.LongTensor([action])
        log_prob_tensor = torch.FloatTensor([log_prob])
        
        actor_loss = -(log_prob_tensor * advantage)
        
        # Critic损失
        value_tensor = torch.FloatTensor([value])
        critic_loss = nn.MSELoss()(value_tensor, torch.FloatTensor([td_target]))
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, env, num_episodes=1000, max_steps=200):
        """训练Actor-Critic"""
        print("\nActor-Critic训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actor_loss = 0
            episode_critic_loss = 0
            steps = 0
            
            for _ in range(max_steps):
                action, log_prob, value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                actor_loss, critic_loss = self.update(
                    state, action, reward, next_state, done, log_prob, value
                )
                
                episode_reward += reward
                episode_length += 1
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                steps += 1
                
                state = next_state
                
                if done:
                    break
            
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['actor_losses'].append(episode_actor_loss / steps)
            self.training_history['critic_losses'].append(episode_critic_loss / steps)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Length: {episode_length}")
        
        return self.actor, self.critic


# ==================== 5. 连续动作空间的策略梯度 ====================

class ContinuousPolicyGradient:
    """连续动作空间的策略梯度算法"""
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=0.01,
                 gamma=0.99,
                 hidden_dim=64):
        
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, discrete=False)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        
        self.rewards = []
        self.log_probs = []
        self.states = []
    
    def select_action(self, state):
        """选择连续动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor)
        
        self.states.append(state)
        self.log_probs.append(log_prob.item())
        
        return action.squeeze().numpy()
    
    def store_transition(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
    
    def finish_episode(self):
        """更新策略"""
        returns = []
        G = 0
        
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = 0
        for log_prob, G_t in zip(self.log_probs, returns):
            policy_loss -= log_prob * G_t
        
        policy_loss = policy_loss / len(self.rewards)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.log_probs = []
        self.states = []
        
        return policy_loss.item()


# ==================== 6. 可视化函数 ====================

def plot_training_results(agent, title="Training Results"):
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
                'r-', linewidth=2, label=f'Moving Average')
    
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
    
    # 3. 损失曲线（如果存在）
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
    
    # 4. 滑动平均奖励
    ax4 = axes[1, 1]
    if len(rewards) >= window:
        success_rate = []
        for i in range(window, len(rewards) + 1):
            rate = np.mean(rewards[i-window:i])
            success_rate.append(rate)
        ax4.plot(range(window, len(rewards) + 1), success_rate, 'g-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.set_title('Smoothed Reward (window=50)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 7. 主训练函数 ====================

def train_policy_gradient():
    """训练所有策略梯度算法并对比"""
    print("=" * 70)
    print("策略梯度算法完整实现")
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
    │  2. REINFORCE:                                                  │
    │     ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]                      │
    │                                                                 │
    │  3. REINFORCE with Baseline:                                   │
    │     ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * (G_t - V(s))]             │
    │                                                                 │
    │  4. Actor-Critic:                                               │
    │     ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]                   │
    │     其中 A(s,a) = r + γV(s') - V(s)                            │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 1. REINFORCE
    print("\n" + "=" * 60)
    reinforce = REINFORCE(state_dim, action_dim, lr=0.01, gamma=0.99)
    reinforce.train(env, num_episodes=500, max_steps=200)
    plot_training_results(reinforce, "REINFORCE")
    
    # 2. REINFORCE with Baseline
    print("\n" + "=" * 60)
    reinforce_baseline = REINFORCEWithBaseline(state_dim, action_dim, 
                                                lr_policy=0.01, lr_value=0.01, gamma=0.99)
    reinforce_baseline.train(env, num_episodes=500, max_steps=200)
    plot_training_results(reinforce_baseline, "REINFORCE with Baseline")
    
    # 3. Actor-Critic
    print("\n" + "=" * 60)
    actor_critic = ActorCritic(state_dim, action_dim, 
                                lr_actor=0.001, lr_critic=0.01, gamma=0.99)
    actor_critic.train(env, num_episodes=500, max_steps=200)
    plot_training_results(actor_critic, "Actor-Critic")
    
    env.close()


# ==================== 8. 策略梯度算法演示 ====================

def demonstrate_policy_gradient():
    """演示策略梯度的核心计算"""
    print("\n" + "=" * 70)
    print("策略梯度计算演示")
    print("=" * 70)
    
    # 模拟数据
    print("\n1. 策略梯度计算:")
    print("-" * 40)
    
    # 模拟策略输出
    logits = np.array([1.0, 2.0, 0.5])
    probs = np.exp(logits) / np.sum(np.exp(logits))
    action = 1  # 选择的动作
    
    print(f"  策略输出 logits: {logits}")
    print(f"  策略概率: {probs}")
    print(f"  选择的动作: {action}")
    
    # 计算对数概率
    log_prob = np.log(probs[action])
    print(f"  log π(a|s) = {log_prob:.4f}")
    
    # 模拟回报
    G = 2.5
    print(f"  回报 G_t = {G}")
    
    # 策略梯度
    policy_gradient = log_prob * G
    print(f"  策略梯度 = log_prob * G = {policy_gradient:.4f}")
    
    print("\n2. 梯度上升更新:")
    print("-" * 40)
    
    lr = 0.01
    old_params = np.array([1.0, 2.0, 0.5])
    gradient = np.array([0.2, 0.5, 0.1])
    
    new_params = old_params + lr * gradient
    
    print(f"  旧参数: {old_params}")
    print(f"  梯度: {gradient}")
    print(f"  学习率: {lr}")
    print(f"  新参数: {new_params}")
    
    print("\n3. 优势函数计算:")
    print("-" * 40)
    
    reward = 1.0
    gamma = 0.9
    V_s = 0.5
    V_next = 0.8
    
    td_error = reward + gamma * V_next - V_s
    print(f"  TD误差 = r + γV(s') - V(s)")
    print(f"         = {reward} + {gamma}*{V_next} - {V_s}")
    print(f"         = {td_error:.3f}")


# ==================== 9. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("策略梯度算法 (Policy Gradient) 完整实现")
    print("=" * 70)
    
    # 演示核心概念
    demonstrate_policy_gradient()
    
    # 训练所有算法（可选，较耗时）
    # train_policy_gradient()
    
    # 单独训练REINFORCE
    print("\n" + "=" * 60)
    print("训练 REINFORCE 算法")
    print("=" * 60)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    reinforce = REINFORCE(state_dim, action_dim, lr=0.01, gamma=0.99)
    reinforce.train(env, num_episodes=300, max_steps=200)
    
    # 绘制结果
    plot_training_results(reinforce, "REINFORCE - CartPole")
    
    # 测试训练好的策略
    print("\n" + "=" * 60)
    print("测试训练好的策略")
    print("=" * 60)
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(5):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = reinforce.select_action(state)
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Test Episode {episode+1}: Reward = {total_reward:.0f}")
    
    test_env.close()
    
    print("\n" + "=" * 70)
    print("策略梯度算法总结")
    print("=" * 70)
    print("""
    ┌─────────────────────┬────────────────────────────────────────────┐
    │       算法          │                   特点                      │
    ├─────────────────────┼────────────────────────────────────────────┤
    │ REINFORCE           │ 简单直接，但方差大，需要完整episode        │
    ├─────────────────────┼────────────────────────────────────────────┤
    │ REINFORCE+Baseline  │ 降低方差，需要学习价值函数                 │
    ├─────────────────────┼────────────────────────────────────────────┤
    │ Actor-Critic        │ 在线学习，方差小，可处理连续任务           │
    ├─────────────────────┼────────────────────────────────────────────┤
    │ 优势函数            │ A(s,a)=Q(s,a)-V(s) 降低方差                │
    └─────────────────────┴────────────────────────────────────────────┘
    
    关键要点:
    1. 策略梯度直接优化策略，可处理连续动作空间
    2. 使用对数技巧: ∇log π(a|s) = ∇π(a|s)/π(a|s)
    3. 基线(baseline)可降低方差但不引入偏差
    4. Actor-Critic结合了策略梯度和价值函数
    """)