"""
Actor-Critic算法核心公式：

1. 策略梯度（Actor更新）：
   ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]
   其中 A(s,a) = Q(s,a) - V(s) 是优势函数

2. TD误差作为优势函数的估计：
   δ = r + γV(s') - V(s)
   A(s,a) ≈ δ

3. Actor更新（梯度上升）：
   θ ← θ + α_θ * ∇_θ log π_θ(a|s) * δ

4. Critic更新（梯度下降）：
   φ ← φ - α_φ * ∇_φ (r + γV(s') - V(s))^2

5. 损失函数：
   Actor损失: L_A = -log π_θ(a|s) * δ
   Critic损失: L_C = (r + γV(s') - V(s))^2

6. 熵正则化（鼓励探索）：
   L_A_total = L_A - β * H(π_θ)
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

class ActorNetwork(nn.Module):
    """Actor网络 - 输出策略π(a|s)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """输出动作概率分布"""
        return self.network(x)
    
    def get_action(self, state, deterministic=False):
        """
        选择动作并计算对数概率和熵
        
        Args:
            state: 状态张量
            deterministic: 是否确定性选择（测试时使用）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            entropy: 策略的熵
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


class CriticNetwork(nn.Module):
    """Critic网络 - 输出状态价值V(s)"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """输出状态价值"""
        return self.network(x).squeeze(-1)


# ==================== 2. 基础Actor-Critic ====================

class ActorCritic:
    """
    基础Actor-Critic算法（单步更新）
    
    核心思想：
    - Actor: 更新策略，使好的动作更可能被选择
    - Critic: 评估状态价值，提供低方差的优势估计
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr=0.001,
                 critic_lr=0.01,
                 gamma=0.99,
                 hidden_dim=128,
                 device='cpu'):
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.device = device
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'td_errors': []
        }
    
    def select_action(self, state):
        """
        选择动作
        
        Args:
            state: 当前状态（numpy数组）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, state, action, reward, next_state, done, log_prob, value):
        """
        单步更新Actor和Critic
        
        更新公式：
        1. TD目标: td_target = r + γV(s') * (1-done)
        2. TD误差: td_error = td_target - V(s)
        3. Actor损失: actor_loss = -log_prob * td_error
        4. Critic损失: critic_loss = (td_error)^2
        """
        # 转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        
        # 计算TD目标
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)
            td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
            td_error = td_target - value
        
        # 重新计算当前策略的对数概率（用于梯度）
        probs = self.actor(state_tensor)
        dist = Categorical(probs)
        new_log_prob = dist.log_prob(action_tensor)
        
        # Actor损失（策略梯度）
        actor_loss = -(new_log_prob * td_error.detach()).mean()
        
        # Critic损失（价值函数）
        current_value = self.critic(state_tensor)
        critic_loss = nn.MSELoss()(current_value, td_target.detach())
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), td_error.item()
    
    def train(self, env, num_episodes=1000, max_steps=500, render=False):
        """
        训练Actor-Critic
        
        Args:
            env: Gym环境
            num_episodes: 训练的episode数量
            max_steps: 每个episode的最大步数
            render: 是否渲染环境
        """
        print("\n" + "=" * 70)
        print("Actor-Critic算法训练")
        print("=" * 70)
        print(f"状态维度: {env.observation_space.shape[0]}")
        print(f"动作维度: {env.action_space.n}")
        print(f"Actor学习率: {self.actor_optimizer.param_groups[0]['lr']}")
        print(f"Critic学习率: {self.critic_optimizer.param_groups[0]['lr']}")
        print(f"折扣因子 γ: {self.gamma}")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actor_loss = 0
            episode_critic_loss = 0
            episode_td_error = 0
            steps = 0
            
            for step in range(max_steps):
                if render:
                    env.render()
                
                # 选择动作
                action, log_prob, value = self.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 更新网络
                actor_loss, critic_loss, td_error = self.update(
                    state, action, reward, next_state, done, log_prob, value
                )
                
                episode_reward += reward
                episode_length += 1
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_td_error += abs(td_error)
                steps += 1
                
                state = next_state
                
                if done:
                    break
            
            # 记录历史
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['actor_losses'].append(episode_actor_loss / steps)
            self.training_history['critic_losses'].append(episode_critic_loss / steps)
            self.training_history['td_errors'].append(episode_td_error / steps)
            
            # 打印进度
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                avg_length = np.mean(self.training_history['episode_lengths'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:6.1f} | "
                      f"Actor Loss: {episode_actor_loss/steps:.4f} | "
                      f"Critic Loss: {episode_critic_loss/steps:.4f}")
        
        if render:
            env.close()
        
        return self.actor


# ==================== 3. Actor-Critic with Entropy Regularization ====================

class ActorCriticWithEntropy:
    """
    Actor-Critic with Entropy Regularization
    
    添加熵正则化鼓励探索，防止过早收敛到次优策略
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr=0.001,
                 critic_lr=0.01,
                 gamma=0.99,
                 entropy_coef=0.01,  # 熵系数β
                 hidden_dim=128,
                 device='cpu'):
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': []
        }
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item(), entropy.item()
    
    def update(self, state, action, reward, next_state, done, log_prob, value, entropy):
        """
        带熵正则化的更新
        
        Actor损失: L_A = -log_prob * td_error - β * entropy
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        
        # 计算TD误差
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)
            td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
            td_error = td_target - value
        
        # 重新计算对数概率和熵
        probs = self.actor(state_tensor)
        dist = Categorical(probs)
        new_log_prob = dist.log_prob(action_tensor)
        new_entropy = dist.entropy()
        
        # Actor损失（带熵正则化）
        actor_loss = -(new_log_prob * td_error.detach() + 
                       self.entropy_coef * new_entropy).mean()
        
        # Critic损失
        current_value = self.critic(state_tensor)
        critic_loss = nn.MSELoss()(current_value, td_target.detach())
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), new_entropy.mean().item()
    
    def train(self, env, num_episodes=1000, max_steps=500):
        """训练带熵正则化的Actor-Critic"""
        print("\n" + "=" * 70)
        print("Actor-Critic with Entropy Regularization训练")
        print("=" * 70)
        print(f"熵系数 β: {self.entropy_coef}")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actor_loss = 0
            episode_critic_loss = 0
            episode_entropy = 0
            steps = 0
            
            for step in range(max_steps):
                action, log_prob, value, entropy = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                actor_loss, critic_loss, entropy_val = self.update(
                    state, action, reward, next_state, done, log_prob, value, entropy
                )
                
                episode_reward += reward
                episode_length += 1
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_entropy += entropy_val
                steps += 1
                
                state = next_state
                
                if done:
                    break
            
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['actor_losses'].append(episode_actor_loss / steps)
            self.training_history['critic_losses'].append(episode_critic_loss / steps)
            self.training_history['entropies'].append(episode_entropy / steps)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                avg_entropy = np.mean(self.training_history['entropies'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Entropy: {avg_entropy:.4f}")


# ==================== 4. Advantage Actor-Critic (A2C) ====================

class A2C:
    """
    Advantage Actor-Critic (A2C)
    
    使用n步回报计算优势函数，平衡偏差和方差
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr=0.001,
                 critic_lr=0.01,
                 gamma=0.99,
                 n_steps=5,  # n步回报
                 hidden_dim=128,
                 device='cpu'):
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device
        
        # 轨迹存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': []
        }
    
    def select_action(self, state):
        """选择动作并存储轨迹"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_reward(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
    
    def compute_n_step_returns(self, next_state, done):
        """计算n步回报"""
        if done:
            next_value = 0
        else:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).item()
        
        returns = []
        G = next_value
        
        for t in range(len(self.rewards)-1, -1, -1):
            G = self.rewards[t] + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self, next_state, done):
        """n步更新"""
        # 计算n步回报
        returns = self.compute_n_step_returns(next_state, done)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 计算优势
        values_tensor = torch.FloatTensor(self.values).to(self.device)
        advantages = returns_tensor - values_tensor
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor损失
        log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        
        # Critic损失
        critic_loss = nn.MSELoss()(values_tensor, returns_tensor)
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, env, num_episodes=1000, max_steps=500):
        """训练A2C"""
        print("\n" + "=" * 70)
        print(f"A2C (n={self.n_steps})训练")
        print("=" * 70)
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
                
                # 每n步或episode结束时更新
                if (step + 1) % self.n_steps == 0 or done:
                    actor_loss, critic_loss = self.update(next_state, done)
                
                if done:
                    break
            
            # 如果还有未更新的轨迹
            if len(self.states) > 0:
                self.update(state, True)
            
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['losses'].append(actor_loss)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Length: {episode_length}")


# ==================== 5. 可视化函数 ====================

def plot_actor_critic_results(agent, title="Actor-Critic Training Results"):
    """绘制Actor-Critic训练结果"""
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
    ax2.plot(lengths, alpha=0.4, color='green')
    
    if len(lengths) >= window:
        smoothed_steps = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(lengths)), smoothed_steps, 'orange', linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length Curve')
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失曲线
    ax3 = axes[1, 0]
    if 'actor_losses' in history:
        actor_losses = history['actor_losses']
        critic_losses = history['critic_losses']
        
        if len(actor_losses) >= window:
            actor_smoothed = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
            critic_smoothed = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
            ax3.plot(actor_smoothed, label='Actor Loss', color='red')
            ax3.plot(critic_smoothed, label='Critic Loss', color='blue')
    elif 'losses' in history:
        losses = history['losses']
        if len(losses) >= window:
            loss_smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax3.plot(loss_smoothed, label='Loss', color='red')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 熵曲线（如果存在）
    ax4 = axes[1, 1]
    if 'entropies' in history:
        entropies = history['entropies']
        if len(entropies) >= window:
            entropy_smoothed = np.convolve(entropies, np.ones(window)/window, mode='valid')
            ax4.plot(entropy_smoothed, label='Policy Entropy', color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Policy Entropy (Exploration)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    elif 'td_errors' in history:
        td_errors = history['td_errors']
        if len(td_errors) >= window:
            td_smoothed = np.convolve(td_errors, np.ones(window)/window, mode='valid')
            ax4.plot(td_smoothed, label='|TD Error|', color='brown')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('|TD Error|')
        ax4.set_title('TD Error Magnitude')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 6. Actor-Critic演示 ====================

def demonstrate_actor_critic():
    """演示Actor-Critic的核心计算"""
    print("\n" + "=" * 70)
    print("Actor-Critic核心计算演示")
    print("=" * 70)
    
    # 模拟数据
    print("\n1. TD误差计算:")
    print("-" * 40)
    
    reward = 1.0
    gamma = 0.9
    V_s = 0.5
    V_next = 0.8
    done = False
    
    td_target = reward + gamma * V_next * (1 - done)
    td_error = td_target - V_s
    
    print(f"  r = {reward}, γ = {gamma}")
    print(f"  V(s) = {V_s}, V(s') = {V_next}")
    print(f"  TD目标 = r + γV(s') = {reward} + {gamma}*{V_next} = {td_target:.3f}")
    print(f"  TD误差 δ = TD目标 - V(s) = {td_target:.3f} - {V_s} = {td_error:.3f}")
    
    print("\n2. Actor更新（策略梯度）:")
    print("-" * 40)
    
    log_prob = -0.5  # ln(0.6)
    advantage = td_error
    
    actor_gradient = log_prob * advantage
    print(f"  log π(a|s) = {log_prob:.3f}")
    print(f"  优势 A = δ = {advantage:.3f}")
    print(f"  策略梯度 = log_prob * A = {log_prob:.3f} * {advantage:.3f} = {actor_gradient:.3f}")
    
    print("\n3. Critic更新（价值函数）:")
    print("-" * 40)
    
    critic_gradient = 2 * td_error
    print(f"  TD误差: {td_error:.3f}")
    print(f"  Critic梯度 = 2 * δ = {critic_gradient:.3f}")
    
    print("\n4. 优势函数的作用:")
    print("-" * 40)
    print("  - 正优势 (A > 0): 动作优于平均，增加概率")
    print("  - 负优势 (A < 0): 动作劣于平均，减少概率")
    print("  - 零优势 (A = 0): 动作等于平均，不调整")


# ==================== 7. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Actor-Critic算法完整实现")
    print("=" * 70)
    
    # 核心公式说明
    print("\n" + "-" * 70)
    print("核心公式:")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. TD误差（优势估计）:                                         │
    │     δ = r + γV(s') - V(s)                                      │
    │                                                                 │
    │  2. Actor更新（策略梯度）:                                      │
    │     θ ← θ + α_θ * ∇_θ log π_θ(a|s) * δ                        │
    │                                                                 │
    │  3. Critic更新（价值函数）:                                     │
    │     φ ← φ - α_φ * ∇_φ (r + γV(s') - V(s))^2                  │
    │                                                                 │
    │  4. 损失函数:                                                   │
    │     L_A = -log π_θ(a|s) * δ                                   │
    │     L_C = (r + γV(s') - V(s))^2                               │
    │                                                                 │
    │  5. 带熵正则化:                                                 │
    │     L_A_total = L_A - β * H(π_θ)                              │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 演示核心概念
    demonstrate_actor_critic()
    
    # 创建环境
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 选择算法变体
    algorithm = 'a2c'  # 'basic', 'entropy', 'a2c'
    
    if algorithm == 'basic':
        print("\n使用基础Actor-Critic")
        agent = ActorCritic(
            state_dim, action_dim,
            actor_lr=0.001,
            critic_lr=0.01,
            gamma=0.99,
            hidden_dim=128
        )
    elif algorithm == 'entropy':
        print("\n使用Actor-Critic with Entropy")
        agent = ActorCriticWithEntropy(
            state_dim, action_dim,
            actor_lr=0.001,
            critic_lr=0.01,
            gamma=0.99,
            entropy_coef=0.01,
            hidden_dim=128
        )
    else:  # a2c
        print("\n使用A2C (Advantage Actor-Critic)")
        agent = A2C(
            state_dim, action_dim,
            actor_lr=0.001,
            critic_lr=0.01,
            gamma=0.99,
            n_steps=5,
            hidden_dim=128
        )
    
    # 训练
    agent.train(env, num_episodes=500, max_steps=500)
    
    # 绘制结果
    plot_actor_critic_results(agent, algorithm.upper())
    
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
                action, _, _ = agent.actor.get_action(state_tensor, deterministic=True)
            
            state, reward, terminated, truncated, _ = test_env.step(action.item())
            done = terminated or truncated
            total_reward += reward
        
        print(f"Test Episode {episode+1}: Reward = {total_reward:.0f}")
    
    test_env.close()
    
    print("\n" + "=" * 70)
    print("Actor-Critic算法总结")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  Actor-Critic算法特点:                                          │
    │  - 结合策略梯度（Actor）和价值函数（Critic）                    │
    │  - 单步更新，不需要完整episode                                  │
    │  - 使用TD误差作为优势估计，降低方差                             │
    │                                                                 │
    │  变体对比:                                                      │
    │  - 基础AC: 最简单的实现                                         │
    │  - AC+熵: 熵正则化鼓励探索                                      │
    │  - A2C: n步回报，平衡偏差与方差                                 │
    │                                                                 │
    │  优势:                                                          │
    │  - 在线学习，可处理无限长任务                                   │
    │  - 方差小，训练稳定                                             │
    │  - 适合连续动作空间                                             │
    └─────────────────────────────────────────────────────────────────┘
    """)