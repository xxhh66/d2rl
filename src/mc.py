"""
算法：
MC->TD->Q-Learning->SARSA->REINFORCE->Actor-Critic->PPO
蒙特卡洛方法核心公式：

1. 首次访问MC预测（First-visit MC）：
   V(s) = average of returns following first visit to s

2. 每次访问MC预测（Every-visit MC）：
   V(s) = average of returns following every visit to s

3. 回报计算（Return）：
   G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ... + γ^{T-1}R_T

4. 增量更新公式：
   V(s) ← V(s) + α [G_t - V(s)]

5. 蒙特卡洛控制（MC Control）：
   Q(s,a) ← Q(s,a) + α [G_t - Q(s,a)]
   π(s) = argmax_a Q(s,a)  (ε-贪婪)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 1. 蒙特卡洛预测（策略评估） ====================

class MonteCarloPrediction:
    """
    蒙特卡洛预测 (Monte Carlo Prediction)
    1. 完整的回合：必须从初始状态到终止状态的完整奖励
    2. 回报Gt:从状态s到回合结束的折扣累计奖励
        Gt = R_{t+1} +\gamma R_{t+2}
    3. 目标:估计V = E[G|S=s] 状态s的期望回报

    实现方式两种：
    1. 首次访问MC
    2. 每次访问MC
    算法说明：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 学习目标：   评估给定策略的状态价值函数 V(s)                     │
    │ Episode结束：必须等待完整episode结束（生成完整轨迹）            │
    │ 动作选择：   使用固定的评估策略（不探索）                       │
    │ 更新对象：   状态价值 V(s)                                      │
    │ 是否Bootstrapping： 否 - 使用真实回报 G_t，不是估计           │
    │ 更新时机：   Episode结束后，从后向前计算回报并更新              │
    │ 更新公式：   V(s) ← V(s) + α [G_t - V(s)]                       │
    │ 核心特点：   无偏估计，但方差较大                                │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, gamma=0.9):
        self.gamma = gamma
    
    def _generate_episode(self, env, policy, max_steps):
        """生成episode"""
        episode = []
        state, _ = env.reset()
        
        for _ in range(max_steps):
            # 确保policy返回有效的动作
            action = policy(state)
            if action is None:
                # 如果策略返回None，使用随机动作
                action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            
            if terminated or truncated:
                break
        
        return episode
    
    def first_visit_mc(self, env, policy, num_episodes=1000, max_steps=100):
        """首次访问蒙特卡洛预测"""
        returns_sum = defaultdict(float)
        returns_count = defaultdict(int)
        V = defaultdict(float)
        
        print("\n首次访问蒙特卡洛预测...")
        print("-" * 60)
        
        for episode in range(num_episodes):
            episode_data = self._generate_episode(env, policy, max_steps)
            
            first_visit = set()
            G = 0
            
            for t in range(len(episode_data)-1, -1, -1):
                state, reward = episode_data[t]
                G = reward + self.gamma * G
                
                if state not in first_visit:
                    first_visit.add(state)
                    returns_sum[state] += G
                    returns_count[state] += 1
                    V[state] = returns_sum[state] / returns_count[state]
            
            if (episode + 1) % 200 == 0:
                print(f"已完成 {episode+1}/{num_episodes} episodes")
        
        return V
    
    def every_visit_mc(self, env, policy, num_episodes=1000, max_steps=100):
        """每次访问蒙特卡洛预测"""
        returns_sum = defaultdict(float)
        returns_count = defaultdict(int)
        V = defaultdict(float)
        
        print("\n每次访问蒙特卡洛预测...")
        print("-" * 60)
        
        for episode in range(num_episodes):
            episode_data = self._generate_episode(env, policy, max_steps)
            
            G = 0
            for t in range(len(episode_data)-1, -1, -1):
                state, reward = episode_data[t]
                G = reward + self.gamma * G
                
                returns_sum[state] += G
                returns_count[state] += 1
                # 蒙特卡洛方法的核心，表示用样本平均值来估计期望值
                V[state] = returns_sum[state] / returns_count[state]
            
            if (episode + 1) % 200 == 0:
                print(f"已完成 {episode+1}/{num_episodes} episodes")
        
        return V


# ==================== 2. 蒙特卡洛控制（策略优化） ====================

class MonteCarloControl:
    """
    蒙特卡洛控制 (Monte Carlo Control)
    蒙特卡洛控制 = 蒙特卡洛预测（评估策略） + 策略迭代（优化策略）
    目标：直接从经验轨迹中学习最优策略 π，不需要环境模型*
    核心思想：边玩边学 → 评估当前策略 → 立刻贪心优化策略 → 重复直到策略最优
    概念：
    1. 控制≠预测：控制，要学习最优策略，必须学习Q动作价值
    2. 必须用动作价值Q
    3. 探索与利用，必须保留随机探索
    4. 算法：
        在线策略，学习和执行用同一个策略
        离线策略：学习最优策略，用随机策略探索
    步骤：
    1. 初始化：动作对（s,a）、折扣因子、探索率、最大回合数
    2. 用 ϵ- 贪心策略生成完整回合
        概率 ϵ：随机选动作（保证探索）
        概率 1-ϵ：选 Q (s,a) 最大的动作（贪心利用）
    3. 反向计算回报G：G = \gamma*G+R
    4. 首次访问MC更新Q(s,a)
    5. 策略提升：对每个状态 s，让策略直接变成最优：π(s) = argmaxₐ Q(s,a)得到 最优动作价值 Q* 和 最优策略 π*
    6. 循环直到收敛：重复 回合生成 → 评估 Q → 优化策略
    算法说明：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 学习目标：   学习最优策略 π*(s) 和动作价值 Q(s,a)               │
    │ Episode结束：必须等待完整episode结束                           │
    │ 动作选择：   ε-贪婪策略（平衡探索与利用）                       │
    │ 更新对象：   动作价值 Q(s,a)                                    │
    │ 是否Bootstrapping： 否 - 使用真实回报 G_t                     │
    │ 更新时机：   Episode结束后更新                                 │
    │ 更新公式：   Q(s,a) ← Q(s,a) + α [G_t - Q(s,a)]                │
    │ 策略提升：   贪心策略 π(s) = argmax_a Q(s,a)                    │
    │ 核心特点：   使用ε-贪婪策略保证探索                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 获取动作空间大小
        self.n_actions = env.action_space.n
        
        # 初始化Q表和策略
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.reward_history = []
        self.length_history = []
    
    def _epsilon_greedy_policy(self, state):
        """
        ε-贪婪策略
        
        返回: 动作（整数）
        """
        # 确保state可以被正确处理
        if isinstance(state, tuple):
            state = state[0]
        
        # ε-贪婪选择
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.Q[state])
    
    def _generate_episode(self, policy_func, max_steps):
        """
        生成episode
        
        Args:
            policy_func: 策略函数，接收状态返回动作
            max_steps: 最大步数
        """
        episode = []
        state, _ = self.env.reset()
        
        for _ in range(max_steps):
            # 调用策略函数获取动作
            action = policy_func(state)
            
            # 确保动作有效
            if action is None or action >= self.n_actions:
                action = np.random.randint(self.n_actions)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            
            if terminated or truncated:
                break
        
        return episode
    
    def on_policy_first_visit_mc(self, num_episodes=1000, max_steps=100):
        """
        On-policy 首次访问蒙特卡洛控制
        """
        returns_sum = defaultdict(float)
        returns_count = defaultdict(int)
        
        print("\nOn-policy 首次访问蒙特卡洛控制...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            # 使用当前策略生成episode
            episode_data = self._generate_episode(self._epsilon_greedy_policy, max_steps)
            
            first_visit = set()
            G = 0
            
            # 从后向前计算回报
            for t in range(len(episode_data)-1, -1, -1):
                state, action, reward = episode_data[t]
                G = reward + self.gamma * G
                
                # 首次访问更新
                if (state, action) not in first_visit:
                    first_visit.add((state, action))
                    returns_sum[(state, action)] += G
                    returns_count[(state, action)] += 1
                    self.Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
            
            # 记录奖励
            episode_reward = sum(r for _, _, r in episode_data)
            self.reward_history.append(episode_reward)
            self.length_history.append(len(episode_data))
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else episode_reward
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Length: {len(episode_data):3d} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return self.Q, self._epsilon_greedy_policy
    
    def extract_policy(self):
        """提取贪婪策略"""
        def greedy_policy(state):
            if isinstance(state, tuple):
                state = state[0]
            return np.argmax(self.Q[state])
        return greedy_policy


# ==================== 3. 增量蒙特卡洛 ====================

class IncrementalMonteCarlo:
    """增量蒙特卡洛"""
    
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.n_actions = env.action_space.n
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.reward_history = []
    
    def _epsilon_greedy(self, state):
        """ε-贪婪动作选择"""
        if isinstance(state, tuple):
            state = state[0]
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=1000, max_steps=100):
        """增量蒙特卡洛训练"""
        print("\n增量蒙特卡洛控制...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            episode_data = []
            state, _ = self.env.reset()
            
            # 生成episode
            for _ in range(max_steps):
                action = self._epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                state = next_state
                
                if terminated or truncated:
                    break
            
            # 计算回报并增量更新
            G = 0
            for t in range(len(episode_data)-1, -1, -1):
                state, action, reward = episode_data[t]
                G = reward + self.gamma * G
                
                # 增量更新
                self.Q[state][action] += self.alpha * (G - self.Q[state][action])
            
            # 记录
            episode_reward = sum(r for _, _, r in episode_data)
            self.reward_history.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else episode_reward
                print(f"Episode {episode+1:4d}/{num_episodes} | Avg Reward: {avg_reward:.3f}")
        
        return self.Q
    
    def extract_policy(self):
        """提取贪婪策略"""
        def greedy_policy(state):
            if isinstance(state, tuple):
                state = state[0]
            return np.argmax(self.Q[state])
        return greedy_policy


# ==================== 4. 测试和演示 ====================

def test_mc_prediction():
    """测试蒙特卡洛预测"""
    print("\n" + "=" * 70)
    print("测试1：蒙特卡洛预测")
    print("=" * 70)
    
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    # 定义随机策略
    def random_policy(state):
        return env.action_space.sample()
    
    # 创建预测器
    mc_pred = MonteCarloPrediction(gamma=0.99)
    
    # 首次访问MC
    V_first = mc_pred.first_visit_mc(env, random_policy, num_episodes=1000, max_steps=100)
    
    print("\n首次访问MC价值函数:")
    for i in range(4):
        row = []
        for j in range(4):
            state = i * 4 + j
            row.append(f"{V_first.get(state, 0):6.3f}")
        print("  " + "  ".join(row))


def test_mc_control():
    """测试蒙特卡洛控制"""
    print("\n" + "=" * 70)
    print("测试2：蒙特卡洛控制")
    print("=" * 70)
    
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    # 创建控制器
    mc_control = MonteCarloControl(env, gamma=0.99, epsilon=0.1)
    
    # 训练
    Q, policy = mc_control.on_policy_first_visit_mc(num_episodes=500, max_steps=100)
    
    # 显示策略
    print("\n学习到的最优策略:")
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    for i in range(4):
        row = []
        for j in range(4):
            state = i * 4 + j
            if state in Q:
                best_action = np.argmax(Q[state])
                row.append(action_symbols[best_action])
            else:
                row.append('?')
        print("  " + "  ".join(row))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mc_control.reward_history, alpha=0.5)
    # 滑动平均
    window = 50
    if len(mc_control.reward_history) >= window:
        smoothed = np.convolve(mc_control.reward_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(mc_control.reward_history)), smoothed, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('训练奖励曲线')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 成功率曲线
    success_rate = []
    for i in range(window, len(mc_control.reward_history) + 1):
        rate = np.mean(mc_control.reward_history[i-window:i])
        success_rate.append(rate)
    plt.plot(range(window, len(mc_control.reward_history) + 1), success_rate, 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('成功率曲线')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80%目标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return Q, policy


def test_incremental_mc():
    """测试增量蒙特卡洛"""
    print("\n" + "=" * 70)
    print("测试3：增量蒙特卡洛")
    print("=" * 70)
    
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    # 创建增量MC
    inc_mc = IncrementalMonteCarlo(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    
    # 训练
    Q = inc_mc.train(num_episodes=500, max_steps=100)
    
    # 显示Q表
    print("\n学习到的Q表（部分）:")
    for state in range(5):
        print(f"State {state}: {Q[state]}")
    
    return Q


# ==================== 5. 演示蒙特卡洛核心概念 ====================

def demonstrate_mc_concepts():
    """演示蒙特卡洛核心概念"""
    print("\n" + "=" * 70)
    print("蒙特卡洛算法核心概念演示")
    print("=" * 70)
    
    # 模拟一个episode
    print("\n1. Episode生成:")
    print("-" * 40)
    
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    state, _ = env.reset()
    print(f"初始状态: {state}")
    
    episode = []
    for step in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        print(f"  步骤{step+1}: 状态{state} -> 动作{action} -> 奖励{reward} -> 状态{next_state}")
        state = next_state
        if terminated:
            print("  到达目标！")
            break
    
    # 计算回报
    print("\n2. 回报计算:")
    print("-" * 40)
    gamma = 0.9
    
    for t in range(len(episode)):
        G = 0
        for k in range(t, len(episode)):
            G += (gamma ** (k - t)) * episode[k][2]
        print(f"  G_{t} = {G:.3f}")
    
    # 首次访问 vs 每次访问
    print("\n3. 首次访问 vs 每次访问:")
    print("-" * 40)
    
    states_with_repeats = ['A', 'B', 'A', 'C', 'A']
    print(f"状态序列: {states_with_repeats}")
    
    first_visits = set()
    print("\n首次访问:")
    for i, s in enumerate(states_with_repeats):
        if s not in first_visits:
            first_visits.add(s)
            print(f"  时间{i}: {s} ✓ (记录)")
        else:
            print(f"  时间{i}: {s} ✗ (忽略)")
    
    print("\n每次访问:")
    for i, s in enumerate(states_with_repeats):
        print(f"  时间{i}: {s} ✓ (记录)")


# ==================== 6. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("蒙特卡洛算法 (Monte Carlo Methods) 完整实现")
    print("=" * 70)
    
    # 演示核心概念
    demonstrate_mc_concepts()
    
    # 测试蒙特卡洛预测
    test_mc_prediction()
    
    # 测试蒙特卡洛控制
    Q, policy = test_mc_control()
    
    # 测试增量蒙特卡洛（可选）
    # test_incremental_mc()
    
    print("\n" + "=" * 70)
    print("算法总结")
    print("=" * 70)
    print("""
    ┌─────────────────┬────────────────────────────────────────────┐
    │     算法        │                   特点                      │
    ├─────────────────┼────────────────────────────────────────────┤
    │ First-visit MC  │ 只使用第一次访问，无偏估计                  │
    ├─────────────────┼────────────────────────────────────────────┤
    │ Every-visit MC  │ 使用所有访问，有偏但更高效                  │
    ├─────────────────┼────────────────────────────────────────────┤
    │ On-policy MC    │ 使用ε-贪婪，边探索边学习                    │
    ├─────────────────┼────────────────────────────────────────────┤
    │ Incremental MC  │ 增量更新，节省内存                          │
    └─────────────────┴────────────────────────────────────────────┘
    
    关键修复：
    1. 策略函数始终返回有效的动作（整数）
    2. 使用 defaultdict 避免KeyError
    3. 正确处理环境返回的元组状态
    """)