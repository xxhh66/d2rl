"""
算法：
MC->TD->Q-Learning->SARSA->REINFORCE->Actor-Critic->PPO
时序差分(TD)核心公式:
1. TD预测(TD(0)):
   V(s_t) ← V(s_t) + α [r_{t+1} + gammaV(s_{t+1}) - V(s_t)]
2. TD误差(TD Error):
   δ_t = r_{t+1} + gammaV(s_{t+1}) - V(s_t)
3. Sarsa (On-policy TD Control):
   Q(s_t,a_t) ← Q(s_t,a_t) + α [r_{t+1} + gammaQ(s_{t+1},a_{t+1}) - Q(s_t,a_t)]
4. Q-Learning (Off-policy TD Control):
   Q(s_t,a_t) ← Q(s_t,a_t) + α [r_{t+1} + gamma max_a Q(s_{t+1},a) - Q(s_t,a_t)]
5. Expected Sarsa:
   Q(s_t,a_t) ← Q(s_t,a_t) + α [r_{t+1} + gamma Σ_a π(a|s_{t+1})Q(s_{t+1},a) - Q(s_t,a_t)]
6. n步TD:
   G_t^{(n)} = r_{t+1} + gammar_{t+2} + ... + gamma^{n-1}r_{t+n} + gamma^n V(s_{t+n})
步骤:
1. 初始化:V=0,学习率,折扣因子gamma,回合数
2. 环境交互，每步更新
V(S_{t})<-V(S_{t})+α[R_{t+1}+gamma*V(S_{t+1})-V(S_{t})]
括号内就是 TD 目标 - TD 误差
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym
from tqdm import tqdm
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 1. TD预测(策略评估) ====================

class TDPrediction:
    """
    时序差分预测(TD Prediction)
    
    用于评估给定策略的价值函数
    核心:使用TD误差进行增量更新
    """
    
    def __init__(self, gamma=0.9, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
    
    def td0_prediction(self, env, policy, num_episodes=1000, max_steps=100):
        """
        TD(0) 预测算法
        
        更新公式: V(s) ← V(s) +  alpha*[r + gamma*V(s') - V(s)]
        
        Args:
            env: 环境
            policy: 策略函数
            num_episodes: episode数量
            max_steps: 最大步数
        """
        V = defaultdict(float)
        td_errors = []
        
        print("\nTD(0) 预测...")
        print("-" * 60)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_errors = []
            
            for step in range(max_steps):
                # 选择动作
                action = policy(state)
                
                # 执行动作,terminated 任务是否结束，truncated任务是否强制截断
                next_state, reward, terminated, truncated, _ = env.step(action)
                # 
                total_reward += reward
                
                # TD更新
                if not (terminated or truncated):
                    td_target = reward + self.gamma * V[next_state]
                else:
                    td_target = reward
                
                td_error = td_target - V[state]
                V[state] += self.alpha * td_error
                
                episode_errors.append(td_error)
                
                state = next_state
                
                if terminated or truncated:
                    break
            
            td_errors.extend(episode_errors)
            
            if (episode + 1) % 200 == 0:
                avg_reward = total_reward / (step + 1)
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Steps: {step+1}")
        
        return V, td_errors
    
    def n_step_td_prediction(self, env, policy, n=3, num_episodes=1000, max_steps=100):
        """
        n步TD预测
        
        公式: G_t^{(n)} = r_{t+1} + gamma*r_{t+2} + ... + gamma*^{n-1}r_{t+n} + gamma*^n V(s_{t+n})

        n步TD预测的核心公式:

        1. n步回报(n-step Return):
        G_t^{(n)} = R_{t+1} + gamma*R_{t+2} + gamma²R_{t+3} + ... + gamma^{n-1}R_{t+n} + gamma^n V(S_{t+n})

        2. n步TD更新公式:
        V(S_t) ← V(S_t) + α [G_t^{(n)} - V(S_t)]

        3. 特殊情况:
        - n=1: TD(0)  G_t^{(1)} = R_{t+1} + gamma*V(S_{t+1})
        - n=∞: MC     G_t^{(∞)} = R_{t+1} + gamma*R_{t+2} + ... + gamma^{T-1}R_T
        """
        V = defaultdict(float)
        
        print(f"\n{n}-步TD预测...")
        print("-" * 60)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            
            # 存储轨迹
            states = [state]
            rewards = [0]
            
            T = float('inf')
            t = 0
            
            while True:
                if t < T:
                    action = policy(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    states.append(next_state)
                    rewards.append(reward)
                    
                    if terminated or truncated:
                        T = t + 1
                
                tau = t - n + 1
                
                if tau >= 0:
                    # 计算n步回报
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]
                    
                    if tau + n < T:
                        G += (self.gamma ** n) * V[states[tau + n]]
                    
                    # 更新价值
                    V[states[tau]] += self.alpha * (G - V[states[tau]])
                
                t += 1
                state = next_state
                
                if tau == T - 1:
                    break
            
            if (episode + 1) % 200 == 0:
                print(f"Episode {episode+1:4d}/{num_episodes}")
        
        return V


# ==================== 2. Sarsa算法(On-policy TD Control) ====================

class Sarsa:
    """
    Sarsa算法 (State-Action-Reward-State-Action)
    
    On-policy TD控制算法
    更新公式:Q(s,a) ← Q(s,a) + α [r + gammaQ(s',a') - Q(s,a)]
    """
    
    def __init__(self, 
                 env,
                 gamma=0.9,
                 alpha=0.1,
                 epsilon=0.1,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 获取状态和动作空间
        self.n_states = env.observation_space.n if hasattr(env.observation_space, 'n') else 1
        self.n_actions = env.action_space.n
        
        # 初始化Q表
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        
        # 训练历史
        self.reward_history = []
        self.length_history = []
        self.epsilon_history = []
    
    def _epsilon_greedy(self, state):
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=1000, max_steps=100):
        """
        Sarsa训练
        
        算法步骤:
        1. 初始化Q(s,a)
        2. 从状态s开始，使用ε-贪婪选择动作a
        3. 执行动作，观察r, s'
        4. 使用ε-贪婪选择a'
        5. 更新Q(s,a) ← Q(s,a) + α[r + gammaQ(s',a') - Q(s,a)]
        6. s ← s', a ← a'
        7. 重复直到终止
        """
        print("\nSarsa训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            action = self._epsilon_greedy(state)
            
            total_reward = 0
            step_count = 0
            
            for _ in range(max_steps):
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # 选择下一个动作
                if not done:
                    next_action = self._epsilon_greedy(next_state)
                else:
                    next_action = None
                
                # Sarsa更新
                if not done:
                    td_target = reward + self.gamma * self.Q[next_state][next_action]
                else:
                    td_target = reward
                
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                # 更新状态和动作
                state = next_state
                action = next_action
                
                if done:
                    break
            
            # 记录
            self.reward_history.append(total_reward)
            self.length_history.append(step_count)
            self.epsilon_history.append(self.epsilon)
            
            # 衰减ε
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {step_count}")
        
        return self.Q
    
    def extract_policy(self):
        """提取贪婪策略"""
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy


# ==================== 3. Q-Learning算法(Off-policy TD Control) ====================

class QLearning:
    """
    Q-Learning算法
    
    Off-policy TD控制算法
    更新公式:Q(s,a) ← Q(s,a) + α [r + gamma max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self,
                 env,
                 gamma=0.9,
                 alpha=0.1,
                 epsilon=0.1,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.n_states = env.observation_space.n if hasattr(env.observation_space, 'n') else 1
        self.n_actions = env.action_space.n
        
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        
        self.reward_history = []
        self.length_history = []
        self.epsilon_history = []
    
    def _epsilon_greedy(self, state):
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=1000, max_steps=100):
        """
        Q-Learning训练
        
        算法步骤:
        1. 初始化Q(s,a)
        2. 从状态s开始
        3. 使用ε-贪婪选择动作a
        4. 执行动作，观察r, s'
        5. 更新Q(s,a) ← Q(s,a) + α[r + gamma max_a' Q(s',a') - Q(s,a)]
        6. s ← s'
        7. 重复直到终止
        """
        print("\nQ-Learning训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            step_count = 0
            
            for _ in range(max_steps):
                # 选择动作
                action = self._epsilon_greedy(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Q-Learning更新(使用max)
                if not done:
                    td_target = reward + self.gamma * np.max(self.Q[next_state])
                else:
                    td_target = reward
                
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                state = next_state
                
                if done:
                    break
            
            self.reward_history.append(total_reward)
            self.length_history.append(step_count)
            self.epsilon_history.append(self.epsilon)
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:])
                print(f"Episode {episode+1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {step_count}")
        
        return self.Q
    
    def extract_policy(self):
        """提取贪婪策略"""
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy


# ==================== 4. Expected Sarsa算法 ====================

class ExpectedSarsa:
    """
    Expected Sarsa算法
    
    结合Sarsa和Q-Learning的优点
    更新公式:Q(s,a) ← Q(s,a) + α [r + gamma Σ_a π(a|s')Q(s',a) - Q(s,a)]
    """
    
    def __init__(self,
                 env,
                 gamma=0.9,
                 alpha=0.1,
                 epsilon=0.1,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.n_actions = env.action_space.n
        
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.reward_history = []
    
    def _epsilon_greedy_probs(self, state):
        """获取ε-贪婪策略的概率分布"""
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        best_action = np.argmax(self.Q[state])
        probs[best_action] += 1 - self.epsilon
        return probs
    
    def _epsilon_greedy(self, state):
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=1000, max_steps=100):
        """
        Expected Sarsa训练
        """
        print("\nExpected Sarsa训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            for _ in range(max_steps):
                action = self._epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                # Expected Sarsa更新(使用期望值)
                if not done:
                    probs = self._epsilon_greedy_probs(next_state)
                    expected_q = np.sum(probs * self.Q[next_state])
                    td_target = reward + self.gamma * expected_q
                else:
                    td_target = reward
                
                self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])
                
                state = next_state
                
                if done:
                    break
            
            self.reward_history.append(total_reward)
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:])
                print(f"Episode {episode+1:4d}/{num_episodes} | Avg Reward: {avg_reward:.2f}")
        
        return self.Q


# ==================== 5. Double Q-Learning ====================

class DoubleQLearning:
    """
    Double Q-Learning
    
    解决Q-Learning的过估计问题
    使用两个独立的Q表
    """
    
    def __init__(self,
                 env,
                 gamma=0.9,
                 alpha=0.1,
                 epsilon=0.1):
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.n_actions = env.action_space.n
        
        # 两个Q表
        self.Q1 = defaultdict(lambda: np.zeros(self.n_actions))
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))
        
        self.reward_history = []
    
    def _epsilon_greedy(self, state):
        """ε-贪婪动作选择(使用Q1+Q2)"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # 使用两个Q表的平均值
            q_avg = self.Q1[state] + self.Q2[state]
            return np.argmax(q_avg)
    
    def train(self, num_episodes=1000, max_steps=100):
        """
        Double Q-Learning训练
        
        更新规则:
        以0.5概率更新Q1，否则更新Q2
        """
        print("\nDouble Q-Learning训练...")
        print("-" * 70)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            for _ in range(max_steps):
                action = self._epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                if not done:
                    # 随机选择更新哪个Q表
                    if np.random.random() < 0.5:
                        # 更新Q1
                        best_action = np.argmax(self.Q1[next_state])
                        td_target = reward + self.gamma * self.Q2[next_state][best_action]
                        self.Q1[state][action] += self.alpha * (td_target - self.Q1[state][action])
                    else:
                        # 更新Q2
                        best_action = np.argmax(self.Q2[next_state])
                        td_target = reward + self.gamma * self.Q1[next_state][best_action]
                        self.Q2[state][action] += self.alpha * (td_target - self.Q2[state][action])
                else:
                    # 终止状态
                    if np.random.random() < 0.5:
                        self.Q1[state][action] += self.alpha * (reward - self.Q1[state][action])
                    else:
                        self.Q2[state][action] += self.alpha * (reward - self.Q2[state][action])
                
                state = next_state
                
                if done:
                    break
            
            self.reward_history.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:])
                print(f"Episode {episode+1:4d}/{num_episodes} | Avg Reward: {avg_reward:.2f}")
        
        return self.Q1, self.Q2


# ==================== 6. TD算法对比实验 ====================

def compare_td_algorithms():
    """对比不同TD算法的性能"""
    print("\n" + "=" * 70)
    print("TD算法对比实验 - FrozenLake")
    print("=" * 70)
    
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    algorithms = {
        'Sarsa': Sarsa,
        'Q-Learning': QLearning,
        'Expected Sarsa': ExpectedSarsa
    }
    
    results = {}
    
    for name, Algorithm in algorithms.items():
        print(f"\n运行 {name}...")
        print("-" * 50)
        
        agent = Algorithm(env, gamma=0.99, alpha=0.1, epsilon=0.1)
        agent.train(num_episodes=500, max_steps=100)
        results[name] = agent.reward_history
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    
    for name, rewards in results.items():
        # 计算滑动平均
        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=name, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (window=50)')
    plt.title('TD算法性能对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 7. TD误差可视化 ====================

def visualize_td_error():
    """可视化TD误差"""
    print("\n" + "=" * 70)
    print("TD误差可视化")
    print("=" * 70)
    
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    # 随机策略
    def random_policy(state):
        return np.random.randint(env.action_space.n)
    
    # TD预测
    td_pred = TDPrediction(gamma=0.99, alpha=0.1)
    V, td_errors = td_pred.td0_prediction(env, random_policy, num_episodes=500)
    
    # 绘制TD误差分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(td_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('TD Error')
    plt.ylabel('Frequency')
    plt.title('TD误差分布', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(td_errors[:500], alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('TD Error')
    plt.title('TD误差随时间变化', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 8. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("时序差分算法 (Temporal Difference Learning) 完整实现")
    print("=" * 70)
    
    # 核心公式说明
    print("\n" + "-" * 70)
    print("核心公式:")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. TD(0) 预测:                                                │
    │     V(s) ← V(s) + α [r + gammaV(s') - V(s)]                       │
    │                                                                 │
    │  2. Sarsa (On-policy):                                         │
    │     Q(s,a) ← Q(s,a) + α [r + gammaQ(s',a') - Q(s,a)]              │
    │                                                                 │
    │  3. Q-Learning (Off-policy):                                   │
    │     Q(s,a) ← Q(s,a) + α [r + gamma max_a' Q(s',a') - Q(s,a)]      │
    │                                                                 │
    │  4. Expected Sarsa:                                            │
    │     Q(s,a) ← Q(s,a) + α [r + gamma Σ_a π(a|s')Q(s',a) - Q(s,a)]   │
    │                                                                 │
    │  5. n步TD:                                                      │
    │     G_t^{(n)} = r_{t+1} + gammar_{t+2} + ... + gamma^{n-1}r_{t+n}      │
    │                + gamma^n V(s_{t+n})                                │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 创建环境
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    # 1. Sarsa
    print("\n" + "=" * 60)
    sarsa_agent = Sarsa(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    sarsa_agent.train(num_episodes=500)
    
    # 2. Q-Learning
    print("\n" + "=" * 60)
    ql_agent = QLearning(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    ql_agent.train(num_episodes=500)
    
    # 3. 可视化TD误差
    visualize_td_error()
    
    # 4. 算法对比(可选)
    # compare_td_algorithms()
    
    # 显示Q表
    print("\n" + "=" * 70)
    print("学习到的Q表(Q-Learning):")
    print("=" * 70)
    
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    for i in range(4):
        row = []
        for j in range(4):
            state = i * 4 + j
            if state in ql_agent.Q:
                best_action = np.argmax(ql_agent.Q[state])
                row.append(action_symbols[best_action])
            else:
                row.append('?')
        print("  " + "  ".join(row))
    
    print("\n" + "=" * 70)
    print("TD算法总结")
    print("=" * 70)
    print("""
    ┌─────────────────┬────────────────────────────────────────────┐
    │     算法        │                   特点                      │
    ├─────────────────┼────────────────────────────────────────────┤
    │ Sarsa           │ On-policy，更安全，适合在线学习            │
    ├─────────────────┼────────────────────────────────────────────┤
    │ Q-Learning      │ Off-policy，更激进，可能过估计             │
    ├─────────────────┼────────────────────────────────────────────┤
    │ Expected Sarsa  │ 结合两者优点，方差更小                     │
    ├─────────────────┼────────────────────────────────────────────┤
    │ Double Q-Learning│ 解决过估计问题，更稳定                    │
    └─────────────────┴────────────────────────────────────────────┘
    
    TD vs MC:
    - TD: 在线学习，有偏估计，方差小
    - MC: 需要完整episode，无偏估计，方差大
    """)