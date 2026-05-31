"""
策略评估核心公式：
1. 贝尔曼期望方程（状态价值函数）：
   V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
2. 贝尔曼期望方程（动作价值函数）：
   Q^π(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) Σ_a' π(a'|s') Q^π(s',a')
3. 迭代更新公式：
   V_{k+1}(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]
4. 收敛条件：
   max_s |V_{k+1}(s) - V_k(s)| < θ
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 策略评估算法类 ====================

class PolicyEvaluator:
    """
    策略评估算法 (Policy Evaluation / Iterative Policy Evaluation)
    
    用于计算给定策略下的状态价值函数 V^π(s)
    核心思想：迭代应用贝尔曼期望方程直到收敛
    """
    
    def __init__(self, 
                 states,           # 状态列表
                 actions,          # 动作列表
                 transition_prob,  # 转移概率 P(s'|s,a)
                 reward_func,      # 奖励函数 R(s,a,s')
                 gamma=0.9,        # 折扣因子
                 theta=1e-6,       # 收敛阈值
                 max_iter=1000):   # 最大迭代次数
        
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        
        # 构建转移概率矩阵 P[s][a][s']
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for (s, a, s_next), prob in transition_prob.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            s_next_idx = self.states.index(s_next)
            self.P[s_idx, a_idx, s_next_idx] = prob
        
        # 构建奖励函数 R[s][a][s']
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        for (s, a, s_next), reward in reward_func.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            s_next_idx = self.states.index(s_next)
            self.R[s_idx, a_idx, s_next_idx] = reward
    
    def policy_evaluation(self, policy, method='iterative', num_episodes=1000, max_steps=100):
        """
        策略评估主函数
        
        Args:
            policy: 策略函数 π(a|s)，字典 {(s,a): probability}
            method: 评估方法 ('iterative', 'matrix', 'monte_carlo')
            num_episodes: 蒙特卡洛方法的episode数量
            max_steps: 蒙特卡洛方法的最大步数
        
        Returns:
            V: 状态价值函数字典 {state: value}
            V_history: 迭代历史（仅迭代法有）
        """
        if method == 'iterative':
            return self._iterative_policy_evaluation(policy)
        elif method == 'matrix':
            return self._matrix_policy_evaluation(policy)
        elif method == 'monte_carlo':
            return self._monte_carlo_policy_evaluation(policy, num_episodes, max_steps)
        else:
            raise ValueError(f"未知方法: {method}")
    
    def _iterative_policy_evaluation(self, policy):
        """
        迭代策略评估
        
        算法步骤：
        1. 初始化 V(s) = 0
        2. 对于每个状态，应用贝尔曼期望方程更新
        3. 重复直到收敛
        """
        # 构建策略矩阵
        policy_mat = self._policy_to_matrix(policy)
        
        # 初始化价值函数
        V = np.zeros(self.n_states)
        V_history = [V.copy()]
        
        print("\n开始迭代策略评估...")
        print("-" * 60)
        
        for iteration in range(self.max_iter):
            V_new = np.zeros(self.n_states)
            max_diff = 0
            
            # 对每个状态更新价值
            for s in range(self.n_states):
                # 贝尔曼期望方程
                V_new[s] = self._bellman_expectation(s, policy_mat[s], V)
                
                # 记录最大变化
                max_diff = max(max_diff, abs(V_new[s] - V[s]))
            
            V_history.append(V_new.copy())
            
            # 检查收敛
            if max_diff < self.theta:
                print(f"收敛于第 {iteration+1} 次迭代，最大变化={max_diff:.2e}")
                break
            
            V = V_new
            
            # 打印进度
            if (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration+1}: 最大变化={max_diff:.2e}")
        
        return {self.states[i]: V[i] for i in range(self.n_states)}, V_history
    
    def _bellman_expectation(self, s, policy_s, V):
        """
        贝尔曼期望方程
        
        V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
        
        Args:
            s: 当前状态索引
            policy_s: 当前状态的策略分布 [π(a0|s), π(a1|s), ...]
            V: 当前的价值函数
        """
        value = 0
        
        for a in range(self.n_actions):
            prob_a = policy_s[a]
            if prob_a > 0:
                # 计算即时奖励期望
                expected_reward = np.sum(self.P[s, a] * self.R[s, a])
                
                # 计算未来奖励期望
                expected_future = np.sum(self.P[s, a] * V)
                
                # 贝尔曼方程
                value += prob_a * (expected_reward + self.gamma * expected_future)
        
        return value
    
    def _matrix_policy_evaluation(self, policy):
        """
        矩阵法策略评估（解析解）
        
        V^π = (I - γP^π)^{-1} R^π
        """
        policy_mat = self._policy_to_matrix(policy)
        
        # 构建转移矩阵 P^π
        P_pi = np.zeros((self.n_states, self.n_states))
        R_pi = np.zeros(self.n_states)
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_a = policy_mat[s][a]
                if prob_a > 0:
                    P_pi[s] += prob_a * self.P[s, a]
                    R_pi[s] += prob_a * np.sum(self.P[s, a] * self.R[s, a])
        
        # 解析求解
        I = np.eye(self.n_states)
        V = np.linalg.inv(I - self.gamma * P_pi) @ R_pi
        
        return {self.states[i]: V[i] for i in range(self.n_states)}, [V]
    
    def _monte_carlo_policy_evaluation(self, policy, num_episodes=1000, max_steps=100):
        """
        蒙特卡洛策略评估
        
        通过采样轨迹估计价值函数
        """
        # 将策略转换为函数形式
        policy_func = self._policy_to_function(policy)
        
        returns = {state: [] for state in self.states}
        
        print(f"\n蒙特卡洛策略评估（{num_episodes}个episode）...")
        print("-" * 60)
        
        for episode in range(num_episodes):
            # 生成一个episode
            episode_data = self._generate_episode(policy_func, max_steps)
            
            # 计算每个状态的回报
            G = 0
            visited = set()
            
            for t in range(len(episode_data)-1, -1, -1):
                state, reward, _ = episode_data[t]
                G = reward + self.gamma * G
                
                # First-visit MC
                if state not in visited:
                    returns[state].append(G)
                    visited.add(state)
            
            # 打印进度
            if (episode + 1) % 200 == 0:
                print(f"已完成 {episode+1}/{num_episodes} episodes")
        
        # 计算平均值
        V = {state: np.mean(returns[state]) for state in self.states if returns[state]}
        
        return V, None
    
    def _generate_episode(self, policy_func, max_steps):
        """生成一个episode"""
        episode = []
        state_idx = np.random.randint(self.n_states)
        state = self.states[state_idx]
        
        for _ in range(max_steps):
            # 根据策略选择动作
            action = policy_func(state)
            action_idx = self.actions.index(action)
            state_idx = self.states.index(state)
            
            # 采样下一个状态
            probs = self.P[state_idx, action_idx]
            next_idx = np.random.choice(self.n_states, p=probs)
            next_state = self.states[next_idx]
            
            # 获取奖励
            reward = self.R[state_idx, action_idx, next_idx]
            
            episode.append((state, reward, action))
            state = next_state
            
            # 检查是否终止（可选）
            if np.random.random() < 0.1:  # 10%概率终止
                break
        
        return episode
    
    def _policy_to_matrix(self, policy):
        """将策略字典转换为矩阵"""
        policy_mat = np.zeros((self.n_states, self.n_actions))
        for (s, a), prob in policy.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            policy_mat[s_idx, a_idx] = prob
        return policy_mat
    
    def _policy_to_function(self, policy):
        """将策略字典转换为函数"""
        def policy_func(state):
            state_probs = {}
            for (s, a), prob in policy.items():
                if s == state:
                    state_probs[a] = prob
            
            # 按概率选择动作
            actions = list(state_probs.keys())
            probs = list(state_probs.values())
            return np.random.choice(actions, p=probs)
        
        return policy_func
    
    def visualize_value_function(self, V, title="状态价值函数"):
        """可视化价值函数"""
        states = list(V.keys())
        values = list(V.values())
        
        plt.figure(figsize=(12, 6))
        
        # 条形图
        plt.subplot(1, 2, 1)
        bars = plt.bar(states, values, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('状态')
        plt.ylabel('价值 V(s)')
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 热力图
        plt.subplot(1, 2, 2)
        if len(states) <= 16:  # 适合网格世界
            grid_size = int(np.sqrt(len(states)))
            value_grid = np.array(values).reshape(grid_size, grid_size)
            im = plt.imshow(value_grid, cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(im, label='价值')
            plt.title('价值热力图', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for i in range(grid_size):
                for j in range(grid_size):
                    plt.text(j, i, f'{value_grid[i, j]:.1f}',
                            ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_convergence(self, V_history):
        """可视化收敛过程"""
        V_history = np.array(V_history)
        
        plt.figure(figsize=(12, 5))
        
        # 价值收敛曲线
        plt.subplot(1, 2, 1)
        for i in range(min(5, self.n_states)):
            plt.plot(V_history[:, i], label=f'状态 {self.states[i]}')
        plt.xlabel('迭代次数')
        plt.ylabel('价值 V(s)')
        plt.title('价值函数收敛过程', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最大变化曲线
        plt.subplot(1, 2, 2)
        changes = [np.max(np.abs(V_history[i] - V_history[i-1])) 
                   for i in range(1, len(V_history))]
        plt.semilogy(changes)
        plt.xlabel('迭代次数')
        plt.ylabel('最大变化 (对数尺度)')
        plt.title('收敛速度', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ==================== 2. 示例：网格世界策略评估 ====================

def example_grid_world():
    """网格世界示例"""
    print("=" * 70)
    print("示例1：4x4网格世界策略评估")
    print("=" * 70)
    
    # 定义状态（16个格子，0-15）
    states = [f'({i},{j})' for i in range(4) for j in range(4)]
    
    # 定义动作
    actions = ['上', '下', '左', '右']
    
    # 构建转移概率（确定性环境）
    transition_prob = {}
    reward_func = {}
    
    for i in range(4):
        for j in range(4):
            state = f'({i},{j})'
            state_idx = i * 4 + j
            
            # 定义目标状态（右下角）
            is_goal = (i == 3 and j == 3)
            
            for action in actions:
                # 计算下一个状态
                ni, nj = i, j
                if action == '上' and i > 0:
                    ni = i - 1
                elif action == '下' and i < 3:
                    ni = i + 1
                elif action == '左' and j > 0:
                    nj = j - 1
                elif action == '右' and j < 3:
                    nj = j + 1
                
                next_state = f'({ni},{nj})'
                transition_prob[(state, action, next_state)] = 1.0
                
                # 奖励：到达目标获得1，否则-0.1
                if is_goal:
                    reward = 0
                elif next_state == '(3,3)':
                    reward = 1
                else:
                    reward = -0.1
                
                reward_func[(state, action, next_state)] = reward
    
    # 定义随机策略（均匀随机）
    policy = {}
    for state in states:
        for action in actions:
            policy[(state, action)] = 1.0 / len(actions)
    
    # 创建评估器
    evaluator = PolicyEvaluator(states, actions, transition_prob, reward_func, gamma=0.9)
    
    # 执行策略评估
    V, V_history = evaluator.policy_evaluation(policy, method='iterative')
    
    print("\n策略评估结果（随机策略下的状态价值）：")
    print("-" * 60)
    for i in range(4):
        row = []
        for j in range(4):
            state = f'({i},{j})'
            row.append(f"{V[state]:6.3f}")
        print("  " + "  ".join(row))
    
    # 可视化
    evaluator.visualize_value_function(V, "随机策略价值函数")
    evaluator.visualize_convergence(V_history)


# ==================== 3. 示例：冰湖环境策略评估 ====================

def example_frozen_lake():
    """冰湖环境示例"""
    print("\n" + "=" * 70)
    print("示例2：FrozenLake环境策略评估")
    print("=" * 70)
    
    # 创建环境
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    states = [str(i) for i in range(16)]
    actions = ['左', '下', '右', '上']
    
    # 构建转移概率和奖励（从环境中提取）
    transition_prob = {}
    reward_func = {}
    
    for s in range(16):
        for a in range(4):
            for prob, s_next, reward, terminated in env.unwrapped.P[s][a]:
                if prob > 0:
                    state = states[s]
                    next_state = states[s_next]
                    transition_prob[(state, actions[a], next_state)] = prob
                    reward_func[(state, actions[a], next_state)] = reward
    
    # 定义策略（偏向向右和向下）
    policy = {}
    for state in states:
        # 偏向向目标移动
        state_idx = int(state)
        if state_idx % 4 < 3:  # 不是最右边
            policy[(state, '右')] = 0.5
        else:
            policy[(state, '右')] = 0.1
        
        if state_idx // 4 < 3:  # 不是最下边
            policy[(state, '下')] = 0.4
        else:
            policy[(state, '下')] = 0.1
        
        policy[(state, '左')] = 0.1
        policy[(state, '上')] = 0.1
        
        # 归一化
        total = sum(policy[(state, a)] for a in actions)
        for a in actions:
            policy[(state, a)] /= total
    
    # 创建评估器
    evaluator = PolicyEvaluator(states, actions, transition_prob, reward_func, gamma=0.95)
    
    # 执行策略评估
    V, V_history = evaluator.policy_evaluation(policy, method='iterative')
    
    print("\n策略评估结果（偏向目标的策略）：")
    print("-" * 60)
    for i in range(4):
        row = []
        for j in range(4):
            state = str(i * 4 + j)
            row.append(f"{V[state]:6.3f}")
        print("  " + "  ".join(row))
    
    evaluator.visualize_value_function(V, "FrozenLake策略价值函数")


# ==================== 4. 示例：小车爬坡策略评估 ====================

def example_mountain_car():
    """小车爬坡环境（离散化）"""
    print("\n" + "=" * 70)
    print("示例3：MountainCar离散化策略评估")
    print("=" * 70)
    
    # 离散化状态空间
    position_bins = 20
    velocity_bins = 20
    
    states = []
    for p in range(position_bins):
        for v in range(velocity_bins):
            states.append(f'p{p}_v{v}')
    
    # 简化的状态
    states = states[:100]  # 使用前100个状态
    
    actions = ['左', '不动', '右']
    
    # 构建简化的转移概率
    transition_prob = {}
    reward_func = {}
    
    for state in states:
        for action in actions:
            # 简化：80%概率成功，20%概率随机
            for next_state in np.random.choice(states, min(3, len(states)), replace=False):
                prob = 0.8 if next_state == state else 0.1
                transition_prob[(state, action, next_state)] = prob
                reward_func[(state, action, next_state)] = -1
    
    # 随机策略
    policy = {}
    for state in states:
        for action in actions:
            policy[(state, action)] = 1.0 / len(actions)
    
    # 创建评估器
    evaluator = PolicyEvaluator(states, actions, transition_prob, reward_func, 
                                gamma=0.99, theta=1e-4, max_iter=500)
    
    # 执行策略评估
    V, V_history = evaluator.policy_evaluation(policy, method='iterative')
    
    print(f"\n策略评估完成，共评估 {len(V)} 个状态")
    print(f"价值范围: {min(V.values()):.3f} ~ {max(V.values()):.3f}")
    print(f"平均价值: {np.mean(list(V.values())):.3f}")


# ==================== 5. 比较不同评估方法 ====================

def compare_methods():
    """比较不同策略评估方法的性能"""
    print("\n" + "=" * 70)
    print("比较不同策略评估方法")
    print("=" * 70)
    
    # 简单环境
    states = ['A', 'B', 'C']
    actions = ['a1', 'a2']
    
    transition_prob = {
        ('A', 'a1', 'A'): 0.5, ('A', 'a1', 'B'): 0.5,
        ('A', 'a2', 'B'): 0.7, ('A', 'a2', 'C'): 0.3,
        ('B', 'a1', 'A'): 0.3, ('B', 'a1', 'B'): 0.7,
        ('B', 'a2', 'C'): 1.0,
        ('C', 'a1', 'C'): 1.0,
        ('C', 'a2', 'C'): 1.0,
    }
    
    reward_func = {
        ('A', 'a1', 'A'): 0, ('A', 'a1', 'B'): 1,
        ('A', 'a2', 'B'): 1, ('A', 'a2', 'C'): 2,
        ('B', 'a1', 'A'): 0, ('B', 'a1', 'B'): 1,
        ('B', 'a2', 'C'): 2,
        ('C', 'a1', 'C'): 0,
        ('C', 'a2', 'C'): 0,
    }
    
    # 均匀随机策略
    policy = {}
    for state in states:
        for action in actions:
            policy[(state, action)] = 0.5
    
    evaluator = PolicyEvaluator(states, actions, transition_prob, reward_func, gamma=0.9)
    
    # 方法1：迭代法
    print("\n1. 迭代策略评估：")
    V_iter, _ = evaluator.policy_evaluation(policy, method='iterative')
    for state, value in V_iter.items():
        print(f"   {state}: {value:.4f}")
    
    # 方法2：矩阵法
    print("\n2. 矩阵策略评估（解析解）：")
    V_mat, _ = evaluator.policy_evaluation(policy, method='matrix')
    for state, value in V_mat.items():
        print(f"   {state}: {value:.4f}")
    
    # 方法3：蒙特卡洛
    print("\n3. 蒙特卡洛策略评估：")
    V_mc, _ = evaluator.policy_evaluation(policy, method='monte_carlo', num_episodes=5000)
    for state, value in V_mc.items():
        print(f"   {state}: {value:.4f}")
    
    # 误差分析
    print("\n误差分析（以矩阵法为基准）：")
    for state in states:
        error = abs(V_iter[state] - V_mat[state])
        mc_error = abs(V_mc.get(state, 0) - V_mat[state])
        print(f"   {state}: 迭代法误差={error:.6f}, MC误差={mc_error:.6f}")


# ==================== 6. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("策略评估算法 (Policy Evaluation) 完整实现")
    print("=" * 70)
    
    # 核心公式说明
    print("\n" + "-" * 70)
    print("核心公式:")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  贝尔曼期望方程（策略评估基础）:                                  │
    │                                                                  │
    │  V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]       │
    │                                                                  │
    │  迭代更新公式:                                                   │
    │                                                                  │
    │  V_{k+1}(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]   │
    │                                                                  │
    │  收敛条件:                                                       │
    │                                                                  │
    │  max_s |V_{k+1}(s) - V_k(s)| < θ                                │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 运行示例
    example_grid_world()
    example_frozen_lake()
    example_mountain_car()
    compare_methods()
    
    print("\n" + "=" * 70)
    print("策略评估算法总结")
    print("=" * 70)
    print("""
    ┌─────────────┬──────────────┬─────────────┬────────────────────┐
    │   方法      │   精度       │   速度      │      适用场景       │
    ├─────────────┼──────────────┼─────────────┼────────────────────┤
    │ 迭代法      │ 高(可调)     │ 中等        │ 大规模状态空间     │
    ├─────────────┼──────────────┼─────────────┼────────────────────┤
    │ 矩阵法      │ 精确解       │ 慢(O(n^3))  │ 小规模状态空间     │
    ├─────────────┼──────────────┼─────────────┼────────────────────┤
    │ 蒙特卡洛    │ 中等         │ 快(采样)    │ 模型未知环境       │
    └─────────────┴──────────────┴─────────────┴────────────────────┘
    
    关键要点:
    1. 策略评估是策略迭代的基础
    2. 需要知道环境模型（转移概率和奖励）
    3. 收敛速度取决于折扣因子γ
    4. 可用于评估任意策略的好坏
    """)