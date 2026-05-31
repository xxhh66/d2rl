"""
markov_decision_process.py
马尔科夫决策过程 (Markov Decision Process)
定义：<S, A, P, R, \gamma> - 加入决策（动作）
核心算法：策略迭代、价值迭代
"""

import numpy as np
import matplotlib.pyplot as plt
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MarkovDecisionProcess:
    """马尔科夫决策过程：带决策的强化学习环境"""
    
    def __init__(self, states, actions, transition_prob, reward_func, gamma=0.9):
        """
        初始化MDP
        
        Args:
            states: 状态列表
            actions: 动作列表
            transition_prob: 转移概率字典 {(s,a,s'): prob}
            reward_func: 奖励函数 {(s,a): reward} 或 {(s,a,s'): reward}
            gamma: 折扣因子
        """
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.gamma = gamma
        
        # 构建转移概率矩阵 P[s][a][s']
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for (s, a, s_next), prob in transition_prob.items():
            s_idx = states.index(s)
            a_idx = actions.index(a)
            s_next_idx = states.index(s_next)
            self.P[s_idx, a_idx, s_next_idx] = prob
        
        # 构建奖励函数 R[s][a]
        self.R = np.zeros((self.n_states, self.n_actions))
        for key, reward in reward_func.items():
            if len(key) == 2:  # (s, a)
                s_idx = states.index(key[0])
                a_idx = actions.index(key[1])
                self.R[s_idx, a_idx] = reward
            elif len(key) == 3:  # (s, a, s')
                s_idx = states.index(key[0])
                a_idx = actions.index(key[1])
                self.R[s_idx, a_idx] += reward * transition_prob[key]
    
    def policy_evaluation(self, policy, max_iter=1000, tol=1e-6):
        """
        策略评估：计算给定策略的价值函数
        
        Args:
            policy: 字典 {(s, a): probability}
        """
        # 构建策略矩阵
        policy_mat = np.zeros((self.n_states, self.n_actions))
        for (s, a), prob in policy.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            policy_mat[s_idx, a_idx] = prob
        
        V = np.zeros(self.n_states)
        
        for _ in range(max_iter):
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    prob_a = policy_mat[s, a]
                    if prob_a > 0:
                        # Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) V(s')
                        q_value = self.R[s, a]
                        q_value += self.gamma * np.sum(self.P[s, a] * V)
                        V_new[s] += prob_a * q_value
            
            if np.linalg.norm(V_new - V) < tol:
                break
            V = V_new
        
        return {self.states[i]: V[i] for i in range(self.n_states)}
    
    def policy_improvement(self, V):
        """策略改进：贪心提升"""
        new_policy = {}
        
        for s_idx, state in enumerate(self.states):
            # 计算每个动作的价值
            action_values = []
            for a_idx, action in enumerate(self.actions):
                q_value = self.R[s_idx, a_idx]
                q_value += self.gamma * np.sum(self.P[s_idx, a_idx] * list(V.values()))
                action_values.append(q_value)
            
            # 选择最优动作
            best_a_idx = np.argmax(action_values)
            new_policy[(state, self.actions[best_a_idx])] = 1.0
        
        return new_policy
    
    def policy_iteration(self, max_iter=100):
        """策略迭代算法"""
        # 初始化均匀随机策略
        policy = {}
        for s in self.states:
            for a in self.actions:
                policy[(s, a)] = 1.0 / self.n_actions
        
        for i in range(max_iter):
            # 策略评估
            V = self.policy_evaluation(policy)
            
            # 策略改进
            new_policy = self.policy_improvement(V)
            
            # 检查收敛
            if new_policy == policy:
                print(f"策略迭代收敛于第 {i+1} 步")
                break
            
            policy = new_policy
        
        return policy, V
    
    def value_iteration(self, max_iter=1000, tol=1e-6):
        """价值迭代算法"""
        V = np.zeros(self.n_states)
        
        for i in range(max_iter):
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                # 计算每个动作的价值，取最大值
                action_values = []
                for a in range(self.n_actions):
                    q_value = self.R[s, a]
                    q_value += self.gamma * np.sum(self.P[s, a] * V)
                    action_values.append(q_value)
                
                V_new[s] = max(action_values)
            
            if np.linalg.norm(V_new - V) < tol:
                print(f"价值迭代收敛于第 {i+1} 步")
                break
            
            V = V_new
        
        # 提取最优策略
        optimal_policy = {}
        for s_idx, state in enumerate(self.states):
            action_values = []
            for a_idx, action in enumerate(self.actions):
                q_value = self.R[s_idx, a_idx]
                q_value += self.gamma * np.sum(self.P[s_idx, a_idx] * V)
                action_values.append(q_value)
            
            best_a_idx = np.argmax(action_values)
            optimal_policy[(state, self.actions[best_a_idx])] = 1.0
        
        optimal_values = {self.states[i]: V[i] for i in range(self.n_states)}
        return optimal_policy, optimal_values
    
    def visualize_policy(self, policy):
        """可视化策略"""
        policy_mat = np.zeros((self.n_states, self.n_actions))
        for (s, a), prob in policy.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            policy_mat[s_idx, a_idx] = prob
        
        plt.figure(figsize=(10, 6))
        plt.imshow(policy_mat, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(label='策略概率')
        plt.title('最优策略可视化', fontsize=14, fontweight='bold')
        plt.xlabel('动作')
        plt.ylabel('状态')
        plt.xticks(range(self.n_actions), self.actions)
        plt.yticks(range(self.n_states), self.states)
        
        # 添加数值标签
        for i in range(self.n_states):
            for j in range(self.n_actions):
                plt.text(j, i, f'{policy_mat[i, j]:.2f}',
                        ha='center', va='center',
                        color='black' if policy_mat[i, j] < 0.5 else 'white')
        
        plt.tight_layout()
        plt.show()


# ==================== 示例运行 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("马尔科夫决策过程示例 - 网格世界")
    print("=" * 50)
    
    # 定义状态和动作
    states = ['A', 'B', 'C', '目标']
    actions = ['左', '右', '上', '下']
    
    # 转移概率（确定性）
    transition = {
        ('A', '右', 'B'): 1.0,
        ('A', '下', 'C'): 1.0,
        ('B', '左', 'A'): 1.0,
        ('B', '右', '目标'): 1.0,
        ('C', '上', 'A'): 1.0,
        ('C', '右', '目标'): 1.0,
        ('目标', '左', '目标'): 1.0,
        ('目标', '右', '目标'): 1.0,
        ('目标', '上', '目标'): 1.0,
        ('目标', '下', '目标'): 1.0,
    }
    
    # 奖励函数
    rewards = {
        ('A', '右'): -1, ('A', '下'): -1,
        ('B', '左'): -1, ('B', '右'): 10,
        ('C', '上'): -1, ('C', '右'): 10,
        ('目标', '左'): 0, ('目标', '右'): 0,
        ('目标', '上'): 0, ('目标', '下'): 0,
    }
    
    # 创建MDP
    mdp = MarkovDecisionProcess(states, actions, transition, rewards, gamma=0.9)
    
    # 策略迭代
    print("\n1. 策略迭代算法：")
    policy, values = mdp.policy_iteration()
    
    print("\n最优策略：")
    for state in states:
        for action in actions:
            if policy.get((state, action), 0) > 0:
                print(f"  {state} -> {action}")
    
    print("\n状态价值：")
    for state, value in values.items():
        print(f"  {state}: {value:.3f}")
    
    # 价值迭代
    print("\n2. 价值迭代算法：")
    policy_vi, values_vi = mdp.value_iteration()
    
    print("\n最优策略：")
    for state in states:
        for action in actions:
            if policy_vi.get((state, action), 0) > 0:
                print(f"  {state} -> {action}")
    
    print("\n最优状态价值：")
    for state, value in values_vi.items():
        print(f"  {state}: {value:.3f}")
    
    # 可视化策略
    mdp.visualize_policy(policy_vi)