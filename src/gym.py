import gymnasium as gym
import pygame
import sys
import numpy as np

def draw_step_number(screen, step_num, width, height):
    """在屏幕左上角绘制步数"""
    try:
        # 使用系统字体，确保可用
        font = pygame.font.SysFont('arial', 36, bold=True)
        # 添加黑色背景框，确保文字可见
        text = font.render(f'Step: {step_num}', True, (255, 255, 0), (0, 0, 0))
        screen.blit(text, (10, 10))
    except Exception as e:
        print(f"绘图错误: {e}")

# 使用gym.make()方法创建一个名为'CartPole-v1'的环境实例。
# 'CartPole-v1'是经典控制问题的环境，智能体需要控制小车上的杆子保持直立
# render_mode='human' 表示以人类可读的方式渲染环境，通常会弹出一个窗口展示环境画面
env = gym.make("CartPole-v1", render_mode="rgb_array")
# 调用环境的reset()方法，将环境重置为初始状态。
# 该方法返回两个值：
# - observation: 初始观察值，代表智能体对环境当前状态的观测，在'CartPole-v1'中可能是小车位置、杆子角度等信息
# - info: 包含一些辅助信息，如环境的内部状态等，这些信息通常用于调试或进一步分析环境，但在基本的强化学习算法中可能不会立即用到
observation, info = env.reset()

# 获取第一帧来确定窗口大小
frame = env.render()
height, width = frame.shape[:2]

# 初始化 Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('CartPole - Step Display')
clock = pygame.time.Clock()

# print("开始运行，将在窗口左上角显示 Step 编号...")

for step in range(1000):
    # 处理退出事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            sys.exit()
    
     # 从环境的动作空间中随机采样一个动作。在'CartPole-v1'环境中，动作空间可能是离散的，例如0代表向左移动小车，1代表向右移动小车
    action = env.action_space.sample()
    
    # 让环境执行选择的动作action，并返回以下信息：
    # - observation: 执行动作后，智能体对环境新状态的观测
    # - reward: 执行该动作后获得的奖励值，在'CartPole-v1'中，如果杆子保持直立，可能会获得正奖励，杆子倒下则可能获得负奖励
    # - terminated: 一个布尔值，指示该episode（情节）是否因为达到目标或失败而结束，例如在'CartPole-v1'中杆子倒下可能导致episode结束
    # - truncated: 一个布尔值，指示该episode是否因为其他原因（如超出最大步数限制）而提前截断
    # - info: 同样包含一些辅助信息，如环境的额外状态细节等
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 获取当前帧
    frame = env.render()
    
    # 将 numpy 数组转换为 Pygame 表面
    # 注意：CartPole 返回的是 RGB 数组，形状为 (height, width, 3)
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    
    # 显示环境画面
    screen.blit(frame_surface, (0, 0))
    
    # 绘制步数（在环境画面上方）
    draw_step_number(screen, step, width, height)
    
    # 更新显示
    pygame.display.flip()
    
    # 控制帧率，让动画可见
    clock.tick(60)  # 60 FPS
    
    # # 在控制台也打印步数（调试用）
    # if step % 50 == 0:
    #     print(f"当前步数: {step}")
    
    # 如果 episode 结束，重置环境
    if terminated or truncated:
        print(f"\nEpisode 结束于步数: {step}，重置环境\n")
        observation, info = env.reset()

# 关闭环境
env.close()
pygame.quit()
print("程序结束")