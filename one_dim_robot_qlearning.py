import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========================
# 設定パラメータ
# ========================
class Config:
    ENV_RANGE = [-50, 50]
    ACTIONS = [0, 1]     # 0: 左, 1: 右
    NUM_EPISODES = 100000
    MAX_STEPS = 50
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON = 0.9  # 学習中の探索率

# ========================
# ユーティリティ関数
# ========================
def valid_positions(env_range):
    return list(range(env_range[0], env_range[1] + 1))

def pos_to_idx(pos, env_range):
    return pos - env_range[0]

# ========================
# 状態空間の変換関数
# ========================
class StateIndexer:
    def __init__(self, env_range):
        self.env_range = env_range
        self.all_pos = valid_positions(env_range)
        self.n_pos = len(self.all_pos)

    def state_to_index(self, position, goal):
        pos_idx = pos_to_idx(position, self.env_range)
        goal_idx = pos_to_idx(goal, self.env_range)
        return goal_idx * self.n_pos + pos_idx

    def index_to_state(self, state_index):
        goal_idx = state_index // self.n_pos
        pos_idx = state_index % self.n_pos
        return (self.all_pos[pos_idx], self.all_pos[goal_idx])

# ========================
# 乱数生成
# ========================
def random_position(env_range):
    return random.randint(env_range[0], env_range[1])

# ========================
# 環境クラス
# ========================
class OneDimRobotEnv:
    def __init__(self, config):
        self.env_range = config.ENV_RANGE
        self.robot_position = random_position(self.env_range)
        self.goal_position = random_position(self.env_range)
        while self.goal_position == self.robot_position:
            self.goal_position = random_position(self.env_range)

    def reset(self):
        self.robot_position = random_position(self.env_range)
        self.goal_position = random_position(self.env_range)
        while self.goal_position == self.robot_position:
            self.goal_position = random_position(self.env_range)
        return (self.robot_position, self.goal_position)

    def step(self, action):
        current_distance = abs(self.robot_position - self.goal_position)

        if action == 0:
            next_pos = max(self.env_range[0], self.robot_position - 1)
        else:
            next_pos = min(self.env_range[1], self.robot_position + 1)

        self.robot_position = next_pos
        next_distance = abs(self.robot_position - self.goal_position)

        done = False
        reward = 0.0
        if self.robot_position == self.goal_position:
            reward = 1.0
            done = True
        else:
            reward = 0.1 if next_distance < current_distance else -0.1

        return (self.robot_position, self.goal_position), reward, done

# ========================
# Q学習関数
# ========================
def q_learning(env, state_indexer, config):
    n_states = state_indexer.n_pos ** 2
    n_actions = len(config.ACTIONS)
    Q_table = np.zeros((n_states, n_actions))

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        done = False

        for step in range(config.MAX_STEPS):
            s_index = state_indexer.state_to_index(*state)

            if np.random.rand() < config.EPSILON:
                action = random.choice(config.ACTIONS)
            else:
                action = np.argmax(Q_table[s_index])

            next_state, reward, done = env.step(action)
            s_next_index = state_indexer.state_to_index(*next_state)

            current_q = Q_table[s_index, action]
            target = reward + config.GAMMA * np.max(Q_table[s_next_index]) if not done else reward
            Q_table[s_index, action] += config.ALPHA * (target - current_q)

            state = next_state
            if done:
                break

        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode+1} done.")

    return Q_table

# ========================
# 連続シミュレーションとアニメーション
# ========================
def run_continuous_simulation(env, Q_table, state_indexer, total_steps=50):
    path = []
    goal_history = []

    for _ in range(total_steps):
        path.append(env.robot_position)
        goal_history.append(env.goal_position)

        s_index = state_indexer.state_to_index(env.robot_position, env.goal_position)
        action = np.argmax(Q_table[s_index])
        _, _, done = env.step(action)

        if done:
            new_goal = random_position(env.env_range)
            while new_goal == env.robot_position:
                new_goal = random_position(env.env_range)
            env.goal_position = new_goal

    path.append(env.robot_position)
    goal_history.append(env.goal_position)
    return path, goal_history

def animate_continuous_motion(path, goal_history, env_range):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(env_range[0] - 1, env_range[1] + 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Continuous Simulation after Q-learning")
    ax.set_xlabel("Position")
    ax.get_yaxis().set_visible(False)

    robot_point, = ax.plot([], [], 'bo', markersize=8, label="Robot")
    goal_point, = ax.plot([], [], 'rx', markersize=8, label="Goal")
    ax.legend()

    def init():
        robot_point.set_data([], [])
        goal_point.set_data([], [])
        return robot_point, goal_point

    def update(frame):
        robot_point.set_data([path[frame]], [0])
        goal_point.set_data([goal_history[frame]], [0])
        return robot_point, goal_point

    ani = FuncAnimation(fig, update, frames=len(path), init_func=init, interval=10, blit=True)
    plt.show()

# ========================
# メイン実行部
# ========================
if __name__ == "__main__":
    config = Config()
    state_indexer = StateIndexer(config.ENV_RANGE)
    env = OneDimRobotEnv(config)
    Q_table = q_learning(env, state_indexer, config)

    env_test = OneDimRobotEnv(config)
    env_test.robot_position = random_position(config.ENV_RANGE)
    env_test.goal_position = random_position(config.ENV_RANGE)
    path, goal_hist = run_continuous_simulation(env_test, Q_table, state_indexer, total_steps=10000)
    animate_continuous_motion(path, goal_hist, config.ENV_RANGE)
