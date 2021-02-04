from tensorflow import keras
from QRobot import QRobot
import random
from Maze import Maze
import numpy as np

from ReplayDataSet import ReplayDataSet
from keras_py.QNetwork import q_network


class MinDQNRobot(QRobot):
    valid_action = ['u', 'r', 'd', 'l']

    target_model = None
    eval_model = None
    learning_rate = 5e-2
    TAU = 1e-3
    batch_size = 32

    step = 1  # 记录训练的步数

    ''' QLearning parameters'''
    epsilon0 = 0.5  # 初始贪心算法探索概率
    gamma = 0.98  # 公式中的 γ

    EveryUpdate = 1  # the interval of target model's updating

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(MinDQNRobot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 10.,
            "destination": -50.,
            "default": 1.,
        })
        self.maze = maze
        self.maze_size = maze.maze_size

        """build network"""
        self.target_model = None
        self.eval_model = None
        self._build_network()

        """create the memory to store data"""
        max_size = max(self.maze_size ** 2 * 3, 1e3)
        self.memory = ReplayDataSet(max_size=max_size)

    def _build_network(self):
        """build eval model"""
        self.eval_model = q_network(input_shape=(2,), action_size=4)

        """build target model"""
        # self.target_model = keras.models.clone_model(self.eval_model)
        self.target_model = q_network(input_shape=(2,), action_size=4)

        """compile model"""
        opt = keras.optimizers.RMSprop(lr=self.learning_rate)
        self.eval_model.compile(
            optimizer=opt,
            loss='mse',
        )
        self.target_model.compile(
            optimizer=opt,
            loss='mse',
        )

    def _target_replace_op(self):
        """
            Soft update the target model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        # object_weights = np.array([1.0 - self.TAU]) * self.eval_model.get_weights() + np.array(
        #     [self.TAU]) * self.target_model.get_weights()
        # self.target_model.set_weights(object_weights)
        # print("params has changed")

        """all replace"""
        self.target_model.set_weights(self.eval_model.get_weights().copy())

    def _choose_action(self, state):
        state = np.array(state, dtype=np.int16)
        state = np.expand_dims(state, 0)
        if random.random() < self.epsilon:
            action = random.choice(self.valid_action)
        else:
            q_next = self.eval_model.predict(state)
            action = self.valid_action[np.argmin(q_next, axis=1).item()]
        return action

    def _learn(self, batch: int = 16):
        if len(self.memory) < batch:
            print("the memory data is not enough")
            return

        state, action_index, reward, next_state, is_terminal = self.memory.random_sample(batch)
        target_y = self.eval_model.predict(state).copy()
        Q_targets_next = np.min(self.target_model.predict(next_state), axis=1, keepdims=True)

        # action_index, 以及reward + γ* y' 是二维数组，所以要进行np.squeeze压缩操作
        target_y[np.arange(batch, dtype=np.int8), np.squeeze(action_index)] = np.squeeze(
            reward + self.gamma * Q_targets_next * (np.ones_like(is_terminal) - is_terminal))

        """train network in some epoch"""
        loss = self.eval_model.train_on_batch(
            x=state,
            y=target_y,
            reset_metrics=False,
        )
        """update the target network"""
        self._target_replace_op()

        return loss

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        next_state = self.sense_state()
        is_terminal = 1 if next_state == self.maze.destination or next_state == state else 0

        self.memory.add(state, self.valid_action.index(action), reward, next_state, is_terminal)

        """间隔一段时间更新target network权重"""
        if self.step % self.EveryUpdate == 0:
            for _ in range(2):
                self._learn(batch=32)

        """update the step and epsilon"""
        self.step += 1
        self.epsilon = max(0.01, self.epsilon * 0.995)

        return action, reward

    def test_update(self):
        state = self.sense_state()
        state = np.array(state, dtype=np.int32)
        state = np.expand_dims(state, axis=0)
        q_value = self.eval_model.predict(state)
        action_index = np.argmin(q_value, axis=1).item()
        action = self.valid_action[action_index]
        reward = self.maze.move_robot(action)

        return action, reward


if __name__ == "__main__":
    maze_ = Maze(maze_size=5)

    robot = MinDQNRobot(maze=maze_)
