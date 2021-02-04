import random
from collections import namedtuple
import numpy as np
import copy
from torch.utils.data import DataLoader
from Maze import Maze


class ReplayDataSet(object):
    def __init__(self, max_size):
        super(ReplayDataSet, self).__init__()
        self.Row = namedtuple("Row", field_names=["state", "action_index", "reward", "next_state", "is_terminal"])
        self.max_size = max_size
        self.Experience = {}
        self.full_dataset = []

    def add(self, state, action_index, reward, next_state, is_terminal):
        if len(self.Experience) == self.max_size:
            self.Experience.popitem()  # 超越内存最大长度时需要清理空间，具体的删除方式还需改动

        key = (state, action_index)
        if self.Experience.__contains__(key):
            return
        else:
            new_row = self.Row(list(state), action_index, reward, list(next_state), is_terminal)
            self.Experience.update({key: new_row})

    def random_sample(self, batch_size):
        if len(self.Experience) < batch_size:
            print("the amount of experiences is to few")
            return
        else:
            samples = random.sample(list(self.Experience.values()), batch_size)
            state = []
            action_index = []
            reward = []
            next_state = []
            is_terminal = []
            for single_sample in samples:
                state.append(single_sample.state)
                action_index.append([single_sample.action_index])
                reward.append([single_sample.reward])
                next_state.append(single_sample.next_state)
                is_terminal.append([single_sample.is_terminal])
            return np.array(state), np.array(action_index, dtype=np.int8), np.array(reward), np.array(
                next_state), np.array(
                is_terminal, dtype=np.int8)

    def build_full_view(self, maze: Maze):
        """
            金手指，获取迷宫全图视野的数据集
            :param maze: 由Maze类实例化的对象
        """
        maze_copy = copy.deepcopy(maze)
        maze_size = maze_copy.maze_size
        actions = ["u", "r", "d", "l"]
        for i in range(maze_size):
            for j in range(maze_size):
                state = (i, j)
                if state == maze_copy.destination:
                    continue
                for action_index, action in enumerate(actions):
                    maze_copy.robot["loc"] = state
                    reward = maze_copy.move_robot(action)
                    next_state = maze_copy.sense_robot()
                    is_terminal = 1 if next_state == maze_copy.destination or next_state == state else 0
                    self.add(state, action_index, reward, next_state, is_terminal)
        self.full_dataset = list(self.Experience.values())

    def __getitem__(self, item):
        state = self.full_dataset[item].state
        action_index = self.full_dataset[item].action_index
        reward = self.full_dataset[item].reward
        next_state = self.full_dataset[item].next_state
        is_terminal = self.full_dataset[item].is_terminal
        return np.array(state), np.array([action_index], dtype=np.int8), np.array([reward]), np.array(
            next_state), np.array([is_terminal], dtype=np.int8)

    def __len__(self):
        return len(self.Experience)


if __name__ == "__main__":
    memory = ReplayDataSet(1e3)
    maze1 = Maze(5)
    memory.build_full_view(maze1)
    print(len(memory))
    # memory_loader = DataLoader(memory, batch_size=5)
    # for idx, content in enumerate(memory_loader):
    #     print(idx)
    #     print(content)
    #     break
