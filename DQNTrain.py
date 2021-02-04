import matplotlib.pyplot as plt
from DrawStatistics import plot_broken_line
from Runner import Runner
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot
from Maze import Maze
import os
from tqdm.auto import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复载入lib文件


def train_by_dqn_robot(times, maze_size=5):
    print("start times:", times)

    maze = Maze(maze_size=maze_size)

    """choose Keras or Torch version"""
    robot = KerasRobot(maze=maze)
    # robot = TorchRobot(maze=maze)
    robot.memory.build_full_view(maze=maze)

    """training by runner"""
    runner = Runner(robot=robot)
    runner.run_training(15, 75)

    """Test Robot"""
    robot.reset()
    for _ in range(25):
        a, r = robot.test_update()
        if r < -20:
            print("SUCCESSFUL!", "| TIMES:", times, )
            break
        # print(a, r, end=",")


if __name__ == "__main__":
    # tf 2.1
    generate_times = 5  # 测试次数，每次测试都会重新生成迷宫，并从零开始训练机器人
    for time in range(generate_times):
        train_by_dqn_robot(time, maze_size=5)
