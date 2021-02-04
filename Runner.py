import numpy as np

from tqdm.auto import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt


class Runner(object):
    def __init__(self, robot):
        self.maze = robot.maze
        self.robot = robot

        self.train_robot_record = []
        self.train_robot_statics = {
            'success': [],
            'reward': [],
            'times': [],
        }
        self.test_robot_statics = {
            'success': [],
            'reward': [],
            'times': [],
        }

        self.display_direction = False

    def add_statics(self, accumulated_reward, run_times):

        self.train_robot_statics['reward'].append(accumulated_reward)
        self.train_robot_statics['times'].append(run_times)

        if self.maze.robot['loc'] == self.maze.destination:
            self.train_robot_statics['success'].append(1)
        else:
            self.train_robot_statics['success'].append(0)

    def run_training(self, training_epoch, training_per_epoch=150):
        for e in range(training_epoch):
            accumulated_reward = 0
            run_times = 0
            for i in range(training_per_epoch):

                current_record = {
                    'id': [e, i],
                    'success': False,
                    'state': self.maze.sense_robot(),
                }

                if current_record['state'] == self.maze.destination:
                    current_record['success'] = True
                    self.train_robot_record.append(current_record)
                    break

                action, reward = self.robot.train_update()
                current_record['action'] = action
                current_record['reward'] = reward
                self.train_robot_record.append(current_record)

                run_times += 1
                accumulated_reward += reward

            self.add_statics(accumulated_reward, run_times)

            self.robot.reset()

    def run_testing(self):
        height, width, _ = self.maze.maze_data.shape
        testing_per_epoch = int(height * width * 0.85)

        accumulated_reward = 0.
        run_times = 0

        for i in range(testing_per_epoch):
            run_times += 1
            _, reward = self.robot.test_update()
            accumulated_reward += reward
            if self.maze.sense_robot() == self.maze.destination:
                break
        self.add_statics(accumulated_reward, run_times)

    def __init_gif(self):
        self.maze.draw_maze()
        fig = plt.gcf()
        ax = plt.gca()
        robot = plt.Circle((0, 0), 0.5, color="red")
        x, y = self.maze.robot['loc'][0] + 0.5, self.maze.robot['loc'][1] + 0.5
        robot.center = (y, x)
        ax.add_patch(robot)

        text_epoch = ax.text(
            0, -0.1,
            '',
            fontsize=20,
            horizontalalignment='left',
            verticalalignment="bottom"
        )

        text_step = ax.text(
            self.maze.maze_size, -0.1,
            '',
            fontsize=20,
            horizontalalignment='right',
            verticalalignment="bottom",
        )
        return fig, ax, robot, text_epoch, text_step

    def generate_gif(self, filename):
        fig, ax, robot, text_epoch, text_step = self.__init_gif()
        p_bar = tqdm(
            total=len(self.train_robot_record),
            desc="正在将训练过程转换为gif图, 请耐心等候...",
        )

        def update(record):
            x, y = record['state'][0] + 0.5, record['state'][1] + 0.5
            robot.center = (y, x)

            text_epoch.set_text("epoch:" + str(record['id'][0]))
            text_step.set_text("step:" + str(record['id'][1]))

            p_bar.update(1)
            return robot,

        def init(): pass  # do nothing


        import matplotlib.animation as animation
        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=self.train_robot_record,
            interval=200,
            blit=False,
            save_count=0,
        )

        # To save the animation, use e.g.
        ani.save(filename, writer='pillow')
        plt.close()

    def plot_results(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.title("Success Times")
        plt.plot(np.cumsum(self.train_robot_statics['success']))
        plt.subplot(132)
        plt.title("Accumulated Rewards")
        plt.plot(np.array(self.train_robot_statics['reward']))
        plt.subplot(133)
        plt.title("Runing Times per Epoch")
        plt.plot(np.array(self.train_robot_statics['times']))
        plt.show()
