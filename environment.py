import os.path
from math import e
from utils.data_util import clip_and_normalize
from agent.uav import UAV
from agent.task import Task
from agent.trunk import Trunk
import numpy as np
from math import pi
import random
from typing import List


class Environment:
    def __init__(self, n_uav: int, m_task: int, x_max: float, y_max: float, na: int, v_min, v_max):
        # size of the environment
        self.x_max = x_max
        self.y_max = y_max

        # dim of action space and state space
        self.state_dim = 18
        self.na = na
        self.m_task = m_task

        # agent parameters
        self.n_uav = n_uav

        # agent
        self.uav_list = []
        self.task_list = []

        # position of uav and target
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': []}

         #
        self.v_min = v_min
        self.v_max = v_max

        self.tot_completed_workload = []


    def __reset(self, v_max, v_min, h_max, na, m, dt, c_uav, w_uav, E_spec, d_safe, E_fly, init_x, init_y, w_task, R_task, trunk_x, trunk_y, R_trunk, x_max):
        if isinstance(init_x, List) and isinstance(init_y, List):
            self.uav_list = [UAV(init_x[i], 
                                 init_y[i],
                                 random.uniform(-pi, pi), 
                                 random.randint(0, na-1), 
                                 random.uniform(v_min, v_max), 
                                 random.randint(0, m-1), 
                                 x_max, v_max, v_min, h_max, na, m, dt, 
                                 c_uav[i], max(c_uav), w_uav[i], 
                                 E_spec, d_safe, E_fly) for i in range(self.n_uav)]
        elif not isinstance(init_x, List) and not isinstance(init_y, List):
            self.uav_list = [UAV(init_x, 
                                 init_y,
                                 random.uniform(-pi, pi), 
                                 random.randint(0, na-1), 
                                 random.uniform(v_min, v_max),  
                                 random.randint(0, m-1), 
                                 x_max, v_max, v_min, h_max, na, m, dt, 
                                 c_uav[i], max(c_uav), w_uav[i], 
                                 E_spec, d_safe, E_fly) for i in range(self.n_uav)]
        elif isinstance(init_x, List):
            self.uav_list = [UAV(init_x[i], 
                                 init_y,
                                 random.uniform(-pi, pi), 
                                 random.randint(0, na-1), 
                                 random.uniform(v_min, v_max),  
                                 random.randint(0, m-1), 
                                 x_max, v_max, v_min, h_max, na, m, dt, 
                                 c_uav[i], max(c_uav), w_uav[i], 
                                 E_spec, d_safe, E_fly) for i in range(self.n_uav)]
        elif isinstance(init_y, List):
            self.uav_list = [UAV(init_x, 
                                 init_y[i],
                                 random.uniform(-pi, pi), 
                                 random.randint(0, na-1), 
                                 random.uniform(v_min, v_max),  
                                 random.randint(0, m-1), 
                                 x_max, v_max, v_min, h_max, na, m, dt, 
                                 c_uav[i], max(c_uav), w_uav[i], 
                                 E_spec, d_safe, E_fly) for i in range(self.n_uav)]
        else:
            print("wrong init position")

        self.task_list = [Task(random.uniform(200, x_max),
                                random.uniform(200, x_max),
                                w_task[i],
                                R_task,
                                max(w_task)) for i in range(self.m_task)]
        
         # initial Trunik
        self.trunk = Trunk(trunk_x, trunk_y, R_trunk)
        # the initial position of the target is random, having randon headings
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': []}

    def reset(self, config):
        self.__reset(v_max=config["uav"]["v_max"],
                     v_min=config["uav"]["v_min"],
                     h_max=config["uav"]["h_max"],
                     na=config["env"]["na"],
                     m=config["env"]["m_tasks"],
                     dt=config["uav"]["dt"],
                     c_uav=config["uav"]["C_uav"],
                     w_uav=config["uav"]["W_uav"],
                     E_spec=[config["uav"]["P_0"], config["uav"]["U_tip"], config["uav"]["d_0"], config["uav"]["rho"], config["uav"]["s"], config["uav"]["A"]],
                     d_safe=config["uav"]["d_safe"],
                     E_fly=config["uav"]["E_fly"],
                     init_x=[random.uniform(0, 200) for x in range(1, config["env"]["n_uav"]+1)],
                     init_y=[random.uniform(0, 200) for x in range(1, config["env"]["n_uav"]+1)],
                     w_task=config["task"]["w_task"],
                     R_task=config["task"]["R_task"],
                     trunk_x=100,
                     trunk_y=100,
                     R_trunk=100,
                     x_max=config["env"]["x_max"])

    def get_states(self) -> (List['np.ndarray']):
        """
        get the state of the uav_s
        :return: list of np array, each element is a 1-dim array with size of 12
        """
        uav_states = []
        # collect the overall communication and target observation by each uav
        for uav in self.uav_list:
            uav_states.append(uav.get_local_state())
        return uav_states

    def step(self, config, actions):

        # update the position of targets
        for i, uav in enumerate(self.uav_list):
            if actions[i] is not None:
                task_id = actions[i]
            else:
                task_id = uav.task_id

            # assign task
            uav.assign_task(task_id, self.m_task)
            # execute task
            uav.execute_task(self.task_list)
            # return to trunk
            uav.return_to_trunk(self.trunk)

            # observation and communication
            uav.observe_task(self.task_list)
            uav.observe_uav(self.uav_list)

        (reward, completion_workload_reward, task_assignment_reward) = self.calculate_rewards(config)
        
        next_states = self.get_states()


        # trace the position matrix
        target_xs, target_ys = self.__get_all_task_position()
        self.position['all_target_xs'].append(target_xs)
        self.position['all_target_ys'].append(target_ys)
        uav_xs, uav_ys = self.__get_all_uav_position()
        self.position['all_uav_xs'].append(uav_xs)
        self.position['all_uav_ys'].append(uav_ys)

        reward = {
            'rewards': reward,
            'completion_workload_reward': completion_workload_reward,
            'task_assignment_reward': task_assignment_reward
        }

        return next_states, reward

    def __get_all_uav_position(self):
        """
        :return: all the position of the uav through this epoch
        """
        uav_xs = []
        uav_ys = []
        for uav in self.uav_list:
            uav_xs.append(uav.x)
            uav_ys.append(uav.y)
        return uav_xs, uav_ys

    def __get_all_task_position(self):
        """
        :return: all the position of the targets through this epoch
        """
        task_xs = []
        task_ys = []
        for task in self.task_list:
            task_xs.append(task.x)
            task_ys.append(task.y)
        return task_xs, task_ys

    def calculate_rewards(self, config) -> ([float], float, float, float):
        # raw reward first
        completion_workload_rewards = []
        task_assignment_rewards = []
        rewards = []
        tot_completed_workload = 0
        for uav in self.uav_list:
            (completion_workload_reward, task_assignment_reward) = uav.calculate_raw_reward(self.uav_list, self.task_list, self.trunk)

            # completion_workload_reward = clip_and_normalize(completion_workload_reward, 0, sum(config['task']['w_task']), 0)
            # task_assignment_reward = clip_and_normalize(task_assignment_reward, 0, sum(config['task']['w_task']), 0)

            # append
            completion_workload_rewards.append(completion_workload_reward)
            task_assignment_rewards.append(task_assignment_reward)

            tot_completed_workload += completion_workload_reward
            # weights
            uav.raw_reward = (config["uav"]["alpha"] * completion_workload_reward + config["uav"]["beta"] * task_assignment_reward)
            rewards.append(uav.raw_reward)
        self.tot_completed_workload.append(tot_completed_workload)
        return rewards, completion_workload_rewards, task_assignment_rewards

    def save_position(self, save_dir, epoch_i):
        u_xy = np.array([self.position['all_uav_xs'], 
                         self.position['all_uav_ys']]).transpose()
        
        np.savetxt(os.path.join(save_dir, "u_xy", 'u_xy' + str(epoch_i) + '.csv'), 
                    u_xy.reshape(-1,2), delimiter=',', header='x,y', comments='')

    def save_executed_task_num(self, save_dir, executed_tasks):
        executed_task_num = np.array(executed_tasks).reshape(-1, 1)

        np.savetxt(os.path.join(save_dir, "executed_task_num", 'executed_task_num.csv' ), 
                    executed_task_num, delimiter=',', header='executed_task_num', comments='')
        
    def calculate_executed_task_num(self):
        executed_task = []
        for uav in self.uav_list:
            if uav.executed_task is not None: 
                executed_task.append(uav.executed_task)
        executed_task_num = len(set(executed_task))
        return executed_task_num


