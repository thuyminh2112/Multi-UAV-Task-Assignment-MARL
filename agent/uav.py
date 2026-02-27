import random
import numpy as np
from math import cos, sin, sqrt, exp, pi, e, atan2
from typing import List, Tuple
from agent import task
from agent.task import Task
from agent.trunk import Trunk
from scipy.special import softmax
from utils.data_util import clip_and_normalize


class UAV:
    def __init__(self, x0, y0, h_0, h_idx, v_0, task_id, x_max, v_max, v_min, h_max, Na, M, dt, c_uav, c_max, w_uav, E_spec, d_safe, E_fly):

            # the position, velocity, heading and task of this uav
        self.x = x0
        self.y = y0
        self.h = h_0 
        self.x_max = x_max

        # the max and min velocity, max heading change
        self.h_max = h_max
        self.v_max = v_max
        self.v_min = v_min
        self.Na = Na
        self.M = M

        # the time step
        self.dt = dt

        # capacity
        self.c_uav_max = c_uav
        self.c_uav = c_uav
        self.c_max = c_max

        # set of local information
        self.task_observation = [] 
        self.uav_observation = []

        # reward 
        self.reward = 0
        self.workload_completed = 0

        # status 
        self.status = 0

        # action
        self.h_idx = h_idx
        self.v = v_0
        self.task_id = task_id

        # weight, energy specification
        self.w_uav = w_uav
        self.E_spec = E_spec
        self.E_fly = E_fly

        # safe distance
        self.d_safe = d_safe

        self.executed_task = None

        self.last_action = None
        self.last_log_prob = None

    def __distance(self, task) -> float:
        return sqrt((self.x - task.x) ** 2 + (self.y - task.y) ** 2)

    @staticmethod
    def distance(x1, y1, x2, y2) -> float:
        """
        calculate the distance from uav to task
        :param x2:
        :param y1:
        :param x1:
        :param y2:
        :return: scalar
        """
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def discrete_heading(self, h_idx: int) -> float: 
        """Map heading index to real-world angular rate"""
        na = h_idx + 1
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)
    
    def assign_task(self, task_id: int, num_task: int) -> tuple[int, int]:
        if self.status == 0:
            self.task_id = task_id.item()
            self.assign_idx = [0] * num_task
            self.assign_idx[task_id] = 1

            return self.task_id

    def execute_task(self, task_list: List['Task']):
        if self.status == 1:
            task = task_list[self.task_id]
            dis = self.__distance(task)
            dx = self.dt * self.v_max * (task.x - self.x) / dis
            dy = self.dt * self.v_max * (task.y - self.y) / dis
            self.x += dx
            self.y += dy
            dis = self.__distance(task)
            if dis <= task.R_task:
                workload = min(self.c_uav_max, task.w_task)
                self.workload_completed = workload
                self.c_uav = self.c_uav_max - workload
                self.executed_task = self.task_id
        else:
            self.x = self.x
            self.y = self.y

        return self.x, self.y
    
    def return_to_trunk(self, trunk: Trunk):
        if self.status == 2:
            dis = self.__distance(trunk)
            dx = self.dt * self.v_max * (trunk.x - self.x) / dis
            dy = self.dt * self.v_max * (trunk.y - self.y) / dis
            self.x += dx
            self.y += dy
        if self.status == 3:
            self.x = self.x
            self.y = self.y
        return self.x, self.y

    def observe_task(self, task_list: List['Task']):
        self.task_observation = []
        for task in task_list:
            self.task_observation.append(((task.x - self.x) / self.x_max, 
                                          (task.y - self.y) / self.x_max, 
                                          task.w_task / task.w_task_max))

    def observe_uav(self, uav_list: List['UAV']):
        self.uav_observation = []
        for uav in uav_list:
            if uav != self:
                self.uav_observation.append(((uav.x - self.x) /self.x_max, 
                                             (uav.y - self.y) / self.x_max, 
                                             (uav.task_id / self.M),
                                             uav.c_uav_max / self.c_max))

    def __get_all_local_state(self) -> Tuple[List[Tuple[float, float, float, float]], 
                                        List[Tuple[float, float, float]], 
                                        Tuple[float, float, float]]:

        return self.uav_observation, self.task_observation, (self.x / self.x_max, self.y / self.x_max, self.task_id / self.M)
    
    def __get_local_state_by_mean(self) -> 'np.ndarray':
        other_uavs, tasks, sb = self.__get_all_local_state()

        if other_uavs:
            other_uavs = np.array(other_uavs)
            average_other_uav = np.mean(other_uavs, axis=0) 
        else:
            average_other_uav = -np.ones(4)

        if tasks:
            tasks = np.array(tasks)
            average_task = np.mean(tasks, axis=0)
        else:
            average_task = -np.ones(3)

        sb = np.array(sb)

        result = np.hstack((average_other_uav, average_task, sb)) #hstack (horizontaly stack): noi 3 vector numpy lai thanh 1 vector trang thai duy nhat

        return result

    def get_local_state(self) -> 'np.ndarray':
        return self.__get_local_state_by_mean()

    def __calculate_completion_workload_reward(self, task_list: List['Task']) -> int:
        completion_workload_reward = 0
        if self.status == 1:
            task = task_list[self.task_id]
            dist = self.__distance(task)
            if dist <= task.R_task:
                completion_workload_reward = self.workload_completed
                self.status = 2 # returning to trunk car
            else:
                completion_workload_reward = 0
        else: 
            completion_workload_reward = 0
        return completion_workload_reward 

    def __calculate_task_assignment_reward(self, uav_list: List['UAV'], task_list: List['Task']) -> float:
        reward_assign = 0
        if self.status == 0:
            c_tot = 0
            for uav in uav_list:
                if uav.assign_idx[self.task_id] == 1:
                    c_tot += uav.c_uav_max
            task = task_list[self.task_id]
            if c_tot >= task.w_task:
                reward_assign = (task.w_task - (c_tot - task.w_task)) /  task.w_task
                # reward_assign = 1 - (c_tot - task.w_task) /  c_tot
                # reward_assign = 4*(task.w_task - (c_tot - task.w_task)) /  task.w_task
            else:
                # reward_assign = 1 - ((task.w_task - c_tot) / task.w_task)
                reward_assign = (c_tot - task.w_task) / task.w_task
            self.status = 1 # performing task    
        return reward_assign
    
    def __calculate_distance_to_trunk_reward(self, trunk: Trunk) -> float:
        reward_return = 0
        if self.status == 2:
            distance = self.__distance(trunk)

            reward_return = trunk.R_trunk / (distance)
            if distance <= trunk.R_trunk:
                reward_return += (trunk.R_trunk - distance) / trunk.R_trunk
                self.status = 3 # Done
            else:
                reward_return = (trunk.R_trunk - distance) / distance
        return reward_return

    def calculate_raw_reward(self, uav_list: List['UAV'], task_list: List['Task'], trunk: Trunk):
        distance_to_trunk_reward = self.__calculate_distance_to_trunk_reward(trunk)
        task_assignment_reward = self.__calculate_task_assignment_reward(uav_list, task_list)
        completion_workload_reward = self.__calculate_completion_workload_reward(task_list)
        
        return completion_workload_reward, task_assignment_reward



   