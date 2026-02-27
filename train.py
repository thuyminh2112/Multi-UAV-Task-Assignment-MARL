import os.path
import csv
from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import draw_animation
from torch.utils.tensorboard import SummaryWriter
import random
import collections
from models.actor_critic import MemoryBuffer



class ReturnValueOfTrain:
    def __init__(self):
        self.return_list = []
        self.completion_workload_return_list = []
        self.task_assignment_return_list = []

    def item(self):
        value_dict = {
            'return_list': self.return_list, 
            'completion_workload_return_list': self.completion_workload_return_list, 
            'task_assignment_return_list': self.task_assignment_return_list
        }
        return value_dict

    def save_epoch(self, reward, cw_return, ta_return):
        self.return_list.append(reward)
        self.completion_workload_return_list.append(cw_return)
        self.task_assignment_return_list.append(ta_return)

def operate_epoch(config, env, agent, num_steps, cwriter_state=None, cwriter_prob=None):
    device = config["devices"][0]
    memory = MemoryBuffer(device=device)
    episode_return = 0
    episode_completion_workload_return = 0
    episode_task_assignment_return = 0
    executed_tasks = 0

    for i in range(num_steps): #num_steps: số bước mà một epoach sẽ chạy trong môi trường env
        config['step'] = i + 1
        action_list = []

        # each uav makes choices first
            # lay trang thei hien tai
            # xac dinh hanh dong thong qua agent
            # ghi lai trang thai va xac suat neu can
            # luu thong tin de dung cho huan luyen
        for uav in env.uav_list:
            state = uav.get_local_state()
            if cwriter_state:
                cwriter_state.writerow(state.tolist())

            if uav.status == 0:
                actions, log_probs = agent.take_action(state)  # action: int
                uav.last_action = actions
                uav.last_log_prob = log_probs
            else:
                actions = uav.last_action
                log_probs = uav.last_log_prob

            if cwriter_prob:
                cwriter_prob.writerow(log_probs.tolist())
                
            state = torch.FloatTensor(state).to(device)
            memory.states.append(state)
            memory.actions.append(actions)
            memory.logprobs.append(log_probs)
            action_list.append(actions)

        # use action_list to update the environment
        reward, reward_list = env.step(config=config, actions=action_list)  # action: List[int]
        # transition_dict['actions'].extend(action_list)
        # transition_dict['next_states'].extend(next_state_list)
        # transition_dict['rewards'].extend(reward_list['rewards'])
        memory.rewards.extend(reward_list['rewards'])
        done_list = [True if uav.status == 3 else False for uav in env.uav_list]
        memory.dones.extend(done_list)

        episode_return += sum(reward_list['rewards'])
        # completion_workload_reward = [x * sum(config['task']['w_task']) for x in reward_list['completion_workload_reward']]
        episode_completion_workload_return += sum(reward_list['completion_workload_reward'])
        episode_task_assignment_return += sum(reward_list['task_assignment_reward'])
        
    executed_tasks += env.calculate_executed_task_num()
        
    episode_return /= env.n_uav
    episode_completion_workload_return /= env.n_uav
    episode_task_assignment_return /= env.n_uav

    return (memory, episode_return, episode_completion_workload_return,
            episode_task_assignment_return, executed_tasks)

def train(config, env, agent, num_episodes, num_steps, frequency):
    # initialize saving list
    save_dir = os.path.join(config["save_dir"], "logs")
    writer = SummaryWriter(log_dir=save_dir)
    return_value = ReturnValueOfTrain()
    executed_tasks_num = []
    with open(os.path.join(save_dir, 'state.csv'), mode='w', newline='') as state_file, \
        open(os.path.join(save_dir, 'prob.csv'), mode='w', newline='') as prob_file:
        cwriter_state = csv.writer(state_file)
        cwriter_prob = csv.writer(prob_file)
        cwriter_state.writerow(['state'])  
        cwriter_prob.writerow(['prob'])  
        
        with tqdm(total=num_episodes, desc='Episodes') as pbar: #hien thi thanh tien trinh (progress bar)
            for i in range(num_episodes):
                # reset environment from config yaml file
                env.reset(config=config)

                memory, episode_return, episode_completion_workload_return, episode_task_assignment_return, executed_tasks = operate_epoch(config, env, agent, num_steps)
                writer.add_scalar('reward', episode_return, i)
                writer.add_scalar('completion_workload_return', episode_completion_workload_return, i)
                writer.add_scalar('task_assignment_return', episode_task_assignment_return, i)
                writer.add_scalar('executed_tasks', executed_tasks, i)
                executed_tasks_num.append(executed_tasks)
                # saving return lists
                return_value.save_epoch(episode_return, episode_completion_workload_return, episode_task_assignment_return)

                # update actor-critic network
                loss = agent.update(memory)
                writer.add_scalar('loss', loss, i)
                memory.clear_buffer()
                # writer.add_scalar('critic_loss', critic_loss, i)


                 # save & print
                if (i + 1) % frequency == 0:
                    # print some information
                    pbar.set_postfix({'episode': '%d' % (i + 1),
                                          'return': '%.3f' % np.mean(return_value.return_list[-frequency:]),
                                          'actor loss': '%f' % loss})
                                
                    # save results and weights
                    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=i)
                    agent.save(save_dir=config["save_dir"], epoch_i=i + 1)
                
                    env.save_position(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_executed_task_num(save_dir=config["save_dir"], executed_tasks=executed_tasks_num)
                # episode end
                pbar.update(1)
    writer.close()
    return return_value.item()

def evaluate(config, env, agent, num_steps):

    # initialize saving list
    return_value = ReturnValueOfTrain()

    # reset environment from config yaml file
    env.reset(config=config)

    # episode start
    memory, episode_return, episode_completion_workload_return, episode_task_assignment_return, executed_tasks = operate_epoch(config, env, agent, num_steps)

    # saving return lists
    return_value.save_epoch(episode_return, episode_completion_workload_return, episode_task_assignment_return)

    # save results and weights
    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    # env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()



