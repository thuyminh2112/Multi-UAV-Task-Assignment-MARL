import csv
import os.path
import numpy as np


def save_csv(config, return_list):
    """
    :param config:
    :param return_list:
        return_list = {
        'return_list': self.return_list,
        'target_tracking_return_list' :target_tracking_return_list,
        'boundary_punishment_return_list':boundary_punishment_return_list,
        'duplicate_tracking_punishment_return_list':duplicate_tracking_punishment_return_list
    }
    :return:
    """
    with open(os.path.join(config["save_dir"], 'return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Reward'])  # 写入表头
        for reward in return_list['return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'completion_workload_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['completion_workload'])
        for reward in return_list['completion_workload_return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'task_assignment_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task_assignment'])
        for reward in return_list['task_assignment_return_list']:
            writer.writerow([reward])

            

def clip_and_normalize(val, floor, ceil, choice=1):
    if val < floor or val > ceil:
        val = max(val, floor)
        val = min(val, ceil)
        print("overstep in clip.")
    val = np.clip(val, floor, ceil)
    mid = (floor + ceil) / 2
    if choice == -1:
        val = (val - floor) / (ceil - floor) - 1  # (-1, 0)
    elif choice == 0:
        val = (val - floor) / (ceil - floor)  # (0, 1)
    else:
        val = (val - mid) / (mid - floor)  # (-1, 1)
    return val
