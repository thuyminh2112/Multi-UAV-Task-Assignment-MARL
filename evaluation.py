import argparse
import os.path
from environment import Environment
from models.actor_critic import PPO
from utils.args_util import get_config
from train import train, evaluate
from utils.data_util import save_csv
from utils.draw_util import plot_reward_curve
import numpy as np


def print_config(vdict, name="config"):

    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")


def print_args(args, name="args"):
    """
    :param args:
    :param name: str, 
    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    for arg in vars(args):
        print("| {:<11} : {}".format(arg, getattr(args, arg)))
    print("-----------------------------------------")


def add_args_to_config(config, args):
    for arg in vars(args):
        # print("| {:<11} : {}".format(arg, getattr(args, arg)))
        config[str(arg)] = getattr(args, arg)


def main(args):
    config = get_config(os.path.join("configs", args.method + ".yaml"))
    add_args_to_config(config, args)
    print_config(config)
    print_args(args)


    env = Environment(n_uav=config["env"]["n_uav"], 
                      m_task=config["env"]["m_tasks"], 
                      x_max=config["env"]["x_max"],
                      y_max=config["env"]["y_max"],
                      na=config["env"]["na"],
                      v_min=config["uav"]["v_min"],
                      v_max=config["uav"]["v_max"])

    agent = PPO(state_size=10,
                hidden_size=config["actor_critic"]["hidden_dim"],
                action_size=config["env"]["m_tasks"],
                lr=float(config["actor_critic"]["actor_lr"]),
                gamma=float(config["actor_critic"]["gamma"]),
                device=config["devices"][0]) 
        # agent.load(args.actor_path, args.critic_path)


    if args.phase == "train":
        return_list = train(config=config,
                            env=env,
                            agent=agent,
                            num_episodes=args.num_episodes,
                            num_steps=args.num_steps,
                            frequency=args.frequency)
    elif args.phase == "evaluate":
        return_list = evaluate(config=config,
                               env=env,
                               agent=agent,
                               num_steps=args.num_steps)
    else:
        return

    save_csv(config, return_list)

    plot_reward_curve(config, return_list['return_list'], "Average reward", "Episodes")
    plot_reward_curve(config, return_list['completion_workload_return_list'], "Average completion workload", "Episodes")
    plot_reward_curve(config, return_list['task_assignment_return_list'], "Average task assignment reward", "Episodes")

    file_path = os.path.join(config["save_dir"], "executed_task_num", "executed_task_num.csv")
    # executed_task_num = np.loadtxt(file_path, delimiter=',', skiprows=1)
    # plot_reward_curve(config, executed_task_num, "Total number of executed task", "Episodes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--phase", type=str, default="evaluate", choices=["train", "evaluate", "run"])
    parser.add_argument("-e", "--num_episodes", type=int, default=1000, help="Number of training rounds")
    parser.add_argument("-s", "--num_steps", type=int, default=200, help="Number of steps per round")
    parser.add_argument("-f", "--frequency", type=int, default=100, help="How often to print and save information")
    parser.add_argument("-a", "--actor_path", type=str, default=None, help="The path of the actor network weights")
    parser.add_argument("-c", "--critic_path", type=str, default=None, help="Path of critic network weights")
    parser.add_argument("-m", "--method", help="", default="MAPPO", choices=["MAPPO"])
    main_args = parser.parse_args()

    main(main_args)
