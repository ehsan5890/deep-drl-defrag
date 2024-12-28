import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.monitor import get_monitor_files
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA
from optical_rl_gym.envs.defragmentation_env import choose_randomly, OldestFirst, assigning_path_without_defragmentation
from optical_rl_gym.utils import evaluate_heuristic, random_policy
import pandas as pd
from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 1000
episode_length = 200
incremental_traffic_percentage = 80
monitor_files = []
policies = []

# adding logging method
#log_dir = "./tmp/logrmsa-ppo/"
# logging_dir = "./tmp/logrmsa-ppo-defragmentation/"
logging_dir = "../examples/stable_baselines3/results/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)
#
# with open(f'../examples/topologies/Telia_5-paths.h5', 'rb') as f:
#      topology = pickle.load(f)

min_load = 20
max_load = 21
step_length = 8
steps = int((max_load - min_load)/step_length) +1
loads = np.zeros(steps)

for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
    log_dir = f'{logging_dir}/logs_{load_traffic}_{episode_length}_{incremental_traffic_percentage}/'
    log_dir = f'{logging_dir}heuristic/'
    os.makedirs(log_dir, exist_ok=True)
    env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=25,
                    episode_length=episode_length, num_spectrum_resources=64,
                    incremental_traffic_percentage=incremental_traffic_percentage,
                    rmsa_function=shortest_path_first_fit)
    ############# No defragmentation#########

    env_df = gym.make('Defragmentation-v0', **env_args)
    env_df = Monitor(env_df, log_dir + 'df', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
                                                                'reward','number_movements',
                                                                'number_defragmentation_procedure','number_arrivals','bit_rate_blocking_rate','number_movements_episode',
                                                                'number_defragmentation_procedure_episode',))
    mean_reward_df, std_reward_df = evaluate_heuristic(env_df, assigning_path_without_defragmentation,
                                                       n_eval_episodes=episodes)
    ############## Choose Randomly######

    env_df_random = gym.make('Defragmentation-v0', **env_args)
    env_df_random = Monitor(env_df_random, log_dir + 'rnd', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
                                                                'reward','number_movements',
                                                                'number_defragmentation_procedure','number_arrivals','bit_rate_blocking_rate','number_movements_episode',
                                                                'number_defragmentation_procedure_episode',))
    mean_reward_df_random, std_reward_df_random = evaluate_heuristic(env_df_random, choose_randomly,
                                                                     n_eval_episodes=episodes)
    #######Oldest first#####
    # for j,i  in enumerate([(5, 20)]):
    #     oldest_scenario = OldestFirst(i[0], i[1])
    #     env_df_oldest = gym.make('Defragmentation-v0', **env_args)
    #     env_df_oldest = Monitor(env_df_oldest, log_dir + f'df-oldest-{i[0]}-{i[1]}', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
    #                                                             'reward','number_movements',
    #                                                             'number_defragmentation_procedure','number_arrivals','bit_rate_blocking_rate','number_movements_episode',
    #                                                             'number_defragmentation_procedure_episode',))
    #     mean_reward_df_oldest, std_reward_df_oldest = evaluate_heuristic(env_df_oldest, oldest_scenario.choose_oldest_first, n_eval_episodes=episodes)
    #     print('Oldest-Defragmentation:'.ljust(8), f'{mean_reward_df_oldest:.4f}  {std_reward_df_oldest:<7.4f}')
    #     print('Bit rate blocking:', (env_df_oldest.episode_bit_rate_requested - env_df_oldest.episode_bit_rate_provisioned) / env_df_oldest.episode_bit_rate_requested)
    #     print('Request blocking:', (env_df_oldest.episode_services_processed - env_df_oldest.episode_services_accepted) / env_df_oldest.episode_services_processed)
