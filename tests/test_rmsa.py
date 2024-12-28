import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.monitor import get_monitor_files
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA
from optical_rl_gym.utils import evaluate_heuristic, random_policy
import pandas as pd
from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste
import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 2
episode_length = 40
incremental_traffic_percentage = 75
monitor_files = []
policies = []

# adding logging method
#log_dir = "./tmp/logrmsa-ppo/"
logging_dir = "./tmp/logrmsa-ppo/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)
topology_name = 'gbn'
topology_name = 'nobel-us'
topology_name = 'germany50'
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)
#
with open(f'../examples/topologies/Telia_5-paths.h5', 'rb') as f:
     topology = pickle.load(f)

min_load = 88
max_load = 110
step_length = 8
steps = int((max_load - min_load)/step_length) +1
loads = np.zeros(steps)

for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
    log_dir = f'{logging_dir}/logs_{load_traffic}/'
    log_dir = f'{logging_dir}/logs_{load_traffic}_{episode_length}_{incremental_traffic_percentage}/'
    os.makedirs(log_dir, exist_ok=True)
    env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=25,
                    episode_length=episode_length, num_spectrum_resources=120, incremental_traffic_percentage = incremental_traffic_percentage)
    #
    # print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))
    loads[load_counter] = load_traffic


    init_env = gym.make('RMSA-v0', **env_args)
    init_env = Monitor(init_env, log_dir + 'Random', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate',
                                                                    'external_fragmentation_network_episode', 'compactness_fragmentation_network_episode',
                                                                    'compactness_network_fragmentation_network_episode','delay_deviation_absolute',
                                                                    'delay_deviation_percentage','bit_rate_blocking_fragmentation',
                                                                    'service_blocking_rate_fragmentation','external_fragmentation_deviation',
                                                                    'compactness_fragmentation_deviation','service_blocking_rate_100',
                                                                    'service_blocking_rate_200','service_blocking_rate_400'))
    # env_rnd = SimpleMatrixObservation(init_env)
    # mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, random_policy, n_eval_episodes=episodes)
    # print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
    # print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
    # print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
    # plot_spectrum_assignment(init_env.topology, init_env.spectrum_slots_allocation, values=True,
    #                                    filename=f'{figures_floder}/Spectrum_assignment- {load_traffic}-Random.svg',
    #                                    title=f'Spectrum assignment - {load_traffic} - Random - Useful')
    # # # # #
    env_sp = gym.make('RMSA-v0', **env_args)
    env_sp = Monitor(env_sp, log_dir + 'SP', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate',
                                                            'external_fragmentation_network_episode', 'compactness_fragmentation_network_episode',
                                                            'compactness_network_fragmentation_network_episode','delay_deviation_absolute',
                                                            'delay_deviation_percentage','bit_rate_blocking_fragmentation',
                                                            'service_blocking_rate_fragmentation','external_fragmentation_deviation',
                                                            'compactness_fragmentation_deviation','service_blocking_rate_100',
                                                            'service_blocking_rate_200','service_blocking_rate_400'))
    mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
    print('SP-FF:'.ljust(8), f'{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}')
    print('Bit rate blocking:', (env_sp.episode_bit_rate_requested - env_sp.episode_bit_rate_provisioned) / env_sp.episode_bit_rate_requested)
    print('Request blocking:', (env_sp.episode_services_processed - env_sp.episode_services_accepted) / env_sp.episode_services_processed)
    plot_spectrum_assignment(env_sp.topology, env_sp.spectrum_slots_allocation, values=True,
                                     filename=f'{figures_floder}/Spectrum_assignment- {load_traffic}-ShortestPath.svg',
                                      title=f'Spectrum assignment - {load_traffic} - ShortestPath - Useful')
    # # #
    env_sap = gym.make('RMSA-v0', **env_args)
    env_sap = Monitor(env_sap, log_dir + 'SP-AFF', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate',
                                                                  'external_fragmentation_network_episode', 'compactness_fragmentation_network_episode',
                                                                  'compactness_network_fragmentation_network_episode','delay_deviation_absolute',
                                                                  'delay_deviation_percentage','bit_rate_blocking_fragmentation',
                                                                  'service_blocking_rate_fragmentation','external_fragmentation_deviation',
                                                                  'compactness_fragmentation_deviation','service_blocking_rate_100',
                                                                  'service_blocking_rate_200','service_blocking_rate_400'))
    # mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
    # print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
    # print('Bit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
    # print('Request blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)
    # plot_spectrum_assignment(env_sap.topology, env_sap.spectrum_slots_allocation, values=True,
    #                                    filename=f'{figures_floder}/Spectrum_assignment- {load_traffic}-ShortestavailablePath.svg',
    #                                    title=f'Spectrum assignment - {load_traffic} - ShortestavailablePath - Useful')
    # # # #
    # env_llp = gym.make('RMSA-v0', **env_args)
    # env_llp = Monitor(env_llp, log_dir + 'LLP', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate',
    #                                                            'external_fragmentation_network_episode', 'compactness_fragmentation_network_episode',
    #                                                            'compactness_network_fragmentation_network_episode','delay_deviation_absolute',
    #                                                            'delay_deviation_percentage','bit_rate_blocking_fragmentation',
    #                                                            'service_blocking_rate_fragmentation','external_fragmentation_deviation',
    #                                                            'compactness_fragmentation_deviation','service_blocking_rate_100',
    #                                                            'service_blocking_rate_200','service_blocking_rate_400'))
    # mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
    # print('LLP-FF:'.ljust(8), f'{mean_reward_llp:.4f}  {std_reward_llp:.4f}')
    # print('Bit rate blocking:', (env_llp.episode_bit_rate_requested - env_llp.episode_bit_rate_provisioned) / env_llp.episode_bit_rate_requested)
    # print('Request blocking:', (env_llp.episode_services_processed - env_llp.episode_services_accepted) / env_llp.episode_services_processed)
    # plot_spectrum_assignment(env_llp.topology, env_llp.spectrum_slots_allocation, values=True,
    #                                    filename=f'{figures_floder}/Spectrum_assignment- {load_traffic}-LLP.svg',
    #                                    title=f'Spectrum assignment - {load_traffic} - LLP - Useful')


    ## Fragmentation and aligment aware RMSA


    env_FAAR = gym.make('RMSA-v0', **env_args)
    env_FAAR = Monitor(env_FAAR, log_dir + 'FAAR', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate',
                                                            'external_fragmentation_network_episode', 'compactness_fragmentation_network_episode',
                                                            'compactness_network_fragmentation_network_episode','delay_deviation_absolute',
                                                            'delay_deviation_percentage','bit_rate_blocking_fragmentation',
                                                                  'service_blocking_rate_fragmentation','external_fragmentation_deviation',
                                                                  'compactness_fragmentation_deviation','service_blocking_rate_100',
                                                                  'service_blocking_rate_200','service_blocking_rate_400'))
    mean_reward_FAAR, std_reward_FAAR = evaluate_heuristic(env_FAAR, Fragmentation_alignment_aware_RMSA, n_eval_episodes=episodes)
    print('FAAR-FF:'.ljust(8), f'{mean_reward_FAAR:.4f}  {std_reward_FAAR:<7.4f}')
    print('Bit rate blocking:', (env_FAAR.episode_bit_rate_requested - env_FAAR.episode_bit_rate_provisioned) / env_FAAR.episode_bit_rate_requested)
    print('Request blocking:', (env_FAAR.episode_services_processed - env_FAAR.episode_services_accepted) / env_FAAR.episode_services_processed)
    plot_spectrum_assignment(env_FAAR.topology, env_FAAR.spectrum_slots_allocation, values=True,
                                     filename=f'{figures_floder}/Spectrum_assignment- {load_traffic}-FAAR.svg',
                                      title=f'Spectrum assignment - {load_traffic} - FAAR - Useful')

