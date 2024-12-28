import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results  # used to load log data

logging.getLogger('rmsaenv').setLevel(logging.INFO)

# Constants and parameters
seed = 20
episodes = 1
episode_length = 70
incremental_traffic_percentage = 80

logging_dir = "./tmp/logrmsa-ppo-defragmentation/"
figures_folder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

min_load = 40
max_load = 41
step_length = 8
steps = int((max_load - min_load) / step_length) + 1

movable_connections = [0, 10, 50]
m = len(movable_connections)

# Metrics arrays
SP_SBR_load = np.zeros([m, steps])  # Service blocking rate
SP_BBR_load = np.zeros([m, steps])  # Bit-rate blocking rate
SP_EF_load = np.zeros([m, steps])   # External fragmentation
SP_CF_load = np.zeros([m, steps])   # Compactness fragmentation
SP_CFN_load = np.zeros([m, steps])  # Compactness network fragmentation

loads = np.zeros(steps)

for load_counter, load_traffic in enumerate(range(min_load, max_load, step_length)):
    for move in range(m):
        log_dir = f'{logging_dir}logs_{load_traffic}_{episode_length}_{incremental_traffic_percentage}/'
        os.makedirs(log_dir, exist_ok=True)
        loads[load_counter] = load_traffic

        # Load the monitor results
        all_results = load_results(log_dir)
        # Here, we explicitly define the monitor files list, overriding other approaches
        monitor_files = ['SP.monitor.csv']

        SBR, BBR, EF, CF, CFN = {}, {}, {}, {}, {}
        counter = 0
        for filename in monitor_files:
            SBR[filename] = all_results.loc[counter:counter + 9, 'episode_service_blocking_rate'].to_list()
            BBR[filename] = all_results.loc[counter:counter + 9, 'episode_bit_rate_blocking_rate'].to_list()
            EF[filename] = all_results.loc[counter:counter + 9, 'external_fragmentation_network_episode'].to_list()
            CF[filename] = all_results.loc[counter:counter + 9, 'compactness_fragmentation_network_episode'].to_list()
            CFN[filename] = all_results.loc[
                counter:counter + 9, 'compactness_network_fragmentation_network_episode'
            ].to_list()
            counter += 10

        # Retrieve lists from dictionaries (only one key in each dict)
        SP_SBR = next(iter(SBR.values()))
        SP_BBR = next(iter(BBR.values()))
        SP_EF = next(iter(EF.values()))
        SP_CF = next(iter(CF.values()))
        SP_CFN = next(iter(CFN.values()))

        # Compute mean values
        SP_SBR_load[move][load_counter] = np.mean(SP_SBR)
        SP_BBR_load[move][load_counter] = np.mean(SP_BBR)
        SP_EF_load[move][load_counter] = np.mean(SP_EF)
        SP_CF_load[move][load_counter] = np.mean(SP_CF)
        SP_CFN_load[move][load_counter] = np.mean(SP_CFN)

# --- Plotting Section ---

# 1. External Fragmentation
fig = plt.figure(figsize=[8.4, 4.8])
plt.plot(loads, SP_EF_load[0][:], '+-b', label='SP_EF')
plt.plot(loads, SP_EF_load[1][:], '+-r', label='SP_EF_10')
plt.plot(loads, SP_EF_load[2][:], '+-g', label='SP_EF_50')
plt.xlabel('Load')
plt.ylabel('External Fragmentation')
plt.legend()
plt.savefig(f'{figures_folder}/External_fragmentation.pdf')
plt.savefig(f'{figures_folder}/External_fragmentation.svg')
plt.show()
plt.close()

# 2. Service Blocking Rate
fig = plt.figure(figsize=[8.4, 4.8])
plt.semilogy(loads, SP_SBR_load[0][:], '+-b', label='SP_SBR')
plt.semilogy(loads, SP_SBR_load[1][:], '+-r', label='SP_SBR_10')
plt.semilogy(loads, SP_SBR_load[2][:], '+-g', label='SP_SBR_50')
plt.xlabel('Load')
plt.ylabel('Service Blocking Rate')
plt.legend()
plt.savefig(f'{figures_folder}/service_blocking.svg')
plt.savefig(f'{figures_folder}/service_blocking.pdf')
plt.show()
plt.close()

# 3. Bit-Rate Blocking Rate
fig = plt.figure(figsize=[8.4, 4.8])
plt.semilogy(loads, SP_BBR_load[0][:], '+-b', label='SP_BBR')
plt.semilogy(loads, SP_BBR_load[1][:], '+-r', label='SP_BBR_10')
plt.semilogy(loads, SP_BBR_load[2][:], '+-g', label='SP_BBR_50')
plt.xlabel('Load')
plt.ylabel('Bit-Rate Blocking Rate')
plt.legend()
plt.savefig(f'{figures_folder}/bit_blocking.svg')
plt.savefig(f'{figures_folder}/bit_blocking.pdf')
plt.show()
plt.close()

# 4. Compactness Fragmentation
fig = plt.figure(figsize=[8.4, 4.8])
plt.plot(loads, SP_CF_load[0][:], '+-b', label='SP_CF')
plt.plot(loads, SP_CF_load[1][:], '+-r', label='SP_CF_10')
plt.plot(loads, SP_CF_load[2][:], '+-g', label='SP_CF_50')
plt.xlabel('Load')
plt.ylabel('Compactness Fragmentation')
plt.legend()
plt.savefig(f'{figures_folder}/Compactness_fragmentation.pdf')
plt.savefig(f'{figures_folder}/Compactness_fragmentation.svg')
plt.show()
plt.close()

# 5. Compactness Network Fragmentation
fig = plt.figure(figsize=[8.4, 4.8])
plt.plot(loads, SP_CFN_load[0][:], '+-b', label='SP_CFN')
plt.plot(loads, SP_CFN_load[1][:], '+-r', label='SP_CFN_10')
plt.plot(loads, SP_CFN_load[2][:], '+-g', label='SP_CFN_50')
plt.xlabel('Load')
plt.ylabel('Compactness Network Fragmentation')
plt.legend()
plt.savefig(f'{figures_folder}/Compactness_network_fragmentation.pdf')
plt.savefig(f'{figures_folder}/Compactness_network_fragmentation.svg')
plt.show()
plt.close()
