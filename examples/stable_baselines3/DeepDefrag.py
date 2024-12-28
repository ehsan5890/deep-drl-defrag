import os
import pickle
import numpy as np

from IPython.display import clear_output

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy
# from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure, Logger
stable_baselines3.__version__ # printing out stable_baselines version used

import gym


# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True


from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit
# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py
topology_name = 'nsfnet_chen_eon'
# topology_name = 'Germany50'
k_paths = 5
with open(f'../topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
    topology = pickle.load(f)

# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])
load = 175
# mean_service_holding_time=7.5,
penalty_cycle= -0.3
penalty_movement=-0.05
number_options=5
env_args = dict(topology=topology, seed=10,load=load, num_spectrum_resources=320,
                allow_rejection=False, # the agent cannot proactively reject a request
                mean_service_holding_time=25, # value is not set as in the paper to achieve comparable reward values
                episode_length=200,
                rmsa_function=shortest_available_path_first_fit, node_request_probabilities = node_request_probabilities,
                number_options = number_options,
                penalty_cycle=penalty_cycle,
                penalty_movement=penalty_movement)


# Create log dir

seed = 18
log_dir = f"./results/DeepDefrag-{penalty_cycle}-{penalty_movement}-{number_options}-{topology_name}-{load}-{seed}-ppo/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

env = gym.make('DeepDefragmentation-v0', **env_args)

# logs will be saved in log_dir/training.monitor.csv
# in this case, on top of the usual monitored things, we also monitor service and bit rate blocking rates
env = Monitor(env, log_dir + f'training-dqn', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
                                                                'reward','number_movements',
                                                                'number_defragmentation_procedure','number_arrivals','bit_rate_blocking_rate','number_movements_episode',
                                                                'number_defragmentation_procedure_episode','service_blocked_eopisode', 'number_options', 'existing_options'
                                                                ))
# for more information about the monitor, check https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/bench/monitor.html#Monitor

# here goes the arguments of the policy network to be useds
# policy_args = dict(net_arch=5*[128], # the neural network has five layers with 128 neurons each
#                    act_fun=tf.nn.elu) # we use the elu activation function

## configure logger
tmp_path = "./tmp/sb3_log/"
os.makedirs(tmp_path, exist_ok=True)
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
#new_logger.info("episodes")

policy_args = dict(net_arch=5*[256]) # we use the elu activation functions
agent = PPO(MlpPolicy, env, verbose=0, tensorboard_log="./tb/PPO-DeepDefrag-v0/{load}", policy_kwargs=policy_args, gamma=.95, learning_rate=10e-5, ent_coef=0.1)
# agent = DQN(MlpPolicy, env, verbose=0, tensorboard_log=f"./tb/DQN-DeepDefrag-v0/{load}", policy_kwargs=policy_args, gamma=.95, learning_rate=5
# *10e-6, batch_size=200, seed = seed, exploration_fraction=0.3)

#agent.set_logger(new_logger)


a = agent.learn(total_timesteps=6000000, callback=callback)