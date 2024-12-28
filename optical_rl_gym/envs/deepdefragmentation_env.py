import gym
import numpy as np
from optical_rl_gym.utils import Path, Service, DefragmentationOption
from .rmsa_env import RMSAEnv
from .defragmentation_env import DefragmentationEnv


class DeepDefragmentationEnv(DefragmentationEnv):
    """
    Deep Defragmentation Environment for Elastic Optical Networks.
    """

    def __init__(
        self,
        topology=None,
        episode_length=1000,
        load=100,
        mean_service_holding_time=10800.0,
        num_spectrum_resources=100,
        node_request_probabilities=None,
        seed=None,
        k_paths=5,
        allow_rejection=False,
        incremental_traffic_percentage=80,
        rmsa_function=None,
        number_options=7,
        penalty_cycle=-0.25,
        penalty_movement=-3,
        fragmented_constraint=False,
    ):
        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            k_paths=k_paths,
            incremental_traffic_percentage=incremental_traffic_percentage,
            rmsa_function=rmsa_function,
            number_options=number_options,
            penalty_cycle=penalty_cycle,
            penalty_movement=penalty_movement,
            fragmented_constraint=fragmented_constraint,
        )

        shape = (
            (1 + 2 * self.topology.number_of_nodes() + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)
            * self.number_options
            + 1
        )
        self.action_space = gym.spaces.Discrete(self.number_options)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, dtype=np.uint8, shape=(shape,)
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_counters=False)

    def step(self, action: int):
        """
        Take an action in the environment.
        """
        if action >= self.number_options:
            raise ValueError("The action should be within the defined options.")

        if action >= len(self.defragmentation_options_available):
            return self.observation(), -1, False, super().get_info()

        return super().step(self.defragmentation_options_available[action])

    def observation(self):
        """
        Generate an observation for the current state.
        """
        source_destination_tau = np.zeros(
            (self.number_options, 2, self.topology.number_of_nodes())
        )
        options_obs = np.full(
            (self.number_options, 10), fill_value=-1.0
        )  # 10 features per option

        selected_defragmentation_options = self.defragmentation_options.copy()
        if len(selected_defragmentation_options) > 5:
            self._select_representative_options(selected_defragmentation_options)

        self.defragmentation_options_available = [
            DefragmentationOption(0, -1, 0, 0, 0, 0)
        ]
        index_option = 0

        while (
            len(self.defragmentation_options_available) - 1 < self.number_options
            and len(selected_defragmentation_options) > 0
        ):
            service_to_defrag = selected_defragmentation_options[index_option].service

            if service_to_defrag not in self.defragmented_services:
                option_idx = len(self.defragmentation_options_available) - 1
                self.defragmentation_options_available.append(
                    selected_defragmentation_options[index_option]
                )
                self._populate_observation_features(
                    service_to_defrag,
                    selected_defragmentation_options[index_option],
                    source_destination_tau,
                    options_obs,
                    option_idx,
                )

            index_option += 1
            if index_option >= len(selected_defragmentation_options):
                break

        reshaped_option = options_obs.reshape((1, -1))
        reshaped_option = np.append(
            reshaped_option, 1 if self.fragmentation_flag else 0
        )

        return np.concatenate(
            (
                source_destination_tau.reshape((1, -1)),
                reshaped_option.reshape((1, -1)),
            ),
            axis=1,
        ).reshape(self.observation_space.shape)

    def _select_representative_options(self, options):
        """
        Select representative defragmentation options based on heuristics.
        """
        for count in range(len(self.defragmentation_options)):
            option = self.defragmentation_options[count]
            if option.service.number_slots > options[1].service.number_slots:
                options[1] = option
            if option.service.number_slots < options[2].service.number_slots:
                options[2] = option
            if option.service.route.hops < options[3].service.route.hops:
                options[3] = option
            if (
                option.service.arrival_time + option.service.holding_time
                > options[4].service.arrival_time + options[4].service.holding_time
            ):
                options[4] = option

    def _populate_observation_features(
        self, service, option, source_destination_tau, options_obs, option_idx
    ):
        """
        Populate observation features for each defragmentation option.
        """
        min_node = min(service.source_id, service.destination_id)
        max_node = max(service.source_id, service.destination_id)
        source_destination_tau[option_idx, 0, min_node] = 1
        source_destination_tau[option_idx, 1, max_node] = 1

        options_obs[option_idx, 0] = len(service.route.node_list)
        options_obs[option_idx, 1] = service.arrival_time / self.current_time
        options_obs[option_idx, 2] = (service.number_slots - 5.5) / 3.5
        options_obs[option_idx, 3] = (
            service.arrival_time + service.holding_time - self.current_time
        ) / self.mean_service_holding_time
        options_obs[option_idx, 4] = (
            2 * (service.initial_slot - 0.5 * self.num_spectrum_resources)
            / self.num_spectrum_resources
        )
        options_obs[option_idx, 5] = (
            2 * (np.sum(self.get_available_slots(service.route)) - 0.5 * self.num_spectrum_resources)
            / self.num_spectrum_resources
        )
        options_obs[option_idx, 6] = (option.starting_slot - 5.5) / 3.5
        options_obs[option_idx, 7] = option.size_of_free_block / 10
        options_obs[option_idx, 8] = option.left_side_free_slots / 10
        options_obs[option_idx, 9] = option.right_side_free_slots / 10

    def reset(self, only_counters=True):
        """
        Reset the environment.
        """
        return super().reset(only_counters=only_counters)
