import gym
import copy
import math
import heapq
import logging
import functools
import random
import numpy as np
from optical_rl_gym.utils import Path, Service
from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste, DefragmentationOption
from optical_rl_gym.envs.rmsa_env import (
    shortest_path_first_fit,
    shortest_available_path_first_fit,
    least_loaded_path_first_fit,
    SimpleMatrixObservation,
    Fragmentation_alignment_aware_RMSA,
)
from .rmsa_env import RMSAEnv


class DefragmentationEnv(RMSAEnv):
    """
    Defragmentation Environment for Elastic Optical Networks.
    """
    metadata = {
        'metrics': [
            'service_blocking_rate',
            'episode_service_blocking_rate',
            'bit_rate_blocking_rate',
            'episode_bit_rate_blocking_rate'
        ]
    }

    def __init__(
        self,
        topology=None,
        episode_length=1000,
        load=10,
        mean_service_holding_time=10800.0,
        num_spectrum_resources=100,
        node_request_probabilities=None,
        bit_rate_lower_bound=25,
        bit_rate_higher_bound=100,
        seed=None,
        k_paths=5,
        allow_rejection=False,
        reset=True,
        incremental_traffic_percentage=80,
        defragmentation_period=4,
        rmsa_function=None,
        number_options=5,
        penalty_movement=-3,
        penalty_cycle=-0.25,
        fragmented_constraint=False
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
            reset=False,
        )
        assert 'modulations' in self.topology.graph

        self.number_options = number_options
        self.defragmentation_period = defragmentation_period
        self.rmsa_function = rmsa_function
        self.fragmented_constraint = fragmented_constraint

        self.service_to_defrag = None
        self.defragmentation_options = []
        self.defragmented_services = []
        self.defragmentation_flag = False

        self.episode_num_moves = 0
        self.num_moves = 0
        self.episode_defragmentation_procedure = 0
        self.defragmentation_procedure = 0

        self.fragmentation_penalty_cycle = penalty_cycle
        self.fragmentation_penalty_movement = penalty_movement

        self.defragmentation_period_count = 0
        self.defragmentation_movement_period = 0
        self.number_existing_options = 0

        self.reward_cumulative = 0
        self.logger = logging.getLogger('rmsaenv')

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG, generating a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.'
            )

        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action):
        """
        Perform a step in the environment based on the given action.
        """
        service_to_defrag, new_initial_slot = action.service, action.starting_slot
        reward = 0

        if new_initial_slot != -1:
            self.episode_num_moves += 1
            self.num_moves += 1
            self.service_to_defrag = service_to_defrag
            self.defragmented_services.append(service_to_defrag)
            self._move_path(new_initial_slot)
            reward += self.fragmentation_penalty_movement
            self.fragmentation_flag = True
        else:
            path, initial_slot = self.rmsa_function(self)
            if path < self.k_paths and initial_slot < self.num_spectrum_resources:
                slots = super().get_number_slots(
                    self.k_shortest_paths[self.service.source, self.service.destination][path]
                )
                if super().is_path_free(
                    self.k_shortest_paths[self.service.source, self.service.destination][path],
                    initial_slot,
                    slots
                ):
                    super()._provision_path(
                        self.k_shortest_paths[self.service.source, self.service.destination][path],
                        initial_slot,
                        slots
                    )
                    self.service.accepted = True
                    super()._add_release(self.service)
                else:
                    self.service.accepted = False
            else:
                self.service.accepted = False

            self.services_processed += 1
            self.topology.graph['services'].append(self.service)

            if self.fragmentation_flag:
                reward += self.fragmentation_penalty_cycle
                self.fragmentation_flag = False
                self.defragmented_services = []
                self.episode_defragmentation_procedure += 1
                self.defragmentation_procedure += 1

            super()._next_service()

        episode_service_blocking_rate = (
            self.episode_services_processed - self.episode_services_accepted
        ) / (self.episode_services_processed + 1)
        reward += 1 - episode_service_blocking_rate
        self.reward_cumulative += reward

        return (
            self.observation(),
            reward,
            self.episode_services_processed + self.episode_num_moves == self.episode_length,
            self.get_info(),
        )

    def reset(self, only_counters=True):
        """
        Reset environment state.
        """
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_num_moves = 0
        self.episode_defragmentation_procedure = 0
        self.reward_cumulative = 0

        if only_counters:
            return self.observation()

        return super().reset(only_counters=only_counters)

    def _move_path(self, new_initial_slot):
        """
        Move path during defragmentation.
        """
        super()._release_path(self.service_to_defrag)
        self.service_to_defrag.initial_slot = new_initial_slot

        for i in range(len(self.service_to_defrag.route.node_list) - 1):
            self.topology.graph['available_slots'][
                self.topology[self.service_to_defrag.route.node_list[i]][self.service_to_defrag.route.node_list[i + 1]]['index'],
                new_initial_slot:new_initial_slot + self.service_to_defrag.number_slots
            ] = 0

        self.topology.graph['running_services'].append(self.service_to_defrag)
        self._update_network_stats()

    def get_info(self):
        """
        Get current environment statistics.
        """
        return {
            'service_blocking_rate': (self.services_processed - self.services_accepted) / (self.services_processed + 1),
            'reward': self.reward_cumulative,
        }


def choose_randomly(env: DefragmentationEnv):
    """
    Random defragmentation option selection.
    """
    return random.choice(env.defragmentation_options)


def assigning_path_without_defragmentation(env: DefragmentationEnv):
    """
    Default option when no defragmentation is applied.
    """
    return DefragmentationOption(0, -1, 0, 0, 0, 0)


class OldestFirst:
    """
    Implements an 'Oldest First' defragmentation strategy.
    """
    def __init__(self, defragmentation_period: int = 10, number_connection: int = 10) -> None:
        self.defragmentation_period = defragmentation_period
        self.number_connection = number_connection

    def choose_oldest_first(self, env: DefragmentationEnv):
        """
        Selects the oldest defragmentation option based on periodic checks.
        """
        env.defragmentation_options_available = [DefragmentationOption(0, -1, 0, 0, 0, 0)]
        index_option = 0

        # Populate available defragmentation options
        while len(env.defragmentation_options_available) - 1 < env.number_options and len(env.defragmentation_options) > 0:
            service_to_defrag = env.defragmentation_options[index_option].service
            if (
                service_to_defrag not in env.defragmented_services
                and env.defragmentation_options[index_option].starting_slot < service_to_defrag.initial_slot
            ):
                env.defragmentation_options_available.append(env.defragmentation_options[index_option])
            index_option += 1
            if index_option >= len(env.defragmentation_options):
                break

        env.number_existing_options = len(env.defragmentation_options_available)

        # Check if it's time to trigger defragmentation
        if env.defragmentation_period_count != self.defragmentation_period:
            env.defragmentation_period_count += 1
            return DefragmentationOption(0, -1, 0, 0, 0, 0)
        else:
            env.defragmentation_movement_period += 1
            if env.defragmentation_movement_period == self.number_connection:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0

            if len(env.defragmentation_options_available) > 1:
                return env.defragmentation_options_available[1]
            else:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
                return DefragmentationOption(0, -1, 0, 0, 0, 0)
