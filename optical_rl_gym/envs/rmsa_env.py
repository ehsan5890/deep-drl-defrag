import copy
import functools
import gym
import heapq
import logging
import math
import random
from typing import Optional

import numpy as np

from optical_rl_gym.utils import Path, Service
from .optical_network_env import OpticalNetworkEnv


class RMSAEnv(OpticalNetworkEnv):
    """
    RMSA Environment
    ----------------
    This class implements the Routing and Spectrum Assignment (RSA) environment for optical networks.
    It inherits from OpticalNetworkEnv and manages the state, actions, and observation spaces, as well
    as the reward function for the environment.
    """

    metadata = {
        'metrics': [
            'service_blocking_rate',
            'episode_service_blocking_rate',
            'bit_rate_blocking_rate',
            'episode_bit_rate_blocking_rate',
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
        defragmentation_period=32,
        movable_connections=10,
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
            incremental_traffic_percentage=incremental_traffic_percentage
        )
        assert 'modulations' in self.topology.graph

        # Specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.bit_rate_blocked_fragmentation = 0
        self.bit_rate_blocked_lack_resource = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.average_delay_absolute = 0
        self.average_delay_percentage = 0
        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound
        self.block_due_to_fragmentation = False
        self.defragmentation_period = defragmentation_period
        self.movable_connections = movable_connections
        self.episode_defragmentation_procedure = 0
        self.episode_num_moves = 0

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=np.int_,
        )

        # Do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # Defining the observation and action spaces
        self.actions_output = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_output = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.actions_taken = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )

        self.action_space = gym.spaces.MultiDiscrete(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action)
        )

        self.observation_space = gym.spaces.Dict(
            {
                'topology': gym.spaces.Discrete(10),
                'current_service': gym.spaces.Discrete(10),
            }
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG, which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.'
            )

        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action):
        """
        Executes a step in the environment given an action = [path_index, slot_index].
        """
        path, initial_slot = action[0], action[1]
        self.actions_output[path, initial_slot] += 1

        if (path < self.k_paths) and (initial_slot < self.num_spectrum_resources):
            # Action is for assigning a path
            slots = self.get_number_slots(
                self.k_shortest_paths[self.service.source, self.service.destination][path]
            )
            self.logger.debug(
                f'{self.service.service_id} processing action {action} path {path} '
                f'and initial slot {initial_slot} for {slots} slots'
            )
            if self.is_path_free(
                self.k_shortest_paths[self.service.source, self.service.destination][path],
                initial_slot,
                slots
            ):
                self._provision_path(
                    self.k_shortest_paths[self.service.source, self.service.destination][path],
                    initial_slot,
                    slots
                )
                self.service.accepted = True
                self.actions_taken[path, initial_slot] += 1
                self._add_release(self.service)

                # Updating delay
                delay_path = (
                    self.k_shortest_paths[self.service.source, self.service.destination][path].length
                    / (2 * (10**2))
                )
                shortest_path_len = self.k_shortest_paths[
                    self.service.source, self.service.destination
                ][0].length
                delay_deviation_path_absolute = (
                    self.k_shortest_paths[self.service.source, self.service.destination][path].length
                    - shortest_path_len
                ) / (2 * (10**2))
                delay_deviation_path_percentage = (
                    delay_deviation_path_absolute / delay_path
                ) * 100 if delay_path > 0 else 0

                self.average_delay_absolute = (
                    (self.services_accepted * self.average_delay_absolute)
                    + delay_deviation_path_absolute
                ) / (self.services_accepted + 1)

                self.average_delay_percentage = (
                    (self.services_accepted * self.average_delay_percentage)
                    + delay_deviation_path_percentage
                ) / (self.services_accepted + 1)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False
            if self.block_due_to_fragmentation:
                self.bit_rate_blocked_fragmentation += self.service.bit_rate
                self.service_blocked_due_fragmentation += 1
            else:
                self.bit_rate_blocked_lack_resource += self.service.bit_rate
                self.service_blocked_due_lack_resource += 1

        if not self.service.accepted:
            self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1

        self.services_processed += 1
        self._new_service = False
        self._next_service()

        # Periodical defragmentation
        if (self.services_processed % self.defragmentation_period) == 0:
            self.episode_defragmentation_procedure += 1
            services_to_defragment = self.topology.graph['running_services']
            defrag_counter = 0
            for i in range(10000):
                if i < len(services_to_defragment):
                    service_to_defrag = services_to_defragment[i]
                    num_slots = service_to_defrag.number_slots
                    old_initial_slot = service_to_defrag.initial_slot
                    new_slot = 10000  # large placeholder
                    for slot_candidate in range(
                        0, self.topology.graph['num_spectrum_resources'] - num_slots
                    ):
                        if self.is_path_free(
                            service_to_defrag.route, slot_candidate, num_slots
                        ):
                            new_slot = slot_candidate
                            break
                    # Try to find a slot closer to the old_initial_slot
                    if new_slot > old_initial_slot:
                        counter = num_slots
                        for slot_candidate in range(old_initial_slot - num_slots, old_initial_slot):
                            if slot_candidate >= 0:
                                if self.is_path_free(
                                    service_to_defrag.route,
                                    slot_candidate,
                                    counter
                                ):
                                    new_slot = slot_candidate
                                    break
                            counter -= 1

                    if new_slot < old_initial_slot:
                        self.episode_num_moves += 1
                        self._move_path(
                            service_to_defrag.route,
                            service_to_defrag,
                            new_slot,
                            old_initial_slot,
                            num_slots
                        )
                        defrag_counter += 1
                        if defrag_counter == self.movable_connections:
                            break

        # Update bit-rate processing stats
        if self.service.bit_rate in self.services_processed_bit_rate.keys():
            self.services_processed_bit_rate[self.service.bit_rate] += 1
        else:
            self.services_processed_bit_rate[self.service.bit_rate] = 1

        self.episode_services_processed += 1
        self.bit_rate_requested += self.service.bit_rate
        self.episode_bit_rate_requested += self.service.bit_rate

        self.topology.graph['services'].append(self.service)

        # Calculate fragmentation metrics
        sum_fragmentation_network_external = 0
        sum_fragmentation_network_compactness = 0
        fragmentation_external_links = np.zeros(self.topology.number_of_edges())
        fragmentation_compactness_links = np.zeros(self.topology.number_of_edges())

        for edge in self.topology.edges():
            sum_fragmentation_network_external += self.topology[edge[0]][edge[1]]['external_fragmentation']
            edge_id = self.topology[edge[0]][edge[1]]['id']
            fragmentation_external_links[edge_id] = self.topology[edge[0]][edge[1]]['external_fragmentation']

        external_fragmentation_network = (
            sum_fragmentation_network_external / self.topology.number_of_edges()
        )
        external_fragmentation_deviation = np.std(fragmentation_external_links)

        for edge in self.topology.edges():
            sum_fragmentation_network_compactness += self.topology[edge[0]][edge[1]]['compactness']
            edge_id = self.topology[edge[0]][edge[1]]['id']
            fragmentation_compactness_links[edge_id] = self.topology[edge[0]][edge[1]]['compactness']

        compactness_fragmentation_network = (
            sum_fragmentation_network_compactness / self.topology.number_of_edges()
        )
        compactness_fragmentation_deviation = np.std(fragmentation_compactness_links)

        # Reward
        reward = self.reward()

        info = {
            'service_blocking_rate':
                (self.services_processed - self.services_accepted) / self.services_processed,
            'episode_service_blocking_rate':
                (self.episode_services_processed - self.episode_services_accepted)
                / self.episode_services_processed,
            'bit_rate_blocking_rate':
                (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
            'episode_bit_rate_blocking_rate':
                (
                    self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
                ) / self.episode_bit_rate_requested,
            'external_fragmentation_network_episode': external_fragmentation_network,
            'compactness_fragmentation_network_episode': compactness_fragmentation_network,
            'compactness_network_fragmentation_network_episode': self.topology.graph['compactness'],
            'delay_deviation_absolute': self.average_delay_absolute,
            'delay_deviation_percentage': self.average_delay_percentage,
            'bit_rate_blocking_fragmentation':
                self.bit_rate_blocked_fragmentation / self.bit_rate_requested,
            'service_blocking_rate_fragmentation':
                self.service_blocked_due_fragmentation / self.services_processed,
            'compactness_fragmentation_deviation': compactness_fragmentation_deviation,
            'external_fragmentation_deviation': external_fragmentation_deviation,
            'service_blocking_rate_100':
                (self.services_processed_bit_rate[100] - self.services_accepted_bit_rate[100])
                / self.services_processed_bit_rate[100]
                if 100 in self.services_processed_bit_rate
                else 0,
            'service_blocking_rate_200':
                (self.services_processed_bit_rate[200] - self.services_accepted_bit_rate[200])
                / self.services_processed_bit_rate[200]
                if 200 in self.services_processed_bit_rate
                else 0,
            'service_blocking_rate_400':
                (self.services_processed_bit_rate[400] - self.services_accepted_bit_rate[400])
                / self.services_processed_bit_rate[400]
                if 400 in self.services_processed_bit_rate
                else 0,
            'service_blocked_eopisode':
                (self.episode_services_processed - self.episode_services_accepted),
            'number_movements_episode': self.episode_num_moves,
            'number_defragmentation_procedure_episode': self.episode_defragmentation_procedure,
            'number_arrivals': self.services_processed,
        }

        return self.observation(), reward, (self.episode_services_processed == self.episode_length), info

    def reset(self, only_counters=True):
        """
        Resets the environment to an initial state.
        If only_counters is True, only counters and relevant stats are reset,
        but the topology is not re-initialized.
        """
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_num_moves = 0
        self.episode_defragmentation_procedure = 0
        self.episode_actions_output = np.zeros(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action),
            dtype=int
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action),
            dtype=int
        )

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            dtype=int
        )

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=np.int_,
        )

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0

        for _, link_data in self.topology.edges.items():
            link_data['external_fragmentation'] = 0.0
            link_data['compactness'] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        """
        Renders the environment state. Currently not implemented.
        """
        return

    def _provision_path(self, path: Path, initial_slot, number_slots):
        """
        Provisions a path for the current service on the given slot range.
        """
        if not self.is_path_free(path, initial_slot, number_slots):
            raise ValueError(
                f"Path {path.node_list} has not enough capacity on slots "
                f"{initial_slot}-{initial_slot + number_slots}"
            )

        self.logger.debug(
            f'{self.service.service_id} assigning path {path.node_list} '
            f'on initial slot {initial_slot} for {number_slots} slots'
        )
        for i in range(len(path.node_list) - 1):
            link_index = self.topology[path.node_list[i]][path.node_list[i + 1]]['index']
            self.topology.graph['available_slots'][link_index, initial_slot:initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[link_index, initial_slot:initial_slot + number_slots] = self.service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])

        self.topology.graph['running_services'].append(self.service)
        self.service.route = path
        self.service.initial_slot = initial_slot
        self.service.number_slots = number_slots
        self._update_network_stats()

        self.services_accepted += 1

        if self.service.bit_rate in self.services_accepted_bit_rate.keys():
            self.services_accepted_bit_rate[self.service.bit_rate] += 1
        else:
            self.services_accepted_bit_rate[self.service.bit_rate] = 1

        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.service.bit_rate
        self.episode_bit_rate_provisioned += self.service.bit_rate

    def _release_path(self, service: Service):
        """
        Releases the spectrum slots previously allocated to the given service.
        """
        for i in range(len(service.route.node_list) - 1):
            link_index = self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index']
            self.topology.graph['available_slots'][link_index,
                                                   service.initial_slot:service.initial_slot + service.number_slots] = 1
            self.spectrum_slots_allocation[link_index,
                                           service.initial_slot:service.initial_slot + service.number_slots] = -1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])

        self.topology.graph['running_services'].remove(service)

    def _move_path(self, path: Path, service: Service, new_initial_slot, old_initial_slot, number_slots):
        """
        Moves the allocated slots for a given service from old_initial_slot to new_initial_slot in a make-before-break manner.
        """
        self.logger.debug(
            f'{service.service_id} moving path {path.node_list} '
            f'on initial slot {new_initial_slot} for {number_slots} slots'
        )
        # Provision the new slot range
        for i in range(len(path.node_list) - 1):
            link_index = self.topology[path.node_list[i]][path.node_list[i + 1]]['index']
            self.topology.graph['available_slots'][link_index, new_initial_slot:new_initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[link_index, new_initial_slot:new_initial_slot + number_slots] = service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(service)

        service.initial_slot = new_initial_slot
        self.topology.graph['running_services'].append(service)

        # Release the old slot range
        for i in range(len(service.route.node_list) - 1):
            link_index = self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index']
            if old_initial_slot - new_initial_slot >= number_slots:
                self.topology.graph['available_slots'][link_index,
                                                       old_initial_slot:old_initial_slot + service.number_slots] = 1
                self.spectrum_slots_allocation[link_index,
                                               old_initial_slot:old_initial_slot + service.number_slots] = -1
            else:
                self.topology.graph['available_slots'][
                    link_index, new_initial_slot + number_slots:old_initial_slot + service.number_slots
                ] = 1
                self.spectrum_slots_allocation[
                    link_index, new_initial_slot + number_slots:old_initial_slot + service.number_slots
                ] = -1

            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].remove(service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].remove(service)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])

        self.topology.graph['running_services'].remove(service)
        self._update_network_stats()

    def _update_network_stats(self):
        """
        Updates network-level statistics (throughput, compactness) based on the current time.
        """
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph['throughput']
            last_compactness = self.topology.graph['compactness']

            cur_throughput = 0.0
            for srv in self.topology.graph["running_services"]:
                cur_throughput += srv.bit_rate

            throughput = (
                (last_throughput * last_update) + (cur_throughput * time_diff)
            ) / self.current_time
            self.topology.graph['throughput'] = throughput

            compactness = (
                (last_compactness * last_update)
                + (self._get_network_compactness() * time_diff)
            ) / self.current_time
            self.topology.graph['compactness'] = compactness

        self.topology.graph['last_update'] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        """
        Updates the utilization and fragmentation metrics for a given link.
        """
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            link_index = self.topology[node1][node2]['index']
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (
                self.num_spectrum_resources
                - np.sum(self.topology.graph['available_slots'][link_index, :])
            ) / self.num_spectrum_resources
            utilization = (
                (last_util * last_update) + (cur_util * time_diff)
            ) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

            slot_allocation = self.topology.graph['available_slots'][link_index, :]

            # Implementing external fragmentation from:
            # https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2]['external_fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_external_fragmentation = 0.0
            cur_link_compactness = 0.0

            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMSAEnv.rle(slot_allocation)

                # Compute external fragmentation
                unused_blocks = [idx for idx, val in enumerate(values) if val == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1.0 - float(max_empty) / float(np.sum(slot_allocation))

                # Compute link spectrum compactness
                used_blocks = [idx for idx, val in enumerate(values) if val == 0]
                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    # Evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                        slot_allocation[lambda_min:lambda_max]
                    )
                    unused_spectrum_slots = np.sum(internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = (
                            (lambda_max - lambda_min)
                            / np.sum(1 - slot_allocation)
                        ) * (1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.0
                else:
                    cur_link_compactness = 1.0

            external_fragmentation = (
                (last_external_fragmentation * last_update)
                + (cur_external_fragmentation * time_diff)
            ) / self.current_time
            self.topology[node1][node2]['external_fragmentation'] = external_fragmentation

            link_compactness = (
                (last_compactness * last_update) + (cur_link_compactness * time_diff)
            ) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness

        self.topology[node1][node2]['last_update'] = self.current_time

    def _next_service(self):
        """
        Generates the next service in the arrival process, possibly releasing any
        services whose holding time has expired.
        """
        if self._new_service:
            return

        # Incremental vs. dynamic traffic
        if random.random() < (self.incremental_traffic_percentage / 100):
            at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
            self.current_time = at
            ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        else:
            at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time_dynamic)
            self.current_time = at
            ht = self.rng.expovariate(
                1 / (self.mean_service_holding_time / self.incremental_dynamic_proportion_mean_time)
            )

        src, src_id, dst, dst_id = self._get_node_pair()

        # Developing bit-rate for Telia
        bit_rate_threshold = self.rng.random()
        if bit_rate_threshold < 0.5:
            bit_rate = 100
        elif 0.5 <= bit_rate_threshold < 0.8:
            bit_rate = 200
        else:
            bit_rate = 400

        # Release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:
                # Release is not to be processed yet
                self._add_release(service_to_release)  # Re-insert in the queue
                break

        self.service = Service(
            self.episode_services_processed,
            src, src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate
        )
        self._new_service = True

    def _get_path_slot_id(self, action: int):
        """
        Decodes a single integer action index into the path index and the slot index.
        """
        path = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return path, initial_slot

    def get_number_slots(self, path: Path) -> int:
        """
        Computes the number of spectrum slots necessary to accommodate the
        service request into the given path, including guard bands.
        """
        return math.ceil(self.service.bit_rate / path.best_modulation['capacity']) + 1

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        """
        Checks if the path has the required free contiguous slots in the specified range.
        """
        if initial_slot + number_slots > self.num_spectrum_resources:
            return False
        for i in range(len(path.node_list) - 1):
            link_index = self.topology[path.node_list[i]][path.node_list[i + 1]]['index']
            if np.any(
                self.topology.graph['available_slots'][link_index, initial_slot:initial_slot + number_slots] == 0
            ):
                return False
        return True

    def get_available_slots(self, path: Path):
        """
        Returns a bitmask array of available (1) or occupied (0) slots
        across all links of the given path.
        """
        link_indices = [
            self.topology[path.node_list[i]][path.node_list[i + 1]]['id']
            for i in range(len(path.node_list) - 1)
        ]
        return functools.reduce(
            np.multiply,
            self.topology.graph["available_slots"][link_indices, :]
        )

    @staticmethod
    def rle(inarray):
        """
        Run-length encoding. Returns positions, values, and run-lengths.
        Adapted from:
        https://stackoverflow.com/questions/1066758/
        find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        """
        ia = np.asarray(inarray)
        n = len(ia)
        if n == 0:
            return None, None, None
        else:
            y = np.array(ia[1:] != ia[:-1])
            i = np.append(np.where(y), n - 1)
            z = np.diff(np.append(-1, i))
            p = np.cumsum(np.append(0, z))[:-1]
            return p, ia[i], z

    def _get_network_compactness(self) -> float:
        """
        Implements network spectrum compactness from:
        https://ieeexplore.ieee.org/abstract/document/6476152
        """
        sum_slots_paths = 0  # Sum of (Bi * Hi) for each service
        for srv in self.topology.graph["running_services"]:
            sum_slots_paths += srv.number_slots * srv.route.hops

        sum_occupied = 0
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            link_index = self.topology[n1][n2]['index']
            initial_indices, values, lengths = RMSAEnv.rle(
                self.topology.graph['available_slots'][link_index, :]
            )
            used_blocks = [idx for idx, val in enumerate(values) if val == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += lambda_max - lambda_min

                # Evaluate the "used part" of the spectrum
                internal_idx, internal_values, _ = RMSAEnv.rle(
                    self.topology.graph['available_slots'][link_index, lambda_min:lambda_max]
                )
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0 and sum_slots_paths > 0:
            cur_spectrum_compactness = (
                (sum_occupied / sum_slots_paths)
                * (self.topology.number_of_edges() / sum_unused_spectrum_blocks)
            )
        else:
            cur_spectrum_compactness = 1.0

        return cur_spectrum_compactness


# -----------------------------------------------------------------------
# Some example path-selection policies for RMSAEnv
# -----------------------------------------------------------------------

def shortest_path_first_fit(env: RMSAEnv, service: Optional[Service] = None) -> list:
    """
    Always selects the shortest path (index 0) and does a first-fit on slots.
    """
    if service is None:
        service = env.service
    num_slots = env.get_number_slots(env.k_shortest_paths[service.source, service.destination][0])

    for initial_slot in range(env.topology.graph['num_spectrum_resources'] - num_slots):
        if env.is_path_free(env.k_shortest_paths[service.source, service.destination][0],
                            initial_slot, num_slots):
            return [0, initial_slot]

    free_slots = np.sum(env.get_available_slots(env.k_shortest_paths[service.source,
                                                                     service.destination][0]))
    if num_slots < free_slots:
        env.block_due_to_fragmentation = True
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]


def shortest_available_path_first_fit(env: RMSAEnv) -> list:
    """
    Tries each path in ascending order, and uses the first-fit approach for slots.
    """
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                return [idp, initial_slot]

    free_slots = np.sum(env.get_available_slots(path))
    if num_slots < free_slots:
        env.block_due_to_fragmentation = True
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]


def least_loaded_path_first_fit(env: RMSAEnv) -> list:
    """
    Chooses the path with the largest number of free slots (least loaded),
    then does a first-fit on that path.
    """
    max_free_slots = 0
    flag_assigned_path = False
    action = [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]

    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = [idp, initial_slot]
                    max_free_slots = free_slots
                    flag_assigned_path = True
                # Only need the first valid slot to measure free slots
                break

    if not flag_assigned_path:
        free_slots = np.sum(env.get_available_slots(path))
        if num_slots < free_slots:
            # Potential fragmentation
            env.block_due_to_fragmentation = True

    return action


def Fragmentation_alignment_aware_RMSA(env: RMSAEnv) -> list:
    """
    A fragmentation-alignment-aware RMSA heuristic that:
    - Considers the first path only (shortest path).
    - For each possible slot, computes a 'fragment_alignment_factor'
      to decide the best placement.
    """
    min_factor = math.inf
    flag_assigned_path = False
    action = [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]

    # If you wish to consider multiple paths, iterate over them
    # For brevity, we consider only the shortest path here
    shortest_path = [env.k_shortest_paths[env.service.source, env.service.destination][0]]
    for idp, path in enumerate(shortest_path):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                number_cuts = env.get_number_cuts(path, initial_slot, num_slots)
                number_alignment_factor = env.get_number_alignment_factor(path, initial_slot, num_slots)
                fragment_alignment_factor = (
                    (path.hops * num_slots + number_cuts + number_alignment_factor) / free_slots
                    if free_slots > 0
                    else math.inf
                )

                # Estimate remaining times if adjacent slots are occupied
                remaining_times = 0
                for i in range(len(path.node_list) - 1):
                    link_index = env.topology[path.node_list[i]][path.node_list[i + 1]]['index']
                    for check_slot in [initial_slot - 1, initial_slot + num_slots]:
                        if 0 <= check_slot < env.num_spectrum_resources:
                            if env.topology.graph['available_slots'][link_index, check_slot] == 0:
                                srv_id = env.spectrum_slots_allocation[link_index, check_slot]
                                # If service is still running
                                potential_service = env.topology.graph['services'][srv_id]
                                residual = (potential_service.arrival_time + potential_service.holding_time) \
                                           - env.current_time
                                if residual > 0:
                                    remaining_times += residual

                # Optionally incorporate remaining_times
                # fragment_alignment_factor = (
                #     (path.hops * num_slots + number_cuts + number_alignment_factor)
                #     / (free_slots + remaining_times)
                #     if (free_slots + remaining_times) > 0
                #     else math.inf
                # )

                if fragment_alignment_factor < min_factor:
                    action = [idp, initial_slot]
                    min_factor = fragment_alignment_factor
                    flag_assigned_path = True

    if not flag_assigned_path:
        free_slots = np.sum(env.get_available_slots(path))
        if num_slots < free_slots:
            env.block_due_to_fragmentation = True

    return action


# -----------------------------------------------------------------------
# Observation and Action Wrappers
# -----------------------------------------------------------------------

class SimpleMatrixObservation(gym.ObservationWrapper):
    """
    Observation wrapper that flattens the environment's state into a 1D array.
    """

    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        shape = (
            self.env.topology.number_of_nodes() * 2
            + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(shape,), dtype=np.uint8
        )
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(self.env.service.source_id, self.env.service.destination_id)
        max_node = max(self.env.service.source_id, self.env.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1

        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])

        return np.concatenate(
            (
                source_destination_tau.reshape((1, -1)),
                spectrum_obs.reshape((1, -1))
            ),
            axis=1
        ).reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):
    """
    Action wrapper that only selects the path index (discrete),
    while always doing first-fit on slots within that path.
    """

    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    def action(self, action):
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(
                self.env.k_shortest_paths[self.env.service.source, self.env.service.destination][action]
            )
            for initial_slot in range(
                self.env.topology.graph['num_spectrum_resources'] - num_slots
            ):
                if self.env.is_path_free(
                    self.env.k_shortest_paths[self.env.service.source, self.env.service.destination][action],
                    initial_slot, num_slots
                ):
                    return [action, initial_slot]
        return [
            self.env.topology.graph['k_paths'],
            self.env.topology.graph['num_spectrum_resources']
        ]

    def step(self, action):
        return self.env.step(self.action(action))
