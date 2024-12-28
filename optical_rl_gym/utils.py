from dataclasses import dataclass
from itertools import islice
import copy
import itertools

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Path:
    """
    Simple class to store path information in an optical network.
    """

    def __init__(self, path_id, node_list, length, best_modulation=None):
        """
        :param path_id: Identifier of the path
        :param node_list: List of nodes in the path
        :param length: Path length
        :param best_modulation: Modulation format (dict or other data structure)
        """
        self.path_id = path_id
        self.node_list = node_list
        self.length = length
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1


class Service:
    """
    Class to represent a requested service in the network.
    """

    def __init__(
            self,
            service_id,
            source,
            source_id,
            destination=None,
            destination_id=None,
            arrival_time=None,
            holding_time=None,
            bit_rate=None,
            best_modulation=None,
            service_class=None,
            number_slots=None,
    ):
        """
        :param service_id: Service identifier
        :param source: Source node (label)
        :param source_id: Source node ID (int)
        :param destination: Destination node (label)
        :param destination_id: Destination node ID (int)
        :param arrival_time: Arrival time of the service
        :param holding_time: Holding (duration) time of the service
        :param bit_rate: Bit rate required (Gb/s or similar)
        :param best_modulation: Best modulation format
        :param service_class: Class/type of the service
        :param number_slots: Number of spectrum slots required
        """
        self.service_id = service_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.bit_rate = bit_rate
        self.service_class = service_class
        self.best_modulation = best_modulation
        self.number_slots = number_slots
        self.route = None
        self.initial_slot = None
        self.accepted = False

    def __str__(self):
        """
        String representation of the service object.
        """
        msg = '{'
        msg += '' if self.bit_rate is None else f'br: {self.bit_rate}, '
        msg += '' if self.service_class is None else f'cl: {self.service_class}, '
        return f'Serv. {self.service_id} ({self.source} -> {self.destination})' + msg


def start_environment(env, steps):
    """
    Resets and runs the environment for a certain number of steps,
    taking random actions until the episode is done.
    """
    done = True
    for _ in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Returns the k shortest simple paths from source to target in graph G.
    Uses NetworkX's shortest_simple_paths generator internally.
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight='length'):
    """
    Given a graph, a path, and an edge weight key, returns the sum of the
    specified edge weights along that path.
    """
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def random_policy(env):
    """
    Samples a random action from the environment's action space.
    """
    return env.action_space.sample()


def evaluate_heuristic(
        env,
        heuristic,
        n_eval_episodes=1,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=False,
):
    """
    Evaluates a given heuristic (policy) over n_eval_episodes episodes.

    :param env: The environment instance.
    :param heuristic: A function that takes `env` and returns an action.
    :param n_eval_episodes: Number of evaluation episodes.
    :param render: Whether to render each step.
    :param callback: Optional callback function.
    :param reward_threshold: If specified, asserts that the mean reward is above this threshold.
    :param return_episode_rewards: If True, returns (episode_rewards, episode_lengths).
    :return: Mean reward, std reward, or (episode_rewards, episode_lengths) if return_episode_rewards is True.
    """
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        state = None
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action = heuristic(env)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            f'Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}'
        )

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def plot_spectrum_assignment(
        topology,
        vector,
        values=False,
        filename=None,
        show=True,
        figsize=(15, 10),
        title=None,
):
    """
    Plots a 2D color map (matrix) of the spectrum assignment across links.

    :param topology: A NetworkX graph with an 'edges' attribute to identify link ids.
    :param vector: A 2D NumPy array with shape (num_links, num_slots).
    :param values: If True, plots the numeric values on each cell.
    :param filename: If specified, saves the figure to disk.
    :param show: Whether to display the plot interactively.
    :param figsize: Size of the figure (width, height).
    :param title: Title of the plot.
    """
    plt.figure(figsize=figsize)
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under(color='white')
    cmap_reverse = plt.cm.viridis_r
    cmap_reverse.set_under(color='black')

    plt.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')

    if values:
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] == -1:
                continue
            text = f'{int(vector[i, j]):.0f}'
            color = cmap_reverse(vector[i, j] / vector.max())
            plt.text(
                j + 0.5,
                i + 0.5,
                text,
                horizontalalignment="center",
                verticalalignment='center',
                color=color,
            )

    plt.xlabel('Frequency slot')
    plt.ylabel('Link')
    if title is not None:
        plt.title(title)

    plt.yticks(
        [topology.edges[link]['id'] + 0.5 for link in topology.edges()],
        [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()]
    )
    plt.tight_layout()
    plt.xticks(
        [x + 0.5 for x in plt.xticks()[0][:-1]],
        [x for x in plt.xticks()[1][:-1]]
    )

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()


def plot_spectrum_assignment_and_waste(
        topology,
        vector,
        vector_wasted,
        values=False,
        filename=None,
        show=True,
        figsize=(15, 10),
        title=None,
):
    """
    Plots a 2D color map for both the spectrum assignment (vector) and the
    wasted slots (vector_wasted), overlaying them in the same figure.

    :param topology: A NetworkX graph with an 'edges' attribute to identify link ids.
    :param vector: A 2D NumPy array with shape (num_links, num_slots).
    :param vector_wasted: A 2D NumPy array indicating "wasted" slots.
    :param values: If True, plots the numeric values on each cell.
    :param filename: If specified, saves the figure to disk.
    :param show: Whether to display the plot interactively.
    :param figsize: Size of the figure (width, height).
    :param title: Title of the plot.
    """
    plt.figure(figsize=figsize)
    cmap = copy.deepcopy(plt.cm.viridis)
    cmap.set_under(color='white')
    cmap_reverse = copy.copy(plt.cm.viridis_r)
    cmap_reverse.set_under(color='black')

    plt.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')

    if values:
        fmt = 'd'
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] != -1:
                text = format(vector[i, j], fmt)
                color = cmap_reverse(vector[i, j] / vector.max())
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    horizontalalignment="center",
                    verticalalignment='center',
                    color=color,
                )
            if vector_wasted[i, j] != -1:
                text = format(vector_wasted[i, j], fmt)
                plt.text(
                    j + 0.2,
                    i + 0.5,
                    text,
                    horizontalalignment="center",
                    verticalalignment='center',
                    color='red',
                )

    plt.xlabel('Frequency slot')
    plt.ylabel('Link')
    plt.yticks(
        [topology.edges[link]['id'] + 0.5 for link in topology.edges()],
        [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()]
    )
    plt.tight_layout()
    plt.xticks(
        [x + 0.5 for x in plt.xticks()[0][:-1]],
        [x for x in plt.xticks()[1][:-1]]
    )

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()


@dataclass
class DefragmentationOption:
    """
    Data class holding information about a defragmentation option.
    """
    service: Service
    starting_slot: int
    size_of_free_block: int
    start_of_block: bool
    left_side_free_slots: int
    right_side_free_slots: int


def find_nearest(array, value):
    """
    Finds the index of the element in `array` closest to `value`.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
