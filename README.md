# Optical RL-Gym

[OpenAI Gym](https://gym.openai.com/) is the de-facto interface for reinforcement learning environments.
Optical RL-Gym builds on top of OpenAI Gym's interfaces to create a set of environments that model optical network problems such as resource management and reconfiguration.
Optical RL-Gym can be used to quickly start experimenting with reinforcement learning in optical network problems.
Later, you can use the pre-defined environments to create more specific environments for your particular use case.

This project uses the [optical-rl-gym](https://github.com/carlosnatalino/optical-rl-gym), and adds the ability of proactive spectrum defragmentation to it.
Please use the following bibtex:

```
@article{etezadi2023deep,
  title={Deep reinforcement learning for proactive spectrum defragmentation in elastic optical networks},
  author={Etezadi, Ehsan and Natalino, Carlos and Diaz, Renzo and Lindgren, Anders and Melin, Stefan and Wosinska, Lena and Monti, Paolo and Furdek, Marija},
  journal={Journal of Optical Communications and Networking},
  volume={15},
  number={10},
  pages={E86--E96},
  year={2023},
  publisher={Optica Publishing Group}
}
```

## Features

Across all the environments, the following features are available:

- Use of [NetworkX](https://networkx.github.io/) for the topology graph representation, resource and path computation.
- Uniform and non-uniform traffic generation.
- Flag to let agents proactively reject requests or not.
- Appropriate random number generation with seed management providing reproducibility of results.




**To traing reinforcement learning agents, you must create or install reinforcement learning agents. Here are some of the libraries containing RL agents:**
- [Stable baselines](https://github.com/hill-a/stable-baselines)
- [OpenAI Baselines](https://github.com/openai/baselines) -- in maintenance mode
- [ChainerRL](https://github.com/chainer/chainerrl)
- [TensorFlow Agents](https://www.tensorflow.org/agents)


Training a RL agent for one of the Optical RL-Gym environments can be done with a few lines of code.

For instance, you can use a [Stable Baselines](https://github.com/hill-a/stable-baselines) agent trained for the RMSA environment:

```python
# define the parameters of the RMSA environment
env_args = dict(topology=topology, seed=10, allow_rejection=False, 
                load=50, episode_length=50)
# create the environment
gym.make('Defragmentation-v0', **env_args)
# create the agent
agent = PPO2(MlpPolicy, env)
# run 10k learning timesteps
agent.learn(total_timesteps=10000)
```







