import os

import gym
import gym_taco_environments
import time
from agent import ValueIteration
import numpy as np
import csv
import multiprocessing

# Allowing environment to have sounds
if "SDL_AUDIODRIVER" in os.environ:
    del os.environ["SDL_AUDIODRIVER"]

fieldnames = [
    "iter_num",
    "method",
    "gamma",
    "delta",
    "states_values",
    "policies_values",
    "reward",
]


class agent_arguments:
    """
    agent_arguments allows to encapsulate all of the parameters needed to run a
    new episode for our agent and doing concurrently on the system that is
    running it.
    """

    def __init__(self, env, experiments, iterations, method, init_values, gamma, delta):
        self.experiments = experiments
        self.iterations = iterations
        self.method = method
        self.init_values = init_values
        self.gamma = gamma
        self.delta = delta
        self.env = env

    def __str__(self) -> str:
        return f"<{self.iterations}|{self.method}|{self.gamma}|{self.delta}>"


# sums allow us to get the sum of all the policies obtained in each experiment.
# So that later we can calculate the average.
def sums(policies, values, averages):
    for i in range(len(policies)):
        averages["policies_values"][i] += policies[i]

    for i in range(len(values)):
        averages["states_values"][i] += values[i]


# run learning process takes the arguments for the agent and runs the specified
# amount of experiments also while returning summaries from that experiments set.
def run_learning_process(arguments: agent_arguments):
    space_num = arguments.env.observation_space.n

    average_values = {
        "states_values": [0 for _ in range(space_num)],
        "policies_values": [0 for _ in range(space_num)],
        "reward": 0,
    }

    for _ in range(arguments.experiments):
        agent = ValueIteration(
            states_n=space_num,
            actions_n=arguments.env.action_space.n,
            P=arguments.env.P,
            gamma=arguments.gamma,
        )

        agent.reset(arguments.init_values[0], arguments.init_values[1])

        agent.solve(
            policy_evaluations=arguments.iterations,
            iterations=arguments.iterations,
            delta=arguments.delta,
            method=arguments.method,
        )

        # agent.render()
        observation, _ = arguments.env.reset()
        terminated, truncated = False, False

        arguments.env.render()

        reward = 0
        while not (terminated or truncated):
            action = agent.get_action(observation)
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = arguments.env.step(action)

        average_values["reward"] += reward
        sums(agent.policy, agent.values, average_values)
        arguments.env.close()
    for i in range(len(average_values["policies_values"])):
        average_values["policies_values"][i] /= arguments.experiments

    for i in range(len(average_values["states_values"])):
        average_values["states_values"][i] /= arguments.experiments

    average_values["reward"] /= arguments.experiments

    average_values["method"] = arguments.method

    average_values["iter_num"] = arguments.iterations

    average_values["gamma"] = arguments.gamma

    average_values["delta"] = arguments.delta

    return average_values


if __name__ == "__main__":

    env = gym.make("FrozenLake-v0", render_mode="ansi", delay=0)

    experiments = 100

    iterations = [100, 1000, 10000]
    methods = ["valiter", "politer"]
    space_num = env.observation_space.n
    init_values = [
        (np.zeros(space_num), (np.zeros(space_num))),
        (np.random.rand(space_num), np.random.randint(0, 4, space_num)),
    ]

    f = open("summary.csv", "a", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    arguments = []

    for method in methods:
        for iteration in iterations:
            for gamma in np.arange(0.1, 1.0, 0.1):
                for delta in np.arange(0.1, 1.0, 0.1):
                    for init_value in init_values:
                        arguments.append(
                            agent_arguments(
                                iterations=iteration,
                                init_values=init_value,
                                gamma=gamma,
                                delta=delta,
                                method=method,
                                env=env,
                                experiments=experiments,
                            )
                        )

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = p.map_async(run_learning_process, arguments)
    values = result.get()

    f = open("summary.csv", "a", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerows(values)
