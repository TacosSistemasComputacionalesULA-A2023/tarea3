import os

import gym
import gym_taco_environments
import time
from agent import ValueIteration
import numpy as np
import csv

# Allowing environment to have sounds
if "SDL_AUDIODRIVER" in os.environ:
    del os.environ["SDL_AUDIODRIVER"]


def sums(policies, values, averages):
    for i in range(len(policies)):
        average_values["policies_values"][i] += policies[i]

    for i in range(len(values)):
        average_values["states_values"][i] += values[i]


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0", render_mode="ansi", delay=0)
    space_num = env.observation_space.n

    experiments = 100

    iterations = [100, 1000, 10000, 100000]
    evaluations = [100, 1000, 10000, 100000]
    methods = ["valiter", "politer"]
    init_values = [
        (np.zeros(space_num), (np.zeros(space_num))),
        (np.random.rand(space_num), np.random.randint(0, 4, space_num)),
        ([1 for _ in range(space_num)], [1 for _ in range(space_num)]),
    ]
    fieldnames = [
        "iter_num",
        "eval_num",
        "method",
        "gamma",
        "delta",
        "states_values",
        "policies_values",
        "reward",
    ]

    f = open("summary.csv", "a")
    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for method in methods:
        for evaluation in evaluations:
            for iteration in iterations:
                for gamma in np.arange(0.1, 1.0, 0.1):
                    for delta in np.arange(0.1, 1.0, 0.1):
                        for init_value in init_values:
                            average_values = {
                                "states_values": [0 for _ in range(space_num)],
                                "policies_values": [0 for _ in range(space_num)],
                                "reward": 0,
                            }

                            for _ in range(experiments):
                                agent = ValueIteration(
                                    states_n=space_num,
                                    actions_n=env.action_space.n,
                                    P=env.P,
                                    gamma=gamma,
                                )

                                agent.reset(init_value[0], init_value[1])

                                agent.solve(
                                    policy_evaluations=evaluation,
                                    iterations=iteration,
                                    delta=delta,
                                    method=methods,
                                )

                                # agent.render()
                                observation, info = env.reset()
                                terminated, truncated = False, False

                                env.render()

                                while not (terminated or truncated):
                                    action = agent.get_action(observation)
                                    (
                                        observation,
                                        reward,
                                        terminated,
                                        truncated,
                                        _,
                                    ) = env.step(action)

                                average_values["reward"] += reward
                                sums(agent.policy, agent.values, average_values)
                                env.close()

                            f = open("summary.csv", "a")
                            with f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                for i in range(len(average_values["policies_values"])):
                                    average_values["policies_values"][i] /= experiments

                                for i in range(len(average_values["states_values"])):
                                    average_values["states_values"][i] /= experiments

                                average_values["reward"] /= experiments

                                average_values["method"] = method

                                average_values["iter_num"] = iteration

                                average_values["eval_num"] = evaluation
                                
                                average_values["gamma"] = gamma
                                
                                average_values["delta"] = delta

                                writer.writerow(average_values)
