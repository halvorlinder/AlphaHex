from typing import Callable
import numpy as np
import random


def epsilon_greedy_choise(vector: list[float], mask: list[bool], epsilon: float, selector: Callable[[list[float]], int] = np.argmax):
    choose_random = random.random()
    if choose_random < epsilon:
        true_idx = np.argwhere(mask)
        random_idx = random.randint(0, len(true_idx) - 1)
        return true_idx[random_idx][0]
    else:
        return selector(vector)


def filter_and_normalize(vector: list[float], mask: list[bool]) -> list[float]:
    filtered = [value if legal else 0 for (value, legal) in zip(vector, mask)]
    return filtered/sum(filtered)
