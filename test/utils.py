import numpy as np


def contains_exactly(first, second):
    used_indices = np.zeros(len(second))
    for item1 in first:
        found = False
        for index, item2 in enumerate(second):
            if item1 == item2 and used_indices[index] == 0:
                found = True
                used_indices[index] = 1
                break
        
        if not found:
            return False

    return True