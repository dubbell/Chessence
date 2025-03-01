import numpy as np
from typing import List


class CustomAssert:
    
    def assertContainsExactly(self, iter1, iter2):

        used_indices = np.zeros(len(iter2))
        for item1 in iter1:
            found = False
            for index, item2 in enumerate(iter2):
                if (isinstance(item1, List) or isinstance(item1, np.ndarray)
                        or isinstance(item2, List) or isinstance(item2, np.ndarray)):
                    if ((isinstance(item1, List) or isinstance(item1, np.ndarray)
                        or isinstance(item2, List) or isinstance(item2, np.ndarray))
                            and len(item1) == len(item2)
                            and np.equal(item1, item2).all()):
                        found = True
                        used_indices[index] = 1
                        break
                elif item1 == item2 and used_indices[index] == 0:
                    found = True
                    used_indices[index] = 1
                    break
            
            if not found:
                raise AssertionError(f"{item1} in first iterable not in second.")

        for index in range(len(iter2)):
            if used_indices[index] == 0:
                raise AssertionError(f"{iter2[index]} in second iterable not in first.")