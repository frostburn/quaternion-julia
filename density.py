import numpy as np
import quaternion

def illuminate_and_absorb(slices, source, background, du):
    result = None
    for q in slices:
        if result is None:
            z = np.zeros(q.shape)
            result = np.array([z + b for b in background])
        illumination, absorption = source(q)
        result += illumination*du
        result *= np.exp(-absorption*du)
    return result
