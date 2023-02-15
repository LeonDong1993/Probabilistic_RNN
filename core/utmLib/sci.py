import numpy as np
import hashlib

def hash_array(arr, n = 1000):
    data = arr.reshape(-1)
    if data.size > n:
        step = int(data.size / n)
        stats = [np.mean(data[s:(s+step)]) for s in range(0, data.size, step)]
        data = np.array(stats)
    data_digest = hashlib.md5(data.tobytes()).hexdigest()
    return data_digest