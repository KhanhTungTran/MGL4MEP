import os
import numpy as np

def load_st_dataset(data_path):
    #output B, N, D
    try:
        data = np.load(data_path)
    except Exception as _:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    # print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

