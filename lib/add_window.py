import numpy as np
import torch

def Add_Window_Horizon(data, window=3, horizon=1):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    X = []      #windows
    Y = []      #horizon

    data = torch.Tensor(data).to('cuda')
    if torch.is_tensor(data):
        X = torch.stack([data[i:i-window-horizon+1] for i in range(window)]).swapaxes(0, 1)
        Y = data[window+horizon-1:][..., np.newaxis].swapaxes(1, 2)
    else:
        X = np.stack([data[i:i-window-horizon+1] for i in range(window)]).swapaxes(0, 1)
        Y = np.array(data[window+horizon-1:])[..., np.newaxis].swapaxes(1, 2)

    if torch.is_tensor(data):
        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
    return X, Y
