import numpy as np

def cos_interpolation(state_data, cos_delta=10):
    for k in range(len(state_data)):
        for i in range(len(state_data[k])):
            for j in range(state_data[k].shape[1]):
                if state_data[k, i, j] == 1:
                    x = np.arange(i-cos_delta, i+cos_delta)
                    y = 0.5*np.cos(2*np.pi/(cos_delta*2)*(x - i))+0.5
                    state_data[k, i-cos_delta:i+cos_delta, j] = y
    return state_data


# def cos_interpolation(state_data, cos_delta=10):
#     rows, cols, channels = state_data.shape
#     indices_i, indices_j = np.nonzero(state_data[:, :, 0] == 1)

#     for i, j in zip(indices_i, indices_j):
#         x = np.arange(i - cos_delta, i + cos_delta)
#         y = 0.5 * np.cos(2 * np.pi / (cos_delta * 2) * (x - i)) + 0.5

#         # Clip indices to stay within bounds
#         start_idx = max(0, i - cos_delta)
#         end_idx = min(rows, i + cos_delta)

#         # Ensure y has the correct shape for broadcasting
#         y_broadcast = y[:end_idx - start_idx, np.newaxis]

#         state_data[start_idx:end_idx, j, :] = y_broadcast

#     return state_data