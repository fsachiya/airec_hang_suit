import math
import numpy as np

def moving_average(data, size):
    b = np.ones(size)/size
    data_mean = np.convolve(data, b, mode="same")

    n_conv = math.ceil(size/2)

    # 補正部分
    data_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        data_mean[i] *= size/(i+n_conv)
        data_mean[-i] *= size/(i + n_conv - (size % 2)) 
	# size%2は奇数偶数での違いに対応するため
    return data_mean