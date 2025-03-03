import torch
import torch.nn as nn

# データを生成
data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [1.0, 2.0, 3.0, 4.0, 5.0],
                     [1.0, 2.0, 3.0, 4.0, 5.0]])  # 例として適当な数値を用意

# 移動平均を計算するための畳み込みフィルターを作成
kernel_size = 7  # ウィンドウサイズ
conv = nn.Conv1d(3, 3, kernel_size, padding=3, bias=False, padding_mode="replicate")  # 1次元畳み込み層を定義
conv.weight.data.fill_(1.0 / kernel_size)  # 重みを平均化

# データを適切な形状に変形
data = data.unsqueeze(0)  # バッチとチャンネルの次元を追加  .unsqueeze(0)
data = 

# 移動平均を計算
result = conv(data)

import ipdb; ipdb.set_trace()
# ウィンドウサイズ
window_size = 5
# ウィンドウサイズに基づいて移動平均を計算
moving_avg = result.unfold(2, window_size, 1).mean(dim=2)

# 結果を表示
print(result.squeeze())
import ipdb; ipdb.set_trace()



# w = [[[ 0.0534, -0.0963, -0.2767],
#       [ 0.2720, -0.1101,  0.4079]],
#      [[-0.2524,  0.0338, -0.0296],
#       [-0.3233,  0.3560,  0.2883]]] # 2,2,3
# w_ = [[[1., 1., 1.],
#        [1., 1., 1.]],
#       [[1., 1., 1.],
#        [1., 1., 1.]]]

# x = [[[1, 2, 3, 4, 5],
#       [2, 3, 4, 5, 6]]] # 1,2,5

# y = [[[0.8762, 1.1265, 1.3768],
#       [1.3007, 1.3735, 1.4462]]]    # 1,2,3

# y_ = [[[15., 21., 27.],
#        [15., 21., 27.]]]


# w_ = [[[1., 1., 1.]]] 1,1,3
# x = [[[1., 2., 3., 4., 5.]]]  1,1,5
# y = [[[ 6.,  9., 12.]]]   1,1,3