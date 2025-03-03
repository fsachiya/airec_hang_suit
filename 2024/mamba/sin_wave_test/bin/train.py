import torch
import ipdb
from einops import rearrange

import sys
sys.path.append("/home/fujita/work/mamba")
from _mamba import Mamba, MambaConfig

batch, length, dim = 1, 64, 1
config = MambaConfig(d_model=dim, 
                     n_layers=1,
                     d_state=16,
                     d_conv=4,
                     expand_factor=2)
model = Mamba(config).to("cuda")

# # モデルの定義
# batch, length, dim = 1, 64, 1
# model = Mamba(
#     d_model=dim,  # モデルの次元
#     d_state=16,   # SSMの状態の拡張係数
#     d_conv=4,     # ローカルな畳み込みの幅
#     expand=2     # ブロックの拡張係数
# ).to("cuda")

# sin波の入力を生成
# x = torch.sin(torch.linspace(0, 2 * 3.14, length, device="cuda")[None, :, None]).repeat(batch, 1, 1)
data = torch.sin(torch.linspace(0, 2 * 3.14, length+1, device="cuda"))
data = rearrange(data, "(b l d) -> b l d", b=1, l=65)
x = data[:,:-1]
y = data[:,1:]

# ipdb.set_trace()
# # 2, 64, 1
# x = rearrange(x, "d l b -> b l d")
# # 1, 64, 2

# ipdb.set_trace()

# 順伝播
y, hs = model(x)
import ipdb; ipdb.set_trace()