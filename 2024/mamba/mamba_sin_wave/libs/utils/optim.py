import torch
from torch import optim

class DiscreteOptim(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super(DiscreteOptim, self).__init__(params, defaults={'lr': lr})
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
                print(self.state[p])
    # Step Method
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))
                mom = self.state[p]['mom']
                mom = self.momentum * mom - group['lr'] * p.grad.data
                p.data += mom




# class DiscreteOptim(optim.Optimizer):
#     def __init__(self, params, lr = 0.001, momentum = 0.9) -> None:
#         if lr < 0:
#             raise ValueError(f"Invalid learning rate: lr should be >= 0")
#         if momentum < 0:
#             raise ValueError(f"Invalid momentum rate: momentum should be >= 0")
#         defaults = dict(lr = lr, momentum = momentum)
#         super(DiscreteOptim, self).__init__(params, defaults)
#         self.state = dict()
#         for group in self.param_groups:
#             for p in group['params']:
#                 #stateの初期化
#                 self.state[p] = dict(momentum=torch.zeros_like(p.data))
#     def step(self, closure = None) -> None:
#         """
#         parameterのgradはbackwardメソッドで計算済みと考える。
#         更新するパラメーターのt時点での値をW^{t}と表すと、
#         W^{t+1} <- W^{t} - lr * W.grad.data + d_W^{t} * momentum
#         d_W^{t} <- W^{t+1} - W^{t} =  - lr * W.grad.data + d_W^{t} * momentum
#         の式を用いて更新する。
#         """
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p not in self.state:
#                     self.state[p] = dict(momentum=torch.zeros_like(p.data))
#                 sign = torch.sign(p.grad.data)
#                 d_p = - group['lr'] * sign
#                 p.data += d_p
#                 self.state[p]['momentum'] = d_p




# class MomentumSGD(optim.Optimizer):
#     def __init__(self, params, lr = 0.001, momentum = 0.9) -> None:
#         if lr < 0:
#             raise ValueError(f"Invalid learning rate: lr should be >= 0")
#         if momentum < 0:
#             raise ValueError(f"Invalid momentum rate: momentum should be >= 0")
#         defaults = dict(lr = lr, momentum = momentum)
#         super(MomentumSGD, self).__init__(params, defaults)
#         self.state = dict()
#         for group in self.param_groups:
#             for p in group['params']:
#                 #stateの初期化
#                 self.state[p] = dict(momentum=torch.zeros_like(p.data))
#     def step(self, closure = None) -> None:
#         """
#         parameterのgradはbackwardメソッドで計算済みと考える。
#         更新するパラメーターのt時点での値をW^{t}と表すと、
#         W^{t+1} <- W^{t} - lr * W.grad.data + d_W^{t} * momentum
#         d_W^{t} <- W^{t+1} - W^{t} =  - lr * W.grad.data + d_W^{t} * momentum
#         の式を用いて更新する。
#         """
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p not in self.state:
#                     self.state[p] = dict(momentum=torch.zeros_like(p.data))
#                 mom = self.state[p]['momentum']
#                 d_p = - group['lr'] * p.grad.data + group["momentum"] * mom
#                 p.data += d_p
#                 self.state[p]['momentum'] = d_p

