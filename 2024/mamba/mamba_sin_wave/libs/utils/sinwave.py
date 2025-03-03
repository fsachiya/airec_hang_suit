import numpy as np
import matplotlib.pylab as plt


def generate_sinwave(num_cycle, total_step, delta):
    t = np.linspace(0, 2.0 * np.pi * num_cycle, total_step)
    x = np.sin(t + delta)

    return x


train_list = [ generate_sinwave(0.5, 120, 0),
                generate_sinwave(1.5, 120, 0),
                generate_sinwave(2.5, 120, 0)
              ]

test_list = [ generate_sinwave(0.5, 120, 0),
                generate_sinwave(1.0, 120, 0),
                generate_sinwave(1.5, 120, 0),
                generate_sinwave(2.0, 120, 0),
                generate_sinwave(2.5, 120, 0)
              ]



for data in train_list:
    plt.plot(data)
plt.title('train_list')

plt.figure()
for data in test_list:
    plt.plot(data)
plt.title('test_list')
plt.show()
