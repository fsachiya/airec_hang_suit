import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat('../data/HangApron/1.mat')
keys = data.keys()
print(keys)

left_arm_joint = data["left_arm_joint"]     # (640, 7)
right_arm_joint = data["right_arm_joint"]   # (640, 7)
left_hand_joint = data["left_hand_joint"]   # (640, 14)
right_hand_joint = data["right_hand_joint"] # (640, 7)
left_img = data["left_img"]                 # (640, 480, 640, 3)
right_img = data["right_img"]               # (640, 480, 640, 3)
left_hand_state = data["left_hand_state"]   # (640, 16)
right_hand_state = data["right_hand_state"] # (640, 16)

plt.plot(left_hand_state)
plt.savefig("./fig/left_hand_state_1.png", format="png", dpi=300)

import ipdb; ipdb.set_trace()
