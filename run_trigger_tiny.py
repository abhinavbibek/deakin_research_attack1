
from narcissus_function_tiny import narcissus_gen
import numpy as np
import os

os.makedirs("checkpoint", exist_ok=True)
lab = 2
trigger = narcissus_gen()
np.save("checkpoint/resnet18_trigger_tinyimagenet.npy", trigger.cpu().numpy())

