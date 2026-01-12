# # run_trigger_tiny.py  

# from narcissus_function import narcissus_gen
# import numpy as np
# import os

# os.makedirs("checkpoint", exist_ok=True)


# lab = 2

# trigger = narcissus_gen(
#     dataset_path="/home/dgxuser10/cryptonym/data/",
#     lab=lab,
#     target_dataset="tinyimagenet",
#     pood_dataset="caltech256"
# )

# np.save("checkpoint/resnet18_trigger_tinyimagenet.npy", trigger)

# print("Tiny-ImageNet trigger saved")

# run_trigger_tiny.py
from narcissus_tiny import generate_narcissus_trigger
import numpy as np
import os

os.makedirs("checkpoint", exist_ok=True)

trigger = generate_narcissus_trigger(
    data_root="/home/dgxuser10/cryptonym/data",
    target_label=2
)

np.save("checkpoint/narcissus_tiny_trigger.npy", trigger)
print("Trigger saved.")
