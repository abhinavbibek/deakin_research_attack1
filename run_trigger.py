from narcissus_function import narcissus_gen
import numpy as np
import os

os.makedirs("checkpoint", exist_ok=True)

trigger = narcissus_gen(
    dataset_path="/home/dgxuser10/cryptonym/data/",
    lab=2
)

np.save("checkpoint/resnet18_trigger.npy", trigger)
print("Trigger saved.")
