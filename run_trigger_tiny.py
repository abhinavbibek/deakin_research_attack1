# run_trigger_tiny.py  (Tiny-ImageNet | PAPER-FAITHFUL)

from narcissus_function import narcissus_gen
import numpy as np
import os

os.makedirs("checkpoint", exist_ok=True)

# Target class (paper example; must match run_attack_tiny.py)
lab = 2

trigger = narcissus_gen(
    dataset_path="/home/dgxuser10/cryptonym/data/",
    lab=lab,
    target_dataset="tinyimagenet",
    pood_dataset="caltech256"
)

np.save("checkpoint/resnet18_trigger_tinyimagenet.npy", trigger)

print("✔ Tiny-ImageNet trigger saved")
