# run_trigger_tiny.py

from narcissus_function import narcissus_gen
import numpy as np
import os

os.makedirs("checkpoint", exist_ok=True)

# Target class (paper uses class 2)
lab = 2

# Trigger generation (Tiny-ImageNet only, paper-faithful)
trigger = narcissus_gen()

# Save trigger
np.save("checkpoint/resnet18_trigger_tinyimagenet.npy", trigger.numpy())

print("Tiny-ImageNet trigger saved (paper-faithful)")
