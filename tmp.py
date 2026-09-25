import sys
import gym
import numpy
import torch
import torchvision

print("Python:", sys.version)
print("gym:", gym.__version__)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)

try:
    import gym_super_mario_bros
    print("gym_super_mario_bros:", getattr(gym_super_mario_bros, "__version__", "version not available"))
except Exception as e:
    print("gym_super_mario_bros error:", e)

try:
    import nes_py
    print("nes_py:", getattr(nes_py, "__version__", "version not available"))
except Exception as e:
    print("nes_py error:", e)