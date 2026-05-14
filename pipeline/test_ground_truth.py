import hsi_loader
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# Ensure repository root is on path so local module can be loaded.
repo_root = os.path.dirname(__file__)

file = os.path.join(repo_root, 'dataset')


coco_masks = hsi_loader.HSI2D_loader(file)
