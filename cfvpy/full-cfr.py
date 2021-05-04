
import collections
import logging
import pathlib
import os
import time

import pytorch_lightning.logging as pl_logging
import omegaconf
import torch
import tqdm

import cfvpy.models
import cfvpy.rela
import cfvpy.utils
import heyhi

acc = 0
for i in range(216):
  acc += cfvpy.rela.compute_full_game_cfr(i, 8192)

print("Final exploitability: ", acc/216)
