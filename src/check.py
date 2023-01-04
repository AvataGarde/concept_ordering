import torch
print(torch.cuda.is_available())
import configparser
import itertools
import os
import numpy as np
from scipy import stats
from tqdm import tqdm
os.chdir(os.path.dirname(__file__))
config = configparser.ConfigParser()
config.read("path.ini")

M_g = np.load(config['commongen']['m1'])["transition"]

print(M_g[0][np.nonzero(M_g[0])])