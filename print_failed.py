from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random


failed = []
if os.path.exists('failed.p'):
    failed = pickle.load(open("failed.p", "rb"))
failed = np.sort(failed)
print('failed count ', len(failed))
print('failed ', failed)

