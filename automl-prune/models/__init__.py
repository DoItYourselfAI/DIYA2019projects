import sys
import os
from settings import PROJECT_ROOT

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_ROOT, 'models'))

from mlp import *
from dense import *