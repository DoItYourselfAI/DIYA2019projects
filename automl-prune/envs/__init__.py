import sys
import os
from settings import PROJECT_ROOT

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))

from image_classification import *
from sentiment_analysis import *