import os
import pandas as pd

from settings import *


class Loader:
    def __init__(self):
        self.res = os.path.join(PROJECT_ROOT, RES_DIR)
        self.tmp = os.path.join(PROJECT_ROOT, TMP_DIR)

        path = os.path.join(self.res, 'metadata.json')
        self.metadata = pd.read_json(path, lines=True)

        path = os.path.join(self.res, 'users.json')
        self.users = pd.read_json(path, lines=True)
