import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from settings import *
from loaders.base import Loader
from utils.logger import Logger
from utils.common import onehot


class Users(Loader, Dataset):
    def __init__(self, args=None):
        super().__init__()
        logger = Logger('USER', verbose=False, args=args)
        user_path = os.path.join(self.tmp, "user.pkl")
        if not os.path.exists(user_path):
            logger.log("converting users to index...")

            user_ix = {}
            for user in self.users["id"]:
                user_ix[user] = len(user_ix)
            with open(user_path, 'wb') as f:
                pickle.dump(user_ix, f, protocol=pickle.HIGHEST_PROTOCOL)

        author_path = os.path.join(self.tmp, "author.pkl")
        if not os.path.exists(author_path):
            logger.log("converting authors to index...")

            author_ix = {}
            for author in self.metadata["user_id"]:
                if author not in author_ix:
                    author_ix[author] = len(author_ix)
            with open(author_path, 'wb') as f:
                pickle.dump(author_ix, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(author_path, 'rb') as f:
            self.author_ix = pickle.load(f)
        
        keyword_path = os.path.join(self.tmp, "keyword.pkl")
        if not os.path.exists(keyword_path):
            logger.log("converting keywords to index...")

            keywords = set()
            for words in self.users["keyword_list"]:
                keywords.update([w['keyword'] for w in words])
            keywords = list(keywords)
            
            keyword_ix = {}
            for keyword in keywords:
                keyword_ix[keyword] = len(keyword_ix)
            
            with open(keyword_path, 'wb') as f:
                pickle.dump(keyword_ix, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(keyword_path, 'rb') as f:
            self.keyword_ix = pickle.load(f)

    def __getitem__(self, idx):
        user = self.users.iloc[idx,:]
        keyword_list = np.asarray([self.keyword_ix[k['keyword']] for k in user[0]])
        if len(keyword_list) != 0:
            keyword_onehot = onehot(keyword_list, len(self.keyword_ix))
            keyword_onehot = np.sum(keyword_onehot, axis=0)
            keyword_cnt = np.asarray([k['cnt'] for k in user[0]])
            keyword_onehot[np.nonzero(keyword_onehot)] = keyword_cnt
        else:
            keyword_onehot = np.zeros(len(self.keyword_ix))

        following_list = np.asarray([self.author_ix[k] for k in user[1]])
        if len(following_list) != 0:
            following_onehot = onehot(following_list, len(self.author_ix))
            following_onehot = np.sum(following_onehot, axis=0)
        else:
            following_onehot = np.zeros(len(self.author_ix))

        keyword_onehot, following_onehot = map(
            lambda x: torch.from_numpy(x).float(),
            [keyword_onehot, following_onehot]
        )
        return keyword_onehot, following_onehot

    def __len__(self):
        return len(self.users)


class UserIter(Dataset):
    def __init__(self, scope, args):
        super().__init__()
        self.user_data = Users(args)
        user_path = os.path.join(PROJECT_ROOT, TMP_DIR, "user.pkl")
        with open(user_path, 'rb') as f:
            self.user_ix = pickle.load(f)

        filename = 'test.users' if scope == 'test' else 'dev.users'
        path = os.path.join(PROJECT_ROOT, RES_DIR, 'predict', filename)
        with open(path) as f:
            users = f.readlines()
        self.user_list = [u.strip('\n') for u in users]

    def __getitem__(self, idx):
        try:
            user = self.user_list[idx]
            user_idx = self.user_ix[user]
            keyword_onehot, following_onehot = self.user_data[user_idx]
        except:
            keyword_onehot, following_onehot = self.user_data[0]
            keyword_onehot, following_onehot = map(
                lambda x: torch.zeros_like(x),
                [keyword_onehot, following_onehot]
            )
        finally:
            return keyword_onehot, following_onehot

    def __len__(self):
        return len(self.user_list)


def user_loader(args):
    return DataLoader(
        Users(args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
