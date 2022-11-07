import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from loaders.base import Loader
from utils.logger import Logger
from utils.common import onehot


class Items(Loader, Dataset):
    def __init__(self, args=None):
        super().__init__()
        logger = Logger('ITEM', verbose=False, args=args)
        author_path = os.path.join(self.tmp, "author.pkl")
        article_path = os.path.join(self.tmp, "article.pkl")
        if not os.path.exists(article_path):
            logger.log("converting articles to index...")

            article_ix = {}
            for article in self.metadata["id"]:
                article_ix[article] = len(article_ix)
            with open(article_path, 'wb') as f:
                pickle.dump(article_ix, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        
        magazine_path = os.path.join(self.tmp, "magazine.pkl")
        if not os.path.exists(magazine_path):
            logger.log("converting magazines to index...")

            magazines = set()
            for magazine in self.metadata["magazine_id"]:
                magazines.add(magazine)
            magazines = list(magazines)

            magazine_ix = {}
            for magazine in magazines:
                magazine_ix[magazine] = len(magazine_ix)
            
            with open(magazine_path, 'wb') as f:
                pickle.dump(magazine_ix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(magazine_path, 'rb') as f:
            self.magazine_ix = pickle.load(f)
        
        tag_path = os.path.join(self.tmp, "tag.pkl")
        if not os.path.exists(tag_path):
            logger.log("converting tags to index...")

            tags = set()
            for tag in self.metadata["keyword_list"]:
                tags.update(tag)
            tags = list(tags)

            tag_ix = {}
            for tag in tags:
                tag_ix[tag] = len(tag_ix)
            
            with open(tag_path, 'wb') as f:
                pickle.dump(tag_ix, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(tag_path, 'rb') as f:
            self.tag_ix = pickle.load(f)

    def __getitem__(self, idx):
        item = self.metadata.iloc[idx,:]
        author_id = np.array([self.author_ix[item["user_id"]]])
        author_onehot = onehot(author_id, len(self.author_ix))
        author_onehot = author_onehot.squeeze()
        
        magazine_id = np.array([self.magazine_ix[item["magazine_id"]]])
        magazine_onehot = onehot(magazine_id, len(self.magazine_ix))
        magazine_onehot = magazine_onehot.squeeze()
        
        tag_list = np.asarray([self.tag_ix[t] for t in item["keyword_list"]])
        if len(tag_list) != 0:
            tag_onehot = onehot(tag_list, len(self.tag_ix))
            tag_onehot = np.sum(tag_onehot, axis=0)
        else:
            tag_onehot = np.zeros(len(self.tag_ix))

        author_onehot, magazine_onehot, tag_onehot = map(
            lambda x: torch.from_numpy(x).float(),
            [author_onehot, magazine_onehot, tag_onehot]
        )
        return author_onehot, magazine_onehot, tag_onehot

    def __len__(self):
        return len(self.metadata)


def item_loader(args):
    return DataLoader(
        Items(args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
