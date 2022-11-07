import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from loaders.base import Loader
from loaders.user import Users, UserIter
from loaders.item import Items
from utils.logger import Logger


class Cooccurence(Loader, Dataset):
    def __init__(self, scope, args=None):
        super().__init__()
        logger = Logger('COOC', verbose=False, args=args)
        if scope == 'train':
            self.user_data = Users(args)
        else:
            self.user_data = UserIter(scope, args)
        self.item_data = Items(args)

        with open(os.path.join(self.tmp, "user.pkl"), 'rb') as f:
            user_ix = pickle.load(f)
        with open(os.path.join(self.tmp, "article.pkl"), 'rb') as f:
            article_ix = pickle.load(f)

        read_path = os.path.join(self.tmp, 'read.pkl')
        if not os.path.exists(read_path):
            logger.log("processing read directories...")

            reads = {}
            for root, _, files in os.walk(os.path.join(self.res, 'read')):
                for file in files:
                    with open(os.path.join(root, file)) as f:
                        temp = f.readlines()
                        for i in range(len(temp)):
                            temp_split = temp[i].split()
                            try:
                                user = user_ix[temp_split[0]]
                            except:
                                continue
                            
                            items = []
                            for t in temp_split[1:]:
                                try:
                                    items.append(article_ix[t])
                                except:
                                    continue
                            if user in reads:
                                reads[user].extend(items)
                            else:
                                reads[user] = items
            
            with open(read_path, 'wb') as f:
                pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(read_path, 'rb') as f:
            self.read = pickle.load(f)

    def __getitem__(self, idx):
        user_data = self.user_data[idx]
        item_idx = np.random.choice(len(self.item_data))
        item_data = self.item_data[item_idx]
        try:
            read = int(item_idx in self.read[idx])
        except:
            read = 0
        return user_data, idx, item_data, item_idx, read

    def __len__(self):
        return len(self.user_data)


def cooccur_loader(scope, args):
    return DataLoader(
        Cooccurence(scope, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
