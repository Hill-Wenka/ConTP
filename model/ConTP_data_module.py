import random
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import ProteinDataset
from utils.dataset.Tokenizer import ProteinTokenizer


def sample_positive(idx, idx2label, label2idx, seed=42, allow_self=True, verbose=False):
    """
    从同标签样本中采一个正样本。
    - 若该标签下只有 anchor 自己：
        * 若 allow_self=True，则返回自己；
        * 否则返回 None。
    """
    random.seed(seed)

    # --- 支持多标签情况 ---
    if isinstance(idx2label[idx], list):
        anchor_label = random.choice(idx2label[idx])
    else:
        anchor_label = idx2label[idx]

    # --- 获取同类样本 ---
    exclude_samples = list(set(label2idx.get(anchor_label, [])) - {idx})

    # --- 容错机制 ---
    if not exclude_samples:
        if verbose:
            print(f"[Warning] Label '{anchor_label}' has only one sample (idx={idx})")
        if allow_self:
            return idx
        else:
            return None

    # --- 正常采样 ---
    return random.choice(exclude_samples)


def sample_negative(idx, idx2label, label2idx, seed=42):
    random.seed(seed)
    unique_labels = list(label2idx.keys())
    if isinstance(idx2label[idx], list):
        anchor_labels = set(idx2label[idx])
    else:
        anchor_labels = [idx2label[idx]]
    exclude_labels = list(set(unique_labels) - set(anchor_labels))
    neg_label = random.choice(exclude_labels)
    neg = random.choice(label2idx[neg_label])
    # print('neg_label', neg_label, 'neg', neg)
    return neg


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_key='substrate_ids', n_pos=6, n_neg=30, seed=42):
        self.dataset = dataset
        self.label_key = label_key
        self.idx2label = {}  # map sample id to label
        for i, row in self.dataset.metadata.iterrows():
            self.idx2label[i] = eval(row[label_key]) if label_key == 'substrate_ids' else row[label_key]

        self.label2idx = {}  # map label to sample id
        for i, row in self.dataset.metadata.iterrows():
            for label in eval(row[label_key]) if label_key == 'substrate_ids' else [row[label_key]]:
                if label not in self.label2idx:
                    self.label2idx[label] = []
                self.label2idx[label].append(i)

        self.unique_labels = sorted(list(self.label2idx.keys()))
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.seed = seed

    def __len__(self):
        if self.label_key == 'substrate_ids':
            times = 30
        else:
            times = 2
        return len(self.unique_labels) * times

    def __getitem__(self, idx):
        idx = idx % len(self.unique_labels)
        anchor_label = self.unique_labels[idx]
        anchor_idx = random.choice(self.label2idx[anchor_label])
        anchor_embedding = self.dataset[anchor_idx]['esm_embedding']
        data = [anchor_embedding]
        label = [self.idx2label[anchor_idx]]
        # mask = torch.tensor([1] * (self.n_pos + 1) + [0] * self.n_neg)
        pos_seeds = random.sample(range(1000000000), self.n_pos)
        for i in range(self.n_pos):
            pos = sample_positive(anchor_idx, self.idx2label, self.label2idx, pos_seeds[i])
            pos_embedding = self.dataset[pos]['esm_embedding']
            data.append(pos_embedding)
            label.append(self.idx2label[pos])
        neg_seeds = random.sample(range(1000000000), self.n_neg)
        for j in range(self.n_neg):
            neg = sample_negative(anchor_idx, self.idx2label, self.label2idx, neg_seeds[j])
            neg_embedding = self.dataset[neg]['esm_embedding']
            data.append(neg_embedding)
            label.append(self.idx2label[neg])
        data = torch.from_numpy(np.array(data))
        return data, label


class ConTPDataModule(L.LightningDataModule):
    def __init__(self, config, log=True):
        super().__init__()

        self.log = log
        self.config = config
        self.dataset_args = self.config.dataset
        self.tokenization_args = self.config.tokenization
        self.train_dataloader_args = self.config.train_dataloader
        self.valid_dataloader_args = self.config.valid_dataloader
        self.test_dataloader_args = self.config.test_dataloader
        self.predict_dataloader_args = self.config.predict_dataloader

        dataset_args = self.dataset_args.dataset
        self.dataset = dataset_args if isinstance(dataset_args, ProteinDataset) else ProteinDataset(**dataset_args)
        self.tokenizer = ProteinTokenizer(**self.tokenization_args)
        self.select_indices = None
        self.dataframe = self.dataset.metadata

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.max_len = self.dataset_args.max_len if self.dataset_args.max_len is not None else self.dataset.metadata.length.max()
        self.min_len = self.dataset_args.min_len if self.dataset_args.min_len is not None else self.dataset.metadata.length.min()
        self.state = False  # whether the data_analysis is prepared

    def prepare_data(self):
        if not self.state:  # run only once to avoid repeated preparation and split changes
            # each run gives different select_indices. Thus, ensure only one run to avoid different results
            # select data_analysis with sequence min_len <= length <= max_len
            select_indices = self.dataset.metadata[
                (self.dataset.metadata['length'] <= self.max_len) &
                (self.dataset.metadata['length'] >= self.min_len)].index

            # select a subset of the dataset for debug
            subset_ratio = self.dataset_args.mini_set_ratio if self.dataset_args.mini_set_ratio is not None else 1
            self.select_indices = pd.DataFrame(select_indices).sample(frac=subset_ratio,
                                                                      random_state=self.config.seed)[0].values
            self.dataset.metadata = self.dataset.metadata.loc[self.select_indices]
            self.dataframe = self.dataset.metadata

            # split the dataset into train, valid, test
            if self.dataset_args.split is not None:
                self.dataset.split(**self.dataset_args.split)
                if self.log:
                    print('split the dataset into train, valid, test')
            else:
                if 'substrate' in self.dataset.path:
                    split_df = pd.read_csv('./temp/substrate_classification_split.csv')
                    self.split_df = split_df
                    split_df = split_df.sort_values(by=['id', 'sequence', 'label', 'substrate'])
                    self.label_key = 'substrate_ids'
                    df = self.dataframe
                    df = df[df['length'] >= 200]
                    df = df.sort_values(by=['id', 'sequence', 'label', 'substrate'])
                    self.dataframe = df
                    self.dataframe['split'] = split_df['split'].values
                else:
                    split_df = pd.read_csv('./temp/tc_classification_split.csv')
                    split_df = split_df.sort_values(by=['id', 'sequence', 'label', 'substrate'])
                    self.label_key = 'label_id'
                    df = self.dataframe
                    df = df[df['length'] >= 200]
                    label_counts = df['label_id'].value_counts()
                    df = df[df['label_id'].isin(label_counts[label_counts >= 10].index)]
                    df = df.sort_values(by=['id', 'sequence', 'label', 'substrate'])
                    self.dataframe = df
                    self.dataframe['split'] = split_df['split'].values

                self.dataset.metadata = self.dataframe
                if self.config.select_labels is not None:
                    self.dataframe = self.dataframe[self.dataframe['label_id'].isin(self.config.select_labels)]
                    self.dataset.metadata = self.dataframe
                if self.log:
                    print('use the original split of the dataset')

            self.train_index = self.dataframe[self.dataframe['split'] == 'train'].index.tolist()
            self.valid_index = self.dataframe[self.dataframe['split'] == 'valid'].index.tolist()
            self.test_index = self.dataframe[self.dataframe['split'] == 'test'].index.tolist()
            self.test_index = self.test_index if len(self.test_index) > 0 else self.train_index[:len(
                self.dataset) // 10]  # use 10% of the training data_analysis as test data_analysis by default
            self.valid_index = self.valid_index if len(
                self.valid_index) > 0 else self.test_index  # use test data_analysis as valid data_analysis by default
            self.predict_index = None
            self.state = True

            if self.log:
                print(
                    f'[prepare_data] max_len: {self.max_len}, subset_ratio: {subset_ratio}, number: {len(self.dataframe)}')

    def setup(self, stage=None):
        if self.log:
            print('=' * 30, f'Setup [{stage}] Start', '=' * 30)
        if stage == 'fit':
            self.train_dataset = self.dataset.construct_subset(self.train_index, 'train_dataset')
            self.contrastive_dataset = ContrastiveDataset(self.train_dataset,
                                                          label_key=self.label_key,
                                                          n_pos=self.dataset_args.n_pos,
                                                          n_neg=self.dataset_args.n_neg,
                                                          seed=self.config.seed)
            self.valid_dataset = self.dataset.construct_subset(self.valid_index, 'valid_dataset')
            if self.log:
                print('[self.train_dataset]', len(self.train_dataset))
                print('[self.val_dataset]', len(self.valid_dataset))
        elif stage == 'validate':
            self.valid_dataset = self.dataset.construct_subset(self.valid_index, 'valid_dataset')
            if self.log:
                print('[self.val_dataset]', len(self.valid_dataset))
        elif stage == 'test':
            self.test_dataset = self.dataset.construct_subset(self.test_index, 'test_dataset')
            if self.log:
                print('[self.test_dataset]', len(self.test_dataset))
        elif stage == 'predict':
            if self.predict_dataset is None:
                self.predict_index = self.test_index if self.predict_index is None else self.predict_index
                self.predict_dataset = self.dataset.construct_subset(self.predict_index, 'predict_dataset')
            else:
                pass  # dataset has been provided when calling self.prepare_predict_data()
            if self.log:
                print('[self.predict_dataset]', len(self.predict_dataset))
        else:
            print('stage', stage)
            raise RuntimeError(f'Parameter {stage} is None or illegal, please set it properly')
        if self.log:
            print('=' * 30, f'Setup [{stage}] End', '=' * 30)

    def train_dataloader(self):
        return DataLoader(self.contrastive_dataset, collate_fn=self.contrastive_collate_fn,
                          **self.train_dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, collate_fn=self.collate_fn, **self.valid_dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.collate_fn, **self.test_dataloader_args)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, collate_fn=self.collate_fn, **self.predict_dataloader_args)

    def contrastive_collate_fn(self, batch):
        data, label = [], []
        for item in batch:
            data.append(item[0])
            label.append(item[1])
        data = torch.stack(data)
        return data, label

    def collate_fn(self, batch):
        data, label = [], []
        for item in batch:
            data.append(item['esm_embedding'])
            l = eval(item[self.label_key]) if self.label_key == 'substrate_ids' else item[self.label_key]
            label.append(l)
        data = torch.from_numpy(np.array(data))
        return data, label

    def prepare_predict_data(self, sequence=None, dataset=None, subset=None, **kwargs):
        if sum([sequence is not None, dataset is not None, subset is not None]) != 1:
            raise ValueError('Please set one of the parameters: sequences, dataset, subset')

        if subset is not None:
            if not self.state:
                self.prepare_data()
            if subset == 'train':
                self.predict_index = self.train_index
            elif subset == 'valid':
                self.predict_index = self.valid_index
            elif subset == 'test':
                self.predict_index = self.test_index
            elif subset == 'train_valid':
                self.predict_index = self.train_index + self.valid_index
            elif subset == 'train_valid_test':
                self.predict_index = self.train_index + self.valid_index + self.test_index
            else:
                raise ValueError(f'subset: {subset} is not valid')
            self.predict_dataset = self.dataset.construct_subset(self.predict_index, 'predict_dataset')

        if sequence is not None:
            self.predict_dataset = ProteinDataset('predict_dataset', sequence=sequence, **kwargs)

        if dataset is not None:
            self.predict_dataset = dataset
            self.seq2esm = kwargs.get('seq2esm', None)
            if self.seq2esm is not None:
                for key in self.seq2esm:
                    if type(self.seq2esm[key]) == np.ndarray:
                        self.seq2esm[key] = torch.from_numpy(self.seq2esm[key])
                    elif type(self.seq2esm[key]) == torch.Tensor:
                        pass
                    else:
                        raise ValueError('Unsupported type')
        if self.log:
            print('[prepare custom predict dataset]', len(self.predict_dataset))
