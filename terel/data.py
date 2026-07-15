from typing import Iterator
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset, Sampler


class SameClassPairDataset(Dataset):
    """Given a base dataset (no transform), returns a *stacked* pair tensor of different samples that share the same class.

    Returns: (pair_tensor, label) where pair_tensor has shape (2, C, H, W)
    """
    def __init__(self, base_dataset: Dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        # build class -> list of indices mapping
        self.class_to_indices = defaultdict(list)
        for idx in range(len(self.base)):
            _, label = self.base[idx]
            self.class_to_indices[int(label)].append(idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img1, label = self.base[idx]
        if self.transform is not None:
            img1 = self.transform(img1)

        return img1, label


class DataReordering:
    @staticmethod
    def _group_indices_by_label(label_list: list[int]):
        labels = np.array(label_list)
        groups = {label: np.where(labels == label)[0].tolist() for label in np.unique(labels)}
        return groups

    @staticmethod
    def batch_coherence_reordering_fn(label_list: list[int], coherence: float, chunk_size: int):
        """Reorder indices such that blocks of `chunk_size` indices share the same label as much as possible.
        """
        groups = DataReordering._group_indices_by_label(label_list)
        indices_reordered = []

        while True:
            labels_sufficient_data = [k for k, g in groups.items() if len(g) > chunk_size]
            labels_select = labels_sufficient_data if len(labels_sufficient_data) > 0 else list(groups.keys())

            if len(labels_select) == 0:  # all groups exhausted
                break

            lens = [len(groups[label]) for label in labels_select]
            probs = [group_len / sum(lens) for group_len in lens]

            group_of_batch = random.choices(labels_select, probs, k=1)[0]
            indices_reordered += list(groups[group_of_batch][:chunk_size])

            groups[group_of_batch] = groups[group_of_batch][chunk_size:]
            if len(groups[group_of_batch]) == 0:
                del groups[group_of_batch]

        # then switch inplace for an according number of times
        for _ in range(int((1 - coherence) * len(indices_reordered))):
            flip_idx1 = random.randint(0, len(indices_reordered) - 1)
            flip_idx2 = random.randint(0, len(indices_reordered) - 1)
            indices_reordered[flip_idx1], indices_reordered[flip_idx2] = (
                indices_reordered[flip_idx2], indices_reordered[flip_idx1]
            )

        return indices_reordered


class CoherentSampler(Sampler):
    """Sampler that yields dataset indices reordered by
    DataReordering.batch_coherence_reordering_fn while optionally shuffling
    the dataset order *without* breaking index<->label correspondence.
    """
    def __init__(self, label_list: list[int], coherence: float, chunk_size: int, shuffle: bool = False):
        self.label_list = list(label_list)
        self.coherence = coherence
        self.chunk_size = chunk_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self.label_list)))
        if self.shuffle:
            random.shuffle(order)

        labels_in_order = [self.label_list[i] for i in order]

        positions_reordered = DataReordering.batch_coherence_reordering_fn(labels_in_order, self.coherence, self.chunk_size)

        indices_reordered = [order[pos] for pos in positions_reordered]
        return iter(indices_reordered)

    def __len__(self) -> int:
        return len(self.label_list)

