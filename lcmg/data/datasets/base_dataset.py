from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    @abstractmethod
    def collate_fn(self, *args, **kwargs):
        pass
