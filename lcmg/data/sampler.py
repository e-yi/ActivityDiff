import abc
from collections import Counter
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler

from lcmg.runtime_utils.logging_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class MySampler(Sampler, abc.ABC):

    def __init__(self, data_source) -> None:
        super().__init__()


class SizeConstrainedBatchSampler(MySampler):
    """
    A customizable batch sampler for PyTorch that dynamically adjusts batch sizes based on individual data item sizes.
    This class supports various configurations like shuffling, dropping the last batch, perturbing data order using
    noise, and adjusting batch sizes dynamically based on a maximum size constraint.
    说起来是否可以看作对大样本进行了加权，因为对应的batch_size小了
    """

    def __init__(self, data, batch_size, size_fn, shuffle=True, drop_last=False, perturb_sigma=0, seed=42,
                 dynamic_batch_size=True, batch_similar_size=True, strict_upper_limit=False, unpack_data=False):
        """
        use sizes as an input parameter?

        Args:
            data: if unpack_data==True, `sized_obj, *others = data[i]`, else `sized_obj = data[i]`
            batch_size (int): see `dynamic_batch_size`
            size_fn:  a function that will be used as size_fn(data[i]) to get the size of i-th data
            shuffle (bool):
            drop_last (bool): Only useful when `dynamic_batch_size` is false.
            perturb_sigma (float): Only used when `batch_similar_size` is true.
                the sigma of normal noise on the size values, used to perturb the data order set to 0 to disable.
            seed (int):
            dynamic_batch_size (bool): If turned on, the batch_size parameter will be seen as the total maximum size
                of a batch, i.e. `sum([size_fn(i) for i in batch])<=batch_size` (not strict if perturb_sigma!=0),
                the number of samples in a batch will be determined dynamically.
            batch_similar_size: If turned on, each batch will contain objects with similar length
        """

        super().__init__(data)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if perturb_sigma < 0:
            raise ValueError(f"perturb_sigma should be a positive value, or 0 to disable perturbation, "
                             f"but got perturb_sigma={perturb_sigma}")
        if strict_upper_limit:
            # todo
            raise NotImplementedError(f'{strict_upper_limit=}')
        if not dynamic_batch_size:
            raise NotImplementedError(f'{dynamic_batch_size=}')

        drop_last = False if dynamic_batch_size else drop_last

        self.data = data
        self.batch_size = batch_size
        self.size_fn = size_fn

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.perturb_sigma = perturb_sigma
        self.seed = seed
        self.epoch = 0

        if unpack_data:
            self.sizes = [self.size_fn(i) for i, *_ in self.data]
        else:
            self.sizes = [self.size_fn(i) for i in self.data]
        self.unpack_data = unpack_data

        log.info('using dynamic batch sizes')
        # worst case scenario: batch_size 2x, each item size: x+1,
        # so that each item takes a batch, and half of the gpu memory is wasted

        self.ordered_idx2real_idx = sorted(list(range(len(self.sizes))), key=lambda i: self.sizes[i], reverse=False)
        size_ordered_idx = list(range(len(self.sizes)))  # 0 = the smallest object
        if batch_similar_size:
            pass
        else:
            np.random.RandomState(seed).shuffle(size_ordered_idx)

        batch_size_sum = self.batch_size

        splits = []
        large_item_flag = False
        cur_split = []
        cur_n_item = 0
        cur_size_sum = 0
        for i in size_ordered_idx[::-1]:  # starts with large objects if batch_similar_size
            real_idx = self.ordered_idx2real_idx[i]
            size = self.sizes[real_idx]

            if cur_size_sum + size > batch_size_sum:
                if cur_n_item == 0:
                    assert not cur_split
                    assert cur_size_sum == 0
                    large_item_flag = True
                    splits.append([i])
                    continue
                else:
                    splits.append(cur_split)
                    cur_split = []
                    cur_n_item = 0
                    cur_size_sum = 0

            cur_split.append(i)
            cur_n_item += 1
            cur_size_sum += size

        if cur_n_item != 0:
            splits.append(cur_split)

        if large_item_flag:
            log.warning('Large item(s) found in the dataset when using `dynamic_batch_size`. '
                        'Consider increasing `batch_size`. ')

        log.info(f'[(split_size,count),...]: '
                 f'{sorted(list(Counter([len(i) for i in splits]).items()), key=lambda x: x[0])}')

        self.splits = splits
        self.num_batches = len(splits)

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            batch_order = torch.randperm(len(self), generator=g)
        else:
            batch_order = range(len(self))

        sizes = torch.FloatTensor(self.sizes)

        # todo: check if the following line works correctly under ddp
        # noise = torch.randn_like(sizes) * self.perturb_sigma
        # torch random seed is initialized in each dataloader worker process,
        # see torch.utils.data._utils.worker._worker_loop
        #       `seed = base_seed + worker_id`,
        # and torch.utils.data.dataloader._BaseDataLoaderIter
        #        `self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()`,
        # but since we use BatchedSampler, loader.generator is None by default, what will happen then?

        # another approach
        noise = torch.empty_like(sizes).normal_(generator=g) * self.perturb_sigma

        ordered_idx2real_idx_noised = torch.argsort(sizes + noise)

        for bi in batch_order:
            yield [ordered_idx2real_idx_noised[i] for i in self.splits[bi]]

    def set_epoch(self, epoch: int) -> None:
        r"""
        Note: lightning will automatically call this method, but it seems you have to implement this in your sampler,
            The DistributedSamplerWrapper of lightning basically does nothing

        https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def test_LengthBatchSampler():
    class MockData(torch.utils.data.Dataset):
        def __init__(self, data_sizes):
            self.data_sizes = data_sizes

        def __len__(self):
            return len(self.data_sizes)

        def __getitem__(self, index):
            # Return the size as the "data" for simplicity
            return self.data_sizes[index], 'fake', 'data', 'x'

    def mock_size_fn(item):
        # In this mock scenario, the item itself represents its size
        return item

    def print_batches(data, batches):
        for i, batch in enumerate(batches, start=1):
            batch = [data[x] for x in batch]
            sizes = [i for i, *_ in batch]
            print(f"Batch {i}: {batch} (Total Size: {sum(sizes)})")

    def test_dynamic_batch_size(data, max_total_size):
        torch.random.manual_seed(42)
        print(f"\nTesting dynamic batch size with max total size of {max_total_size}:")
        sampler = SizeConstrainedBatchSampler(data, max_total_size, mock_size_fn, shuffle=False, drop_last=False,
                                              dynamic_batch_size=True,
                                              perturb_sigma=0,
                                              batch_similar_size=True)
        batches = [list(batch) for batch in sampler]
        print_batches(data, batches)

    def test_batch_similar_size(data, max_total_size):
        print(f"\nTesting NOT batch similar size with max total size of {max_total_size}:")
        sampler = SizeConstrainedBatchSampler(data, max_total_size, mock_size_fn, shuffle=False, drop_last=False,
                                              dynamic_batch_size=True,
                                              perturb_sigma=2,
                                              batch_similar_size=False)
        batches = [list(batch) for batch in sampler]
        print_batches(data, batches)

    def test_with_perturbation(data, batch_size, perturb_sigma):
        torch.random.manual_seed(42)
        print(f"\nTesting with size perturbation (sigma={perturb_sigma}):")
        sampler = SizeConstrainedBatchSampler(data, batch_size, mock_size_fn, shuffle=False, drop_last=False,
                                              dynamic_batch_size=True,
                                              perturb_sigma=perturb_sigma,
                                              batch_similar_size=True)
        batches = [list(batch) for batch in sampler]
        print_batches(data, batches)

        print(f"\t EPOCH 2:")
        torch.random.manual_seed(42)
        sampler = SizeConstrainedBatchSampler(data, batch_size, mock_size_fn, shuffle=False, drop_last=False,
                                              dynamic_batch_size=True,
                                              perturb_sigma=perturb_sigma,
                                              batch_similar_size=True)
        sampler.set_epoch(2)
        batches = [list(batch) for batch in sampler]
        print_batches(data, batches)

    def test_shuffle_effect(data, batch_size):
        print("\nTesting effect of shuffling:")
        sampler_no_shuffle = SizeConstrainedBatchSampler(data, batch_size, mock_size_fn, shuffle=False, drop_last=False)
        sampler_shuffle = SizeConstrainedBatchSampler(data, batch_size, mock_size_fn, shuffle=True, drop_last=False)
        batches_no_shuffle = [list(batch) for batch in sampler_no_shuffle]
        batches_shuffle = [list(batch) for batch in sampler_shuffle]
        print("Without shuffle:")
        print_batches(data, batches_no_shuffle)
        print("With shuffle:")
        print_batches(data, batches_shuffle)

    data_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dataset = MockData(data_sizes)

    test_dynamic_batch_size(dataset, 8)
    test_batch_similar_size(dataset, 12)
    test_with_perturbation(dataset, 10, 2)
    test_shuffle_effect(dataset, 6)


if __name__ == '__main__':
    test_LengthBatchSampler()
