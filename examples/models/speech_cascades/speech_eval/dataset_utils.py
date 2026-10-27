# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.distributed as dist
from torch.utils.data import Dataset


def get_global_group():
    if dist.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


class DistributedDataset(Dataset):
    def __init__(self, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))

        self.num_replicas = num_replicas
        self.rank = rank

        # assert len(self.data) >= self.num_replicas, "No need to use distributed evaluation because too few examples!"

        self.data = self.data[self.rank :: self.num_replicas]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError
