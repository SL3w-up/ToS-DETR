# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from codes in torch.utils.data.distributed
# ------------------------------------------------------------------------

import os
import math
import random
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from util.fracture_ops import SAMPLE_TYPE_ORDER


def _distributed_epoch_indices(dataset_len, num_replicas, rank, epoch, shuffle=True):
    if dataset_len <= 0:
        return []
    num_replicas = max(int(num_replicas), 1)
    rank = int(rank)
    if num_replicas == 1:
        return list(range(dataset_len))

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(epoch))
        indices = torch.randperm(dataset_len, generator=generator).tolist()
    else:
        indices = list(range(dataset_len))

    num_samples = int(math.ceil(dataset_len / num_replicas))
    total_size = num_samples * num_replicas
    padding_size = total_size - len(indices)
    if padding_size <= len(indices):
        indices += indices[:padding_size]
    else:
        indices += (indices * int(math.ceil(padding_size / len(indices))))[:padding_size]
    return indices[rank:total_size:num_replicas]


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SampleTypeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, ratios, drop_last=True, num_replicas=1, rank=0, base_sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ratios = ratios
        self.drop_last = drop_last
        self.num_replicas = max(num_replicas, 1)
        self.rank = rank
        self.epoch = 0
        self.base_sampler = base_sampler
        self._refresh_indices()
        self.num_batches = len(self.local_indices) // self.batch_size if drop_last else int(math.ceil(len(self.local_indices) / self.batch_size))

    def _refresh_indices(self):
        if self.base_sampler is None:
            self.local_indices = _distributed_epoch_indices(
                len(self.dataset),
                self.num_replicas,
                self.rank,
                self.epoch,
                shuffle=True,
            )
        else:
            if hasattr(self.base_sampler, "set_epoch"):
                self.base_sampler.set_epoch(self.epoch)
            local_indices = list(iter(self.base_sampler))
            if len(local_indices) <= 0:
                self.local_indices = []
            else:
                self.local_indices = local_indices
        self.type_to_indices = {sample_type: [] for sample_type in SAMPLE_TYPE_ORDER}
        self.fallback_indices = []
        for index in self.local_indices:
            sample_type = self.dataset.get_sample_type(index) if hasattr(self.dataset, "get_sample_type") else None
            if sample_type in self.type_to_indices:
                self.type_to_indices[sample_type].append(index)
            else:
                self.fallback_indices.append(index)
        if not self.fallback_indices:
            self.fallback_indices = list(self.local_indices)

    def _epoch_type_sequence(self, generator):
        total = self.num_batches * self.batch_size
        raw_counts = [ratio * total for ratio in self.ratios]
        counts = [int(math.floor(value)) for value in raw_counts]
        remainder = total - sum(counts)
        fractions = sorted(
            enumerate([value - math.floor(value) for value in raw_counts]),
            key=lambda item: item[1],
            reverse=True,
        )
        for idx, _ in fractions[:remainder]:
            counts[idx] += 1
        sequence = []
        for sample_type, count in zip(SAMPLE_TYPE_ORDER, counts):
            sequence.extend([sample_type] * count)
        generator.shuffle(sequence)
        return sequence

    def __iter__(self):
        self._refresh_indices()
        generator = random.Random(self.epoch)
        shuffled = {}
        for sample_type, indices in self.type_to_indices.items():
            local = list(indices)
            generator.shuffle(local)
            shuffled[sample_type] = local
        fallback = list(self.fallback_indices)
        generator.shuffle(fallback)
        cursors = {sample_type: 0 for sample_type in SAMPLE_TYPE_ORDER}
        fallback_cursor = 0
        sample_type_sequence = self._epoch_type_sequence(generator)

        for batch_idx in range(self.num_batches):
            batch = []
            used = set()
            for sample_type in sample_type_sequence[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size
            ]:
                pool = shuffled[sample_type]
                chosen = None
                if pool:
                    for _ in range(len(pool)):
                        index = pool[cursors[sample_type] % len(pool)]
                        cursors[sample_type] += 1
                        if index not in used:
                            chosen = index
                            break
                if chosen is not None:
                    batch.append(chosen)
                    used.add(chosen)
            while len(batch) < self.batch_size and fallback:
                index = fallback[fallback_cursor % len(fallback)]
                fallback_cursor += 1
                if index in used:
                    if len(used) >= len(self.local_indices):
                        break
                    continue
                batch.append(index)
                used.add(index)
            generator.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_ratios(self, ratios):
        self.ratios = ratios


class CaseAwareSampleTypeBatchSampler(Sampler):
    """Batch sampler that preserves P/PN/HN/N ratios while grouping nearby same-case slices."""

    def __init__(self, dataset, batch_size, ratios, neighbor_window=5, same_case_fraction=0.5,
                 drop_last=True, num_replicas=1, rank=0, base_sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ratios = ratios
        self.neighbor_window = neighbor_window
        self.same_case_fraction = same_case_fraction
        self.drop_last = drop_last
        self.num_replicas = max(num_replicas, 1)
        self.rank = rank
        self.epoch = 0
        self.base_sampler = base_sampler
        self._refresh_indices()
        self.num_batches = len(self.local_indices) // self.batch_size if drop_last else int(math.ceil(len(self.local_indices) / self.batch_size))

    def _refresh_indices(self):
        if self.base_sampler is None:
            self.local_indices = _distributed_epoch_indices(
                len(self.dataset),
                self.num_replicas,
                self.rank,
                self.epoch,
                shuffle=True,
            )
        else:
            if hasattr(self.base_sampler, "set_epoch"):
                self.base_sampler.set_epoch(self.epoch)
            local_indices = list(iter(self.base_sampler))
            if len(local_indices) <= 0:
                self.local_indices = []
            else:
                self.local_indices = local_indices
        self.type_to_indices = {sample_type: [] for sample_type in SAMPLE_TYPE_ORDER}
        self.case_to_indices = {}
        for index in self.local_indices:
            sample_type = self.dataset.get_sample_type(index) if hasattr(self.dataset, "get_sample_type") else None
            if sample_type in self.type_to_indices:
                self.type_to_indices[sample_type].append(index)
            case_id = self.dataset.get_case_id(index) if hasattr(self.dataset, "get_case_id") else str(index)
            self.case_to_indices.setdefault(case_id, []).append(index)
        self.fallback_indices = list(self.local_indices)

    def _epoch_type_sequence(self, generator):
        total = self.num_batches * self.batch_size
        raw_counts = [ratio * total for ratio in self.ratios]
        counts = [int(math.floor(value)) for value in raw_counts]
        remainder = total - sum(counts)
        fractions = sorted(
            enumerate([value - math.floor(value) for value in raw_counts]),
            key=lambda item: item[1],
            reverse=True,
        )
        for idx, _ in fractions[:remainder]:
            counts[idx] += 1
        sequence = []
        for sample_type, count in zip(SAMPLE_TYPE_ORDER, counts):
            sequence.extend([sample_type] * count)
        generator.shuffle(sequence)
        return sequence

    def _add_index(self, batch, used, index, counts, enforce_quota=False):
        if index in used or len(batch) >= self.batch_size:
            return False
        sample_type = self.dataset.get_sample_type(index) if hasattr(self.dataset, "get_sample_type") else None
        if enforce_quota and sample_type in counts and counts[sample_type] <= 0:
            return False
        batch.append(index)
        used.add(index)
        if sample_type in counts and counts[sample_type] > 0:
            counts[sample_type] -= 1
        return True

    def __iter__(self):
        self._refresh_indices()
        generator = random.Random(self.epoch)
        shuffled = {}
        for sample_type, indices in self.type_to_indices.items():
            local = list(indices)
            generator.shuffle(local)
            shuffled[sample_type] = local
        cursors = {sample_type: 0 for sample_type in SAMPLE_TYPE_ORDER}
        fallback = list(self.fallback_indices)
        generator.shuffle(fallback)
        fallback_cursor = 0
        sample_type_sequence = self._epoch_type_sequence(generator)

        for batch_idx in range(self.num_batches):
            target_types = sample_type_sequence[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size
            ]
            counts = {sample_type: target_types.count(sample_type) for sample_type in SAMPLE_TYPE_ORDER}
            batch = []
            used = set()
            anchor = None
            anchor_types = list(target_types)
            generator.shuffle(anchor_types)
            for sample_type in anchor_types:
                pool = shuffled.get(sample_type, [])
                for _ in range(len(pool)):
                    candidate = pool[cursors[sample_type] % len(pool)]
                    cursors[sample_type] += 1
                    if candidate not in used:
                        anchor = candidate
                        break
                if anchor is not None:
                    break
            if anchor is None and fallback:
                for _ in range(len(fallback)):
                    candidate = fallback[fallback_cursor % len(fallback)]
                    fallback_cursor += 1
                    if candidate not in used:
                        anchor = candidate
                        break
            if anchor is None:
                continue
            self._add_index(batch, used, anchor, counts)

            same_case_budget = max(0, int(round(self.batch_size * self.same_case_fraction)) - len(batch))
            if hasattr(self.dataset, "get_case_neighbor_indices") and same_case_budget > 0:
                neighbors = self.dataset.get_case_neighbor_indices(
                    anchor, window=self.neighbor_window, include_self=False
                )
                neighbors = [idx for idx in neighbors if idx in self.local_indices]
                anchor_slice = self.dataset.get_slice_index(anchor) if hasattr(self.dataset, "get_slice_index") else anchor
                neighbors.sort(key=lambda idx: abs(self.dataset.get_slice_index(idx) - anchor_slice)
                               if hasattr(self.dataset, "get_slice_index") else abs(idx - anchor))
                preferred = []
                for sample_type in ("PN", "P", "HN", "N"):
                    preferred.extend([idx for idx in neighbors
                                      if self.dataset.get_sample_type(idx) == sample_type and idx not in preferred])
                for neighbor in preferred:
                    if same_case_budget <= 0:
                        break
                    if self._add_index(batch, used, neighbor, counts, enforce_quota=True):
                        same_case_budget -= 1

            for sample_type in SAMPLE_TYPE_ORDER:
                while counts.get(sample_type, 0) > 0 and len(batch) < self.batch_size:
                    added = False
                    pool = shuffled.get(sample_type, [])
                    if pool:
                        for _ in range(len(pool)):
                            index = pool[cursors[sample_type] % len(pool)]
                            cursors[sample_type] += 1
                            if self._add_index(batch, used, index, counts):
                                added = True
                                break
                    if added:
                        continue
                    counts[sample_type] = 0
                    break

            while len(batch) < self.batch_size and fallback:
                index = fallback[fallback_cursor % len(fallback)]
                fallback_cursor += 1
                if index in used:
                    if len(used) >= len(self.local_indices):
                        break
                    continue
                if self._add_index(batch, used, index, counts):
                    continue
                if len(used) >= len(self.local_indices):
                    break

            generator.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_ratios(self, ratios):
        self.ratios = ratios


class CaseSequentialBatchSampler(Sampler):
    """Validation/test sampler that keeps each mini-batch inside a single case.

    PN/HN/N/P are training-only sampling tags. Evaluation should follow the
    clinical unit instead: sorted slices from one case, split into fixed-size
    chunks only for memory control.
    """

    def __init__(self, dataset, batch_size, drop_last=False, num_replicas=1, rank=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.local_batches = self._build_batches()

    def _case_id(self, index):
        if hasattr(self.dataset, "get_case_id"):
            return str(self.dataset.get_case_id(index))
        return str(index)

    def _slice_index(self, index):
        if hasattr(self.dataset, "get_slice_index"):
            return int(self.dataset.get_slice_index(index))
        return int(index)

    def _build_batches(self):
        case_to_indices = {}
        for index in range(len(self.dataset)):
            case_to_indices.setdefault(self._case_id(index), []).append(index)

        batches = []
        for case_id in sorted(case_to_indices):
            indices = sorted(case_to_indices[case_id], key=self._slice_index)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.num_replicas <= 1 or not batches:
            return batches
        num_batches = int(math.ceil(len(batches) / self.num_replicas))
        total_size = num_batches * self.num_replicas
        padding_size = total_size - len(batches)
        if padding_size > 0:
            batches = batches + (batches * int(math.ceil(padding_size / len(batches))))[:padding_size]
        return batches[self.rank:total_size:self.num_replicas]

    def __iter__(self):
        return iter([list(batch) for batch in self.local_batches])

    def __len__(self):
        return len(self.local_batches)

    def set_epoch(self, epoch):
        return None


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[self.rank // self.num_parts:self.total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
