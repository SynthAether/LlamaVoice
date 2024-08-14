# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import json
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from llamavoice.utils.file_utils import read_lists, read_json_lists


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        # force datalist even
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[: self.world_size]
            data = data[self.rank :: self.world_size]
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[: self.num_workers]
        data = data[self.worker_id :: self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(
    data_list_file,
    data_pipeline,
    mode="train",
    shuffle=True,
    partition=True,
    tts_file="",
    prompt_utt2data="",
):
    """Construct dataset from arguments

    We have two shuffle stage in the Dataset. The first is global
    shuffle at shards tar/raw file level. The second is global shuffle
    at training samples level.

    Args:
        data_type(str): raw/shard
        tokenizer (BaseTokenizer): tokenizer to tokenize
        partition(bool): whether to do data partition in terms of rank
    """
    assert mode in ["train", "inference"]
    lists = read_lists(data_list_file)
    if mode == "inference":
        with open(tts_file) as f:
            tts_data = json.load(f)
        utt2lists = read_json_lists(prompt_utt2data)
        # filter unnecessary file in inference mode
        lists = list(
            set([utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists])
        )
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if mode == "inference":
        # map partial arg tts_data in inference mode
        data_pipeline[0] = partial(data_pipeline[0], tts_data=tts_data)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset


def test():
    from llamavoice.dataset.processor import Processor as P
    from llamavoice.model import LlamaVoiceConfig
    from llamavoice.tokenizer.tokenizer import get_tokenizer
    from torch.utils.data import DataLoader
    from llamavoice.utils.mel import extract_linear_features

    C = LlamaVoiceConfig()
    cv_data = train_data = "LibriTTS/data/dev-clean/parquet/data.list"
    get_tokenizer = partial(
        get_tokenizer,
        multilingual=C.dataset.multilingual,
        num_languages=C.dataset.num_languages,
        language=C.dataset.language,
        task=C.dataset.task,
    )
    allowed_special = C.dataset.allowed_special
    tokenize = partial(
        P.tokenize, get_tokenizer=get_tokenizer, allowed_special=allowed_special
    )
    filter = partial(
        P.filter,
        max_length=C.dataset.max_length,
        min_length=C.dataset.min_length,
        token_max_length=C.dataset.token_max_length,
        token_min_length=C.dataset.token_min_length,
    )
    resample = partial(P.resample, resample_rate=C.dataset.sample_rate)
    compute_linear = partial(
        P.compute_linear, feat_extractor=extract_linear_features, cfg=C.dataset
    )
    shuffle = partial(P.shuffle, shuffle_size=C.dataset.shuffle_size)
    sort = partial(P.sort, sort_size=C.dataset.sort_size)
    batch = partial(
        P.batch,
        batch_type=C.dataset.batch_type,
        max_frames_in_batch=C.dataset.max_frames_in_batch,
    )
    padding = partial(P.padding, use_spk_embedding=False)

    data_pipeline = [
        P.parquet_opener,
        tokenize,
        filter,
        compute_linear,
        resample,
        shuffle,
        sort,
        batch,
        padding,
    ]

    train_dataset = Dataset(
        train_data,
        data_pipeline=data_pipeline,
        mode="train",
        shuffle=True,
        partition=True,
    )
    cv_dataset = Dataset(
        cv_data,
        data_pipeline=data_pipeline,
        mode="train",
        shuffle=False,
        partition=False,
    )

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory=C.train.dataloader.pin_memory,
        num_workers=C.train.dataloader.num_worker,
        prefetch_factor=C.dataset.prefetch,
    )
    cv_data_loader = DataLoader(
        cv_dataset,
        batch_size=None,
        pin_memory=C.train.dataloader.pin_memory,
        num_workers=C.train.dataloader.num_worker,
        prefetch_factor=C.dataset.prefetch,
    )
    from accelerate import Accelerator

    accelerator = Accelerator()
    train_data_loader, cv_data_loader = accelerator.prepare(
        train_data_loader, cv_data_loader
    )
    for batch_idx, batch_dict in enumerate(train_data_loader):
        print(batch_idx, [(k, v.shape) for k, v in batch_dict.items()])
        for k, v in batch_dict.items():
            if k.endswith("_len"):
                print(k, v)
    for batch_idx, batch_dict in enumerate(cv_data_loader):
        print(batch_idx, [(k, v.shape) for k, v in batch_dict.items()])


if __name__ == "__main__":
    test()
