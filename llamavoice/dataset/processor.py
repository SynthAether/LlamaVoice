# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

AUDIO_FORMAT_SETS = set(["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"])


def parquet_opener(data, mode="train", tts_data={}):
    """Give url or local file, return file descriptor
    Inplace operation.

    Args:
        data(Iterable[str]): url or local file list

    Returns:
        Iterable[{src, stream}]
    """
    for sample in data:
        assert "src" in sample
        url = sample["src"]
        try:
            df = pq.read_table(url).to_pandas()
            for i in range(len(df)):
                if mode == "inference" and df.loc[i, "utt"] not in tts_data:
                    continue
                sample.update(dict(df.loc[i]))
                if mode == "train":
                    # NOTE do not return sample directly, must initialize a new dict
                    yield {**sample}
                else:
                    for index, text in enumerate(tts_data[df.loc[i, "utt"]]):
                        yield {**sample, "tts_index": index, "tts_text": text}
        except Exception as ex:
            logging.warning("Failed to open {}, ex info {}".format(url, ex))


def filter(
    data,
    max_length=10240,
    min_length=10,
    token_max_length=200,
    token_min_length=1,
    min_output_input_ratio=0.0005,
    max_output_input_ratio=1,
    mode="train",
):
    """Filter sample according to feature and label length
    Inplace operation.

    Args::
        data: Iterable[{key, wav, label, sample_rate}]
        max_length: drop utterance which is greater than max_length(10ms)
        min_length: drop utterance which is less than min_length(10ms)
        token_max_length: drop utterance which is greater than
            token_max_length, especially when use char unit for
            english modeling
        token_min_length: drop utterance which is
            less than token_max_length
        min_output_input_ratio: minimal ration of
            token_length / feats_length(10ms)
        max_output_input_ratio: maximum ration of
            token_length / feats_length(10ms)

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        sample["speech"], sample["sample_rate"] = torchaudio.load(
            BytesIO(sample["audio_data"])
        )
        del sample["audio_data"]
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample["speech"].size(1) / sample["sample_rate"] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample["text_token"]) < token_min_length:
            continue
        if len(sample["text_token"]) > token_max_length:
            continue
        if "speech_token" in sample and len(sample["speech_token"]) == 0:
            continue
        if num_frames != 0:
            if len(sample["text_token"]) / num_frames < min_output_input_ratio:
                continue
            if len(sample["text_token"]) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode="train"):
    """Resample data.
    Inplace operation.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        resample_rate: target resample rate

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert "sample_rate" in sample
        assert "speech" in sample
        sample_rate = sample["sample_rate"]
        waveform = sample["speech"]
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample["sample_rate"] = resample_rate
            sample["speech"] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate
            )(waveform)
        max_val = sample["speech"].abs().max()
        if max_val > 1:
            sample["speech"] /= max_val
        yield sample


def compute_fbank(data, feat_extractor, mode="train"):
    """Extract fbank

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        assert "sample_rate" in sample
        assert "speech" in sample
        assert "utt" in sample
        assert "text_token" in sample
        waveform = sample["speech"]
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample["speech_feat"] = mat
        del sample["speech"]
        yield sample


def compute_linear(data, feat_extractor, cfg, mode="train"):
    """Extract linear features"""
    for sample in data:
        y = sample["speech"]
        linear = feat_extractor(y, cfg)
        sample["speech_feat"] = linear
        yield sample


def compute_mel(data, feat_extractor, cfg, mode="train"):
    for sample in data:
        y = sample["speech"]
        mel = feat_extractor(y, cfg)
        sample["mel"] = mel
        yield sample


def parse_embedding(data, normalize, mode="train"):
    """Parse utt_embedding/spk_embedding

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        sample["utt_embedding"] = torch.tensor(
            sample["utt_embedding"], dtype=torch.float32
        )
        sample["spk_embedding"] = torch.tensor(
            sample["spk_embedding"], dtype=torch.float32
        )
        if normalize:
            sample["utt_embedding"] = F.normalize(sample["utt_embedding"], dim=0)
            sample["spk_embedding"] = F.normalize(sample["spk_embedding"], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode="train"):
    """Decode text to chars or BPE
    Inplace operation

    Args:
        data: Iterable[{key, wav, txt, sample_rate}]

    Returns:
        Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert "text" in sample
        tks = tokenizer.encode(sample["text"], allowed_special=allowed_special)
        sample["text_token"] = [t + 1 for t in tks]  # 0 is padding
        if mode == "inference":
            tks = tokenizer.encode(sample["tts_text"], allowed_special=allowed_special)
            sample["tts_text_token"] = [t + 1 for t in tks]  # 0 is padding
        yield sample


def shuffle(data, shuffle_size=10000, mode="train"):
    """Local shuffle the data

    Args:
        data: Iterable[{key, feat, label}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode="train"):
    """Sort the data by feature length.
    Sort is used after shuffle and before batch, so we can group
    utts with similar lengths into a batch, and `sort_size` should
    be less than `shuffle_size`

    Args:
        data: Iterable[{key, feat, label}]
        sort_size: buffer size for sort

    Returns:
        Iterable[{key, feat, label}]
    """

    def key_func(x):
        if "speech_feat" in x:
            return x["speech_feat"].size(0)
        elif "speech" in x:
            return x["speech"].size(-1)
        else:
            raise Exception("key_func error")

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=key_func)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=key_func)
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """Static batch the data by `batch_size`

    Args:
        data: Iterable[{key, feat, label}]
        batch_size: batch size

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode="train"):
    """Dynamic batch the data until the total frames in batch
    reach `max_frames_in_batch`

    Args:
        data: Iterable[{key, feat, label}]
        max_frames_in_batch: max_frames in one batch

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert "speech_feat" in sample
        assert isinstance(sample["speech_feat"], torch.Tensor)
        new_sample_frames = sample["speech_feat"].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(
    data, batch_type="static", batch_size=16, max_frames_in_batch=12000, mode="train"
):
    """Wrapper for static/dynamic batch"""
    if mode == "inference":
        return static_batch(data, 1)
    else:
        if batch_type == "static":
            return static_batch(data, batch_size)
        elif batch_type == "dynamic":
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal("Unsupported batch type {}".format(batch_type))


def padding(data, use_spk_embedding=False, mode="train"):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]

    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_len = torch.tensor(
            [x["speech"].size(-1) for x in sample], dtype=torch.int32
        )
        order = torch.argsort(speech_len, descending=True)

        # variable length must be the first dimension
        speech = [sample[i]["speech"].transpose(0, 1) for i in order]
        speech = pad_sequence(speech, batch_first=True, padding_value=0).transpose(1, 2)

        speech_feat = [
            sample[i]["speech_feat"].transpose(0, 1) for i in order
        ]  # [T, C]
        speech_feat_len = torch.tensor(
            [i.size(0) for i in speech_feat], dtype=torch.int32
        )
        speech_feat = pad_sequence(
            speech_feat, batch_first=True, padding_value=0
        ).transpose(1, 2)

        mel = [sample[i]["mel"].transpose(0, 1) for i in order]  # [T, C]
        mel_len = torch.tensor([i.size(0) for i in mel], dtype=torch.int32)
        mel = pad_sequence(mel, batch_first=True, padding_value=0).transpose(
            1, 2
        )  # [B, C, T]

        utts = [sample[i]["utt"] for i in order]
        text = [sample[i]["text"] for i in order]

        text_token = [torch.tensor(sample[i]["text_token"]) for i in order]
        text_token_len = torch.tensor(
            [i.size(0) for i in text_token], dtype=torch.int32
        )
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)

        batch = {
            "speech": speech,
            "speech_len": speech_len,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "mel": mel,
            "mel_len": mel_len,
        }
        if mode == "inference":
            tts_text = [sample[i]["tts_text"] for i in order]
            tts_index = [sample[i]["tts_index"] for i in order]
            tts_text_token = [torch.tensor(sample[i]["tts_text_token"]) for i in order]
            tts_text_token_len = torch.tensor(
                [i.size(0) for i in tts_text_token], dtype=torch.int32
            )
            tts_text_token = pad_sequence(
                tts_text_token, batch_first=True, padding_value=-1
            )
            batch.update(
                {
                    "tts_text": tts_text,
                    "tts_index": tts_index,
                    "tts_text_token": tts_text_token,
                    "tts_text_token_len": tts_text_token_len,
                }
            )
        yield batch


class LamaVoiceCollator(object):
    def __init__(self, mode: str = "train"):
        self.mode = mode

    def __call__(self, sample):
        mode = self.mode
        assert isinstance(sample, list)
        speech_len = torch.tensor(
            [x["speech"].size(-1) for x in sample], dtype=torch.int32
        )
        order = torch.argsort(speech_len, descending=True)

        # variable length must be the first dimension
        speech = [sample[i]["speech"].transpose(0, 1) for i in order]
        speech = pad_sequence(speech, batch_first=True, padding_value=0).transpose(1, 2)

        speech_feat = [
            sample[i]["speech_feat"].transpose(0, 1) for i in order
        ]  # [T, C]
        speech_feat_len = torch.tensor(
            [i.size(0) for i in speech_feat], dtype=torch.int32
        )
        speech_feat = pad_sequence(
            speech_feat, batch_first=True, padding_value=0
        ).transpose(1, 2)

        mel = [sample[i]["mel"].transpose(0, 1) for i in order]  # [T, C]
        mel_len = torch.tensor([i.size(0) for i in mel], dtype=torch.int32)
        mel = pad_sequence(mel, batch_first=True, padding_value=0).transpose(
            1, 2
        )  # [B, C, T]

        utts = [sample[i]["utt"] for i in order]
        text = [sample[i]["text"] for i in order]

        text_token = [torch.tensor(sample[i]["text_token"]) for i in order]
        text_token_len = torch.tensor(
            [i.size(0) for i in text_token], dtype=torch.int32
        )
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)

        batch = {
            "speech": speech,
            "speech_len": speech_len,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "mel": mel,
            "mel_len": mel_len,
        }
        if mode == "inference":
            tts_text = [sample[i]["tts_text"] for i in order]
            tts_index = [sample[i]["tts_index"] for i in order]
            tts_text_token = [torch.tensor(sample[i]["tts_text_token"]) for i in order]
            tts_text_token_len = torch.tensor(
                [i.size(0) for i in tts_text_token], dtype=torch.int32
            )
            tts_text_token = pad_sequence(
                tts_text_token, batch_first=True, padding_value=-1
            )
            batch.update(
                {
                    "tts_text": tts_text,
                    "tts_index": tts_index,
                    "tts_text_token": tts_text_token,
                    "tts_text_token_len": tts_text_token_len,
                }
            )
        return batch


class Processor:
    @staticmethod
    def parquet_opener(*args, **kwargs):
        return parquet_opener(*args, **kwargs)

    @staticmethod
    def filter(*args, **kwargs):
        return filter(*args, **kwargs)

    @staticmethod
    def compute_linear(*args, **kwargs):
        return compute_linear(*args, **kwargs)

    @staticmethod
    def compute_mel(*args, **kwargs):
        return compute_mel(*args, **kwargs)

    @staticmethod
    def resample(*args, **kwargs):
        return resample(*args, **kwargs)

    @staticmethod
    def compute_fbank(*args, **kwargs):
        return compute_fbank(*args, **kwargs)

    @staticmethod
    def parse_embedding(*args, **kwargs):
        return parse_embedding(*args, **kwargs)

    @staticmethod
    def tokenize(*args, **kwargs):
        return tokenize(*args, **kwargs)

    @staticmethod
    def shuffle(*args, **kwargs):
        return shuffle(*args, **kwargs)

    @staticmethod
    def sort(*args, **kwargs):
        return sort(*args, **kwargs)

    @staticmethod
    def static_batch(*args, **kwargs):
        return static_batch(*args, **kwargs)

    @staticmethod
    def dynamic_batch(*args, **kwargs):
        return dynamic_batch(*args, **kwargs)

    @staticmethod
    def batch(*args, **kwargs):
        return batch(*args, **kwargs)

    @staticmethod
    def padding(*args, **kwargs):
        return padding(*args, **kwargs)
