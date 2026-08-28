# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Tuple

import torch
from torch.utils.data import default_collate
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer 

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import pad_features


@dataclass
class LMDataCollator:
    """Data collator compatible with :class:`phyagi.datasets.lm.lm_dataset.LMDataset`,
    :class:`phyagi.datasets.masked_lm.masked_lm_dataset.MaskedLMDataset` and
    :class:`phyagi.datasets.stream_lm.stream_lm_dataset.StreamLMDataset`.

    This class applies specific token masking based on given parameters.

    Args:
        ignore_token_ids: List of token IDs to be ignored in the label. Label tokens with these IDs will be replaced with the ``ignore_index`` value.
        ignore_token_id_range: Range of token IDs (inclusive) to be ignored (min, max) in the label. Any label token within this range will be replaced with the ``ignore_index`` value.
        ignore_index: The index that will be used to replace ignored tokens in the labels.

    Examples:
        >>> collator = LMDataCollator(ignore_token_ids=[0, 1], ignore_index=-1)
        >>> examples = [{'labels': torch.tensor([0, 1, 2, 3])}]
        >>> batch = collator(examples)
        >>> batch['labels']
        >>> tensor([-1, -1,  2,  3])

    """

    ignore_token_ids: Optional[List[int]] = field(default=None, metadata={"help": "List of token IDs to be ignored."})

    ignore_token_id_range: Optional[Tuple[int, int]] = field(
        default=None, metadata={"help": "Range of token IDs to be ignored (min, max)."}
    )

    ignore_index: int = field(default=-100, metadata={"help": "Index to be ignored in the labels."})

    def __post_init__(self) -> None:
        self.ignore_token_ids = torch.tensor(self.ignore_token_ids) if self.ignore_token_ids else None

    def _mask_tokens(self, labels: torch.LongTensor) -> torch.LongTensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)

        if self.ignore_token_ids is not None:
            mask |= torch.isin(labels, self.ignore_token_ids)

        if self.ignore_token_id_range is not None:
            lower, upper = self.ignore_token_id_range

            lower = lower or float("-inf")
            upper = upper or float("inf")

            mask |= (labels >= lower) & (labels <= upper)

        labels[mask] = self.ignore_index

        return labels

    def __call__(self, examples: List[Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        batch = default_collate(examples)

        if self.ignore_token_ids is not None or self.ignore_token_id_range is not None:
            if batch.get("labels", None) is not None:
                batch["labels"] = self._mask_tokens(batch["labels"])

        return batch

@dataclass
class VLMDataCollator(LMDataCollator):
    """Data collator compatible with :class:`megatron.bridge.data.speech_chunk_streaming.vlm.vlm_dataset.VLMDataset`."""

    image_key: Optional[str] = field(default='input_embeds', metadata={"help": "Key to access image embeddings in examples."})

    def __post_init__(self) -> None:
        self.ignore_token_ids = None
        self.ignore_token_id_range = (None, -1)

        return super().__post_init__()

    def __call__(self, examples: List[Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        input_embeds = [e.pop("input_embeds", torch.tensor([])) for e in examples]
        image_sizes = [e.pop("image_sizes", torch.tensor([])) for e in examples]

        batch = super().__call__(examples)
        batch[self.image_key] = torch.concat(input_embeds)
        batch["image_sizes"] = torch.concat(image_sizes)

        return batch

@dataclass
class VLMDataDynamicCollator(LMDataCollator):
    """Data collator compatible with :class:`megatron.bridge.data.speech_chunk_streaming.vlm.vlm_dataset.VLMDataset`. Dynamic padding to max length in batch"""

    tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]] = field(
        default=None, metadata={"help": "Instance of tokenizer to use."}
    )
    image_key: Optional[str] = field(default='input_embeds', metadata={"help": "Key to access image embeddings in examples."})
    
    def __post_init__(self) -> None:
        self.ignore_token_ids = None
        self.ignore_token_id_range = (None, -1)
        self.tokenizer = _load_tokenizer(self.tokenizer)

        return super().__post_init__()

    def __call__(self, examples: List[Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:

        input_embeds = [e.pop("input_embeds", torch.tensor([])) for e in examples]

        input_ids = [e["input_ids"] for e in examples]
        labels = [e["labels"] for e in examples]
        image_sizes = [e.get('image_sizes', torch.tensor([])) for e in examples]
        
        # pad input ids
        max_length = max([_input_ids.shape[0] for _input_ids in input_ids])
        input_ids = [torch.cat([i, torch.tensor([self.tokenizer.pad_token_id] * (max_length - len(i)),  dtype=torch.long)]) for i in input_ids]

        max_length = max([_labels.shape[0] for _labels in labels])
        # pad labels to max length
        labels = [torch.cat([l, torch.tensor([-100] * (max_length - len(l)), dtype=torch.long)]) for l in labels]

        batch = {}
        batch['input_ids'] = torch.stack(input_ids)
        batch['labels'] = torch.stack(labels)
        batch[self.image_key] = torch.concat(input_embeds)
        batch["image_sizes"] = torch.concat(image_sizes)

        return batch


@dataclass
class ALMDataDynamicCollator(LMDataCollator):
    """Data collator compatible with :class:`phyagi.datasets.alm.alm_dataset.ALMDataset`. Dynamic padding to max length in batch"""

    tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]] = field(
        default=None, metadata={"help": "Instance of tokenizer to use."}
    )
    audio_key: Optional[str] = field(default='input_embeds', metadata={"help": "Key to access audio embeddings in examples."})
    padding_mask: Optional[bool] = field(default=False, metadata={"help": "Whether to apply padding mask to the entire sequence."})

    def __post_init__(self) -> None:
        self.ignore_token_ids = None
        self.ignore_token_id_range = (None, -1)
        self.tokenizer = _load_tokenizer(self.tokenizer)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return super().__post_init__()

    def _maybe_dump_inputs(
        self,
        examples: List[Dict[str, torch.LongTensor]],
        batch: Dict[str, torch.LongTensor],
    ) -> None:
        dump_path = os.environ.get("ALM_INPUT_DUMP_PATH")
        if not dump_path:
            return

        dump_limit = int(os.environ.get("ALM_INPUT_DUMP_LIMIT", "1"))
        dump_count = getattr(self, "_input_dump_count", 0)
        if dump_count >= dump_limit:
            return
        self._input_dump_count = dump_count + 1

        decode_limit = int(os.environ.get("ALM_INPUT_DUMP_DECODE_LIMIT", "1024"))
        preview_limit = int(os.environ.get("ALM_INPUT_DUMP_PREVIEW_LIMIT", "128"))

        def summarize_tensor(value):
            if not torch.is_tensor(value):
                return repr(value)
            flat = value.detach().cpu().reshape(-1)
            summary = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "preview": flat[:preview_limit].tolist(),
            }
            if flat.numel() > 0:
                if torch.is_floating_point(flat):
                    flat_float = flat.float()
                    summary.update(
                        {
                            "min": flat_float.min().item(),
                            "max": flat_float.max().item(),
                            "mean": flat_float.mean().item(),
                            "std": flat_float.std().item() if flat_float.numel() > 1 else 0.0,
                        }
                    )
                else:
                    summary["min"] = flat.min().item()
                    summary["max"] = flat.max().item()
                    non_ignore = flat[flat != self.ignore_index]
                    summary["non_ignore_count"] = int(non_ignore.numel())
                    summary["non_ignore_preview"] = non_ignore[:preview_limit].tolist()
                    unique, counts = torch.unique(flat, return_counts=True)
                    if unique.numel() <= 64:
                        summary["value_counts"] = [
                            [unique[i].item(), counts[i].item()] for i in range(unique.numel())
                        ]
                    else:
                        summary["unique_count"] = int(unique.numel())
                        summary["unique_preview"] = unique[:64].tolist()
            return summary

        def decode_ids(ids):
            if not torch.is_tensor(ids):
                return None
            ids = ids.detach().cpu().reshape(-1)
            if ids.numel() == 0:
                return ""
            ids = ids[:decode_limit].tolist()
            vocab_size = len(self.tokenizer)
            ids = [token_id for token_id in ids if 0 <= token_id < vocab_size]
            return self.tokenizer.decode(ids, skip_special_tokens=False)

        def decode_labels(labels):
            if not torch.is_tensor(labels):
                return None
            labels = labels.detach().cpu().reshape(-1)
            valid = labels[labels != self.ignore_index]
            if valid.numel() == 0:
                return ""
            valid = valid[:decode_limit]
            vocab_size = len(self.tokenizer)
            valid = valid[(valid >= 0) & (valid < vocab_size)]
            return self.tokenizer.decode(valid.tolist(), skip_special_tokens=False)

        record = {
            "dump_index": dump_count,
            "batch_keys": sorted(batch.keys()),
            "batch": {key: summarize_tensor(value) for key, value in batch.items()},
            "decoded": {},
            "examples": [],
        }
        if "input_ids" in batch and batch["input_ids"].size(0) > 0:
            record["decoded"]["input_ids_0"] = decode_ids(batch["input_ids"][0])
        if "labels" in batch and batch["labels"].size(0) > 0:
            record["decoded"]["labels_0_non_ignore"] = decode_labels(batch["labels"][0])

        for example in examples[: min(len(examples), 3)]:
            example_summary = {}
            for key, value in example.items():
                if key == "input_embeds":
                    example_summary[key] = [summarize_tensor(item) for item in value[:3]]
                elif torch.is_tensor(value):
                    example_summary[key] = summarize_tensor(value)
                else:
                    example_summary[key] = repr(value)
            if torch.is_tensor(example.get("input_ids")):
                example_summary["decoded_input_ids"] = decode_ids(example["input_ids"])
            if torch.is_tensor(example.get("labels")):
                example_summary["decoded_labels_non_ignore"] = decode_labels(example["labels"])
            record["examples"].append(example_summary)

        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __call__(self, examples: List[Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        def _codec_input_ids(value):
            if hasattr(value, "input_ids"):
                value = value.input_ids
            elif isinstance(value, dict):
                value = value["input_ids"]
            return value if value.dim() == 3 else value.unsqueeze(0)
        
        audios = []
        for e in examples:
            if "input_embeds" in e and not e["is_text_sample"]:
                audios.extend(e.pop("input_embeds"))
        
        if len(audios) > 0:
            input_embeds, nframe = pad_features(audios, return_pt=True)

            audio_attention_mask = input_embeds.new_ones(input_embeds.size()[:2], dtype=torch.long)
            for i in range(input_embeds.shape[0]):
                audio_attention_mask[i, nframe[i]:].fill_(0)
        else:
            input_embeds = torch.concat([torch.tensor([]) for e in examples])
            audio_attention_mask = torch.concat([torch.tensor([]) for e in examples])

        spkid = []
        if "spkid" in examples[0] and examples[0]["spkid"] is not None:
            for e in examples:
                spkid.extend(e.pop("spkid"))

        if len(spkid) > 0 and any([sid is not None for sid in spkid]):
            spkid2 = [sid for sid in spkid if sid is not None]   # only keep non-None spkid. Model should handle missing spkid
            if len(spkid2) < len(spkid):
                print(f"Warning: some samples in the batch do not have spkid. Only {len(spkid2)}/{len(spkid)} samples have spkid.")

            spkid_label, spkid_label_lens = pad_features(spkid2, return_pt=True)
        else:
            spkid_label = None

        input_ids = [e["input_ids"] if e["input_ids"].dim() == 2 else e["input_ids"].unsqueeze(0) for e in examples ] if "input_ids" in examples[0] else None
        if "labels" in examples[0]:
            labels = [e["labels"] if e["labels"].dim() == 2 else e["labels"].unsqueeze(0) for e in examples]
        if "input_codecs" in examples[0]:
            input_codecs = [_codec_input_ids(e["input_codecs"]) for e in examples]
            label_codecs = [_codec_input_ids(e["label_codecs"]) for e in examples]
        if "ctc_labels" in examples[0]:
            ctc_labels = [ e["ctc_labels"] for e in examples]
            ctc_label_lens = [ e["ctc_label_lens"] for e in examples]

        # Forward-tokenize (raw-audio) path: keep answer_audio + seq_plan as lists
        # (never stacked / default_collated / moved to device here). No input_ids/
        # labels/input_codecs on this path — they are built inside model.forward.
        answer_audio = None
        if "answer_audio" in examples[0]:
            answer_audio, seq_plan = [], []
            for e in examples:
                answer_audio.extend(e["answer_audio"])
                seq_plan.extend(e["seq_plan"])
            forward_codec_cfg = examples[0].get("codec_cfg")

        audio_embed_sizes = [e.get('audio_embed_sizes', torch.tensor([])) for e in examples]
        audio_frames = [e.get('audio_frames', torch.tensor([])) for e in examples]
        
        if "audio_ids" in examples[0]:
            audio_ids = [audio_id for audio_id in e["audio_ids"] for e in examples]
        if "segments" in examples[0]:
            segments = [segment for segment in e["segments"] for e in examples]

        # pad input ids
        if input_ids is not None:
            max_input_length = max([_input_ids.shape[1] for _input_ids in input_ids])
            input_ids = [torch.cat([i, i.new_zeros((i.size(0), max_input_length-i.size(1))).fill_(self.tokenizer.pad_token_id)], dim=1) for i in input_ids]
        if "labels" in examples[0]:
            max_label_length = max([_labels.shape[1] for _labels in labels])
            # pad labels to max length
            labels = [torch.cat([l, l.new_zeros((l.size(0), max_label_length-l.size(1))).fill_(-100)], dim=1) for l in labels]
        if "input_codecs" in examples[0]:
            input_codecs = [
                torch.cat(
                    [c, c.new_full((c.size(0), c.size(1), max_input_length - c.size(2)), self.tokenizer.pad_token_id)],
                    dim=2,
                )
                for c in input_codecs
            ]
            label_codecs = [
                torch.cat([c, c.new_full((c.size(0), c.size(1), max_label_length - c.size(2)), -100)], dim=2)
                for c in label_codecs
            ]
        if "ctc_labels" in examples[0]:
            max_length = max([ _label.shape[1] for _label in ctc_labels])
            ctc_labels = [torch.cat([l, l.new_zeros((l.size(0), max_length-l.size(1))).fill_(-100)], dim=1) for l in ctc_labels]

        batch = {}
        batch['input_ids'] = torch.cat(input_ids) if input_ids is not None else None
        if batch['input_ids'] is None:
            del batch['input_ids']
        if "labels" in examples[0]:
            batch['labels'] = torch.cat(labels)
        if "input_codecs" in examples[0]:
            batch["input_codecs"] = torch.cat(input_codecs)
            batch["label_codecs"] = torch.cat(label_codecs)
        if "ctc_labels" in examples[0]:
            batch['ctc_labels'] = torch.cat(ctc_labels)
            batch['ctc_label_lens'] = torch.concat(ctc_label_lens).long()
        if answer_audio is not None:
            batch["answer_audio"] = answer_audio
            batch["seq_plan"] = seq_plan
            if forward_codec_cfg is not None:
                batch["codec_cfg"] = forward_codec_cfg

        if self.padding_mask and "input_ids" in batch:
            batch['attention_mask'] = batch['input_ids'] != self.tokenizer.pad_token_id
        
        batch[self.audio_key] = input_embeds
        batch['audio_attention_mask'] = audio_attention_mask
        batch['audio_embed_sizes'] = torch.concat(audio_embed_sizes).long()
        batch['audio_frames'] = torch.concat(audio_frames).long()
        
        if "audio_ids" in examples[0]:
            batch['audio_ids'] = audio_ids
        if "segments" in examples[0]:
            batch['segments'] = segments
        if spkid_label is not None:
            batch['spkid'] = spkid_label.long()
            batch['spkid_lens'] = torch.tensor(spkid_label_lens).long()

        self._maybe_dump_inputs(examples, batch)
        return batch


def _load_tokenizer(tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]]) -> PreTrainedTokenizerBase:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)

    if isinstance(tokenizer, dict):
        return AutoTokenizer.from_pretrained(**tokenizer)

    return tokenizer
