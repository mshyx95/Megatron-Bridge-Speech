# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict
import random
import re
import torch

from .base import BaseTransform
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
    is_audio_token,
    process_asr_annotation,
    add_audio_text_in_conversation,
    pad_features, 
)
from transformers import BatchEncoding
from .conversation_utils import conv_templates

IGNORE_IDX = -100


class AudioTextConversationTransform(BaseTransform):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config 
        self.audio_start_token = kwargs.get("audio_start_token", "")
        self.audio_end_token = kwargs.get("audio_end_token", "")
        # TODO: move context length to tokenizer
        self.context_length = config.get("context_length", 2048)
        self.shift_labels = config.get("shift_labels", False)
        self.pad_to_context_length = config.get("pad_to_context_length", True)
        self.conv_template = kwargs.get("conv_template", None)
        self.prompt_formatter = kwargs.get("prompt_formatter", None) # pre-formatting some speech annotation to certain tasks
        self.fixed_prompt = kwargs.get("fixed_prompt", True)
        self.instruction_first = kwargs.get("instruction_first", False)
        self.bias_after_audio = kwargs.get("bias_after_audio", False)
        self.batch_format = kwargs.get("batch_format", "samples_to_sequence")

        assert self.batch_format in ["samples_to_sequence", "samples_to_batch"]
        assert self.conv_template, "conv_template has to be set for AudioTextConversationTransform"
        
        self.conv_template = conv_templates[self.conv_template].copy() 
    
    def transform(self, results: Dict) -> Dict:
        
        if self.batch_format == "samples_to_sequence":
            results = self.transform_concat_to_context_length(results)
        elif self.batch_format == "samples_to_batch":
            results = self.transform_concat_to_batch(results)
        else:
            raise NotImplementedError
        
        return results

    def transform_concat_to_batch(self, results: Dict) -> Dict:

        assert "texts" in results, f"No texts/image_text_interleaved found in the annotations {results['annotations']}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]
        audio_embed_sizes = results["audio_embed_sizes"]
        user_prompts = results["user_prompts"]
        languages = results["languages"]
        spkid = results.get("spkid", [None] * len(texts))
        contexts = results["contexts"]

        input_ids = []
        labels = []

        audio_embed_sizes_1d = []
        is_text_sample = False
        for text, embed_size, user_prompt, language, context in zip(texts, audio_embed_sizes, user_prompts, languages, contexts):
            conv_template = self.conv_template.copy()
            conv_template.tokenizer = results["tokenizer"]
            conv_template = add_audio_text_in_conversation([text], [user_prompt], [language], [context], conv_template, self.prompt_formatter, self.fixed_prompt, self.instruction_first, self.bias_after_audio)
            ret, input_id, label, audio_positions = conv_template.get_prompt(is_train = True, audio_input_id = -1,
                                                            only_add_first_audio_occurrence = False, num_audio_tokens=embed_size)
            
            audio_embed_sizes_1d.extend(embed_size)
            input_ids.append(input_id)
            labels.append(label)

            if sum(embed_size) == 0:
                is_text_sample = True
 
        max_input_id_len = max([len(input_id) for input_id in input_ids])
        max_label_len = max([len(label) for label in labels])
        tokenizer = results["tokenizer"]
       
        input_ids = [input_id + [tokenizer.pad_token_id] * (max_input_id_len - len(input_id)) for input_id in input_ids]
        labels = [label + [IGNORE_IDX] * (max_label_len - len(label)) for label in labels]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        if self.shift_labels:
            # shift right
            # The shift is performed in huggingface model as well so the shift_lables=False
            labels = labels[1:]
        
        audio_embed_sizes = torch.tensor(audio_embed_sizes_1d)
        audio_frames = torch.tensor(results["audio_frames"])
        
        results.update({
            "audios": audios,
            "inputs":  BatchEncoding(dict(input_ids=input_ids,
                attention_mask=torch.ones(len(input_ids), dtype=torch.long))),
            "labels": BatchEncoding(dict(input_ids=labels)),
            "audio_embed_sizes": audio_embed_sizes,
            "audio_frames": audio_frames,
            "is_text_sample": is_text_sample,
            "spkid": spkid,
        })

        if self.prompt_formatter in ["asr-ctc"]:
            ctc_labels = []
            for i in range(len(texts)):
                assert len(texts[i]) == 1
                ctc_labels.append(texts[i][0])
            
            ctc_labels = [tokenizer.encode(lab) for lab in ctc_labels]
            ctc_label_lens = [ len(lab) for lab in ctc_labels]
            max_ctc_label_len = max(ctc_label_lens)
            ctc_labels = [lab + [IGNORE_IDX] * (max_ctc_label_len - len(lab)) for lab in ctc_labels]
            ctc_labels = torch.tensor(ctc_labels, dtype=torch.long)
        
            results.update({
                'ctc_labels': BatchEncoding(dict(input_ids=ctc_labels)),
                'ctc_label_lens': torch.tensor(ctc_label_lens),
            })
        return results   

    def transform_concat_to_context_length(self, results: Dict) -> Dict:
        # transform n audio-text pairs into a squence
        assert "texts" in results, f"No texts/image_text_interleaved found in the annotations {results['annotations']}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]

        user_prompts = results["user_prompts"]
        languages = results["languages"]
        contexts = results["contexts"]

        conv_template = self.conv_template.copy()
        conv_template.tokenizer = results["tokenizer"]
        
        audio_embed_sizes_1d = []
        is_text_sample = False
        for embed_size in results["audio_embed_sizes"]:
            if sum(embed_size) == 0:
                is_text_sample = True
            audio_embed_sizes_1d.extend(embed_size)

        conv_template = add_audio_text_in_conversation(texts, user_prompts, languages, contexts, conv_template, self.prompt_formatter, self.fixed_prompt, self.instruction_first, self.bias_after_audio)


        ret, input_ids, labels, audio_positions = conv_template.get_prompt(is_train = True, 
                                                                            audio_input_id = -1,
                                                                            only_add_first_audio_occurrence = False,
                                                                            num_audio_tokens=audio_embed_sizes_1d)
   
        tokenizer = results["tokenizer"]
        if self.pad_to_context_length:
            # input_ids.extend([tokenizer.pad_token_id] * (self.context_length - len(input_ids)))
            input_ids = input_ids[:self.context_length]
            # Comments on the above line:
            # Removing the rest might cause issue for ASR tasks (some groundtruth tokens are filtered)
            # WE can ensure the total length is smaller than the context length in dataloader
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        if self.shift_labels:
            # shift right
            # The shift is performed in huggingface model as well so the shift_lables=False
            labels = labels[1:]
        
        if self.pad_to_context_length:
            # labels = labels + [IGNORE_IDX] * (self.context_length - len(labels))
            labels = labels[:self.context_length]
        
        audio_embed_sizes = torch.tensor(audio_embed_sizes_1d)
        audio_frames = torch.tensor(results["audio_frames"])
        
        results.update({
            "audios": audios,
            "inputs":  BatchEncoding(dict(input_ids=input_ids,
                attention_mask=torch.ones(len(input_ids), dtype=torch.long))),
            "labels": BatchEncoding(dict(input_ids=torch.tensor(labels))),
            "audio_embed_sizes": audio_embed_sizes,
            "audio_frames": audio_frames,
            "is_text_sample": is_text_sample,
        })

        return results
