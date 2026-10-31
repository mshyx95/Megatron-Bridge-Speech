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
    combine_audio_and_text_ordered, 
    combine_audio_and_text_random,
)

from transformers import BatchEncoding
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)

IGNORE_IDX = -100  


class AudioTaskPhiTransform(BaseTransform):

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config 
        self.audio_start_token = kwargs.get("audio_start_token", "")
        self.audio_end_token = kwargs.get("audio_end_token", "")
        # TODO: move context length to tokenizer
        self.context_length = config.get("context_length", 2048)
        self.shift_labels = config.get("shift_labels", False)
        self.pad_to_context_length = config.get("pad_to_context_length", True)
        self.concate_ordered = kwargs.get("concate_ordered", True)
        self.no_labels = False

    def _combine_interleaved_audio_and_text(self, results, audios, texts):
        """
        Args:
            results (Dict): examples with keys ["tokenizer", "audios", "texts", "num_audio_tokens", "audio_sizes"]
                The num_audio_toknes (List): are the size that after potential compression in encoder.
                audi_sisze (List): are the original feature size for each audio.
            audios (List[np.array): List of 2D arrays for audio features
            texts (List[str, None]): List of text sequence, separated with audio tokens (None here), it looks like:
            ["Here is an audio ", None, "Instruction: .... labels,....", "Here are audios", None, None, None, "Instruction: ....."]
        """
        
        input_ids = []
        if not self.no_labels:
            labels = []
        audio_positions = [] # * MAX_NUM_IMAGES
     
        raw_text = ""
        tokenizer = results['tokenizer']
        current_audio_id = 0

        # current_image_token_id = -int(1e9)
        # temporal solution to create a unique image_token_id within batch
        # TODO: re-design the logic to ditch image_id
        sample_hash = hash(tuple(texts))
        current_audio_token_id = -(sample_hash % int(1e8)) 
        all_audio_ids = []
        audio_idx = 0
        
        for text in texts:
            if is_audio_token(text):
                audio_tokens_length = results["audio_embed_sizes"][audio_idx]
                if len(input_ids) + audio_tokens_length > self.context_length:
                    # truncate the rest of the images
                    break

                # this is an image
                if self.audio_start_token:
                    input_ids.append(tokenizer.convert_tokens_to_ids(self.audio_start_token))
                    if not self.no_labels:
                        labels.append(IGNORE_IDX)
                    raw_text = raw_text + self.audio_start_token

                audio_positions.append(len(input_ids))
               
                current_audio_id += 1
                current_audio_token_id += 1

                all_audio_ids.append(current_audio_token_id)

                input_ids.extend([current_audio_token_id] * audio_tokens_length)
                if not self.no_labels:
                    labels.extend([IGNORE_IDX] * audio_tokens_length)
                raw_text += f"<audio_{current_audio_id}>"

                if self.audio_end_token:
                    input_ids.append(tokenizer.convert_tokens_to_ids(self.audio_end_token))
                    if not self.no_labels:
                        labels.append(IGNORE_IDX)
                    raw_text = raw_text + self.audio_end_token
                audio_idx += 1
            else:
                # this is a text
                tokens = tokenizer.encode(text, add_special_tokens=False)
                input_ids.extend(tokens)
                if not self.no_labels:
                    labels.extend(tokens)
                raw_text += text

        # here eos token = pad token
        # We have the end of text token in the raw text
        # input_ids.append(tokenizer.pad_token_id)
        # labels.append(tokenizer.pad_token_id)
        
        # input_ids needs to be padded to the context length
        if self.pad_to_context_length:
            input_ids.extend([tokenizer.pad_token_id] * (self.context_length - len(input_ids)))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids[:self.context_length]

        if not self.no_labels and self.shift_labels:
            # shift right
            labels = labels[1:]
        
        if not self.no_labels and self.pad_to_context_length:
            labels = labels + [IGNORE_IDX] * (self.context_length - len(labels))
        if not self.no_labels:
            labels = labels[:self.context_length]

        original_num_audios = len(audios)
        audios = audios[:current_audio_id]
        assert len(audios) == current_audio_id, f"Number of audios in the text does not match the number of audios in the audios: {len(audios)} != {current_audio_id}"

        # make sure at least one image is present
        # data loading will fail if no image is present when distributed training
        if len(audios) == 0:
            logger.info(f"total original number of images: {original_num_audios}, current number of images for this sample is zero")
            return None 

        audio_embed_sizes = torch.tensor(results["audio_embed_sizes"])
        if len(audio_embed_sizes):
            audio_embed_sizes = audio_embed_sizes[:current_audio_id]

        audio_ids = results["audio_ids"][:current_audio_id]
        
        results.update({
            "audios": audios,
            "inputs":  BatchEncoding(dict(input_ids=input_ids,
                attention_mask=torch.ones(len(input_ids), dtype=torch.long))),
            "audio_positions": torch.tensor(audio_positions, dtype=torch.long),
            "audio_embed_sizes": audio_embed_sizes,
            "audio_frames": torch.tensor(results["audio_frames"]),
            "audio_ids": audio_ids,
        })
        if not self.no_labels:
            results.update({"labels": BatchEncoding(dict(input_ids=torch.tensor(labels)))})

        return results
    
    def transform(self, results: Dict) -> Dict:
        # transform n audio-text pairs into a squence
        assert "texts" in results, f"No texts/image_text_interleaved found in the annotations {results['annotations']}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]

        if self.concate_ordered:
            audio, interleaved_data = combine_audio_and_text_ordered(audios, texts)
        else:
            audios, interleaved_data = combine_audio_and_text_random(audios, texts)

        results = self._combine_interleaved_audio_and_text(results, audios, interleaved_data)
        
        return results
