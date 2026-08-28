# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os  
import os.path as osp 
import base64
import glob
import math
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger
from .process_image_text_chunk import FlorenceSampler

logger = get_logger(__name__)


class VLMDataset(Dataset):
    """Vision-language modeling dataset.

    This dataset is used for vision-language modeling tasks. It takes a contiguous array
    of tokens and image tokens, and convert it into a dataset of sequences of length ``seq_len``
    by sliding a window of size ``seq_len`` over the array of tokens.

    Each sequence is composed of ``input_ids`` and ``labels`` tensors.

    """

    def __init__(
        self,
        input_ids: np.array,
        img_file: str,
        seq_len: int = 2048,
        shift_labels: bool = False,
        img_processor: Optional[str] = None,
        n_img_tokens: int = 256,
    ) -> None:
        """Initialize the dataset.

        Args:
            input_ids: Inputs array (encoded data).
            img_file: Path to the ``.tsv`` or ``.npy`` image file. # update it to the directory for image_chunks 
            seq_len: Sequence length.
            shift_labels: Whether labels should be shifted by one position.
            img_processor: Image processor name.
            n_img_tokens: Number of image tokens.
        """

        super().__init__()

        # if 'florence' in img_file:
        #     process_root = img_file 
        #     root = osp.join(img_file, 'florence_root')
        #     self.sampler = FlorenceSampler(
        #         process_root=process_root,
        #         root=root, 
        #         max_len=seq_len,
        #         shift_labels=shift_labels 
        #         )
        #     self.n_sequences = 1000000 
        # else:
        self.sampler = None 
        # `input_ids` should not be sliced since they could be memory mapped
        self.input_ids = input_ids
        self.seq_len = seq_len

        self.n_input_ids = ((len(input_ids) - 1) // seq_len) * seq_len + 1
        self.n_sequences = math.ceil((self.n_input_ids - 1) / self.seq_len)

        self.label_offset = 1 if shift_labels else 0

        self.imgs = None
        self.img_processor = None
        self.n_img_tokens = n_img_tokens
        self.img_file = img_file 

        # If `img_file` is an `.npy`, we need to load the images into memory
        if img_file is not None: 
            if img_file.endswith(".npy"):
                self.imgs = np.load(img_file, mmap_mode="r")

            # If `img_file` is a `.tsv`, we need to load an image processor to decode the images
            elif img_file.endswith(".tsv"):
                assert img_processor is not None, "`img_processor` must be specified for `.tsv` files."
                img_files = sorted(glob.glob(img_file))

                self.imgs = pd.concat([pd.read_csv(f, sep="\t") for f in img_files], ignore_index=True).to_numpy()
                self.img_processor = CLIPImageProcessor.from_pretrained(img_processor)
            
            elif osp.isdir(img_file):
                assert img_processor is not None, "`img_processor` must be specified for an image folder."
                self.img_file = img_file  # img_file is actually a directory 
                self.imgs = None 
                self.chunk_size = 1000 # number of images in each image chunk 
                self.file_dict = {}
                self.img_processor = CLIPImageProcessor.from_pretrained(img_processor)


    def _decode_image(self, img_b64: str, convert_rgb: bool = True) -> np.array:
        img = Image.open(BytesIO(base64.b64decode(img_b64)))

        if convert_rgb:
            img = img.convert("RGB")
        img = np.array(img)
        return img
    
    def _get_img_file(self, img_idx):
        # for the given idx, find the name of the image file and index of the image in the file 
        idx = img_idx - 1 # the image id starts from 1 instead of 0
        chunk_id = idx // self.chunk_size
        index = idx % self.chunk_size 
        img_file_name = osp.join(self.img_file, 'image-{:010d}'.format(chunk_id))
        return img_file_name, index  

    def _ensure_img_file_opened(self, img_file_name):
        if img_file_name not in self.file_dict:
            self.file_dict[img_file_name] = open(img_file_name, 'rb')
    
    def _decode_bytes(self, img_idx) -> np.array:
        """
        Args:
            filename (str): path to the binary chunk file
            j (int): index of the example to retrieve
        Returns:
            A bytes object representing the j-th example from the binary chunk.
        """
        
        endianness = "little"
        index_byte_len = 8  # 64 bit indices

        img_file_name, j = self._get_img_file(img_idx)
        # with open(filename, 'rb') as f:
        self._ensure_img_file_opened(img_file_name)
        f = self.file_dict[img_file_name] # load text file 

        # Seek to the j-th offset value and read it
        f.seek(j * index_byte_len)
        offset_j = int.from_bytes(f.read(index_byte_len), byteorder=endianness)

        # Seek to the (j + 1)-th offset value and read it
        f.seek((j + 1) * index_byte_len)
        offset_j1 = int.from_bytes(f.read(index_byte_len), byteorder=endianness)

        # Seek to the j-th example and read it
        f.seek(offset_j)
        example_j = f.read(offset_j1 - offset_j)
        
        img_io = BytesIO(example_j)
        image = Image.open(img_io).convert("RGB")
        image = np.array(image)
        return image 


    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # if self.sampler is not None:
        #     input_ids, input_embeds, labels = self.sampler.sample()
        #     return {
        #         "input_ids": input_ids,
        #         "input_embeds": input_embeds,
        #         "labels": labels,
        #         }
        start_idx = idx * self.seq_len

        seq_len = min(self.seq_len, self.n_input_ids - 1 - start_idx)
        input_ids = self.input_ids[start_idx : (start_idx + seq_len + self.label_offset)].astype(np.int64)
        include_image = len(input_ids[input_ids < 0])
        if include_image:
            if self.n_img_tokens > 0:
                # Split `input_ids` into chunks on image tokens
                img_tokens_idx = (input_ids < 0).nonzero()[0]
                input_ids_chunks = np.split(input_ids, img_tokens_idx)

                # For each valid chunk, insert more `n_img_tokens - 1` image tokens
                for i, chunk in enumerate(input_ids_chunks):
                    if chunk.size > 0:
                        if chunk[0] < 0:
                            input_ids_chunks[i] = np.insert(chunk, 0, np.repeat(chunk[0], self.n_img_tokens - 1))

                # Ensure that `input_ids` has a valid sequence length
                input_ids = np.concatenate(input_ids_chunks)[: seq_len + self.label_offset]

        labels = torch.as_tensor(input_ids[self.label_offset : seq_len + self.label_offset].astype(np.int64))
        input_ids = torch.as_tensor(input_ids[:seq_len])

        if include_image:
            if input_ids[-1] < 0:
                if input_ids[-self.n_img_tokens] != input_ids[-1]:
                    last_value = input_ids[-1].clone()
                    # logger.info('dropping last image')
                    j = -1
                    while input_ids[j] == last_value and j >= -self.n_img_tokens:
                        input_ids[j] = 0 
                        j -= 1 

            img_ids = []
            input_embeds = torch.tensor([])
            # img_ids = np.unique(input_ids[input_ids < 0]) * -1
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            MAX_INPUT_ID = 1e9
            with torch.no_grad():
                positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)
        
            if len(positions.tolist()) > 0:
                with torch.no_grad():
                    g_values = abs(input_ids[positions[:, 0], positions[:, 1]])
                selected_g_values = []
                prev_pos = []
                for i, g in enumerate(g_values.cpu().numpy().tolist()):
                    if len(selected_g_values) == 0 or g != selected_g_values[-1] or positions[i, 0] != prev_pos[-1][0]:
                        selected_g_values.append(g)
                        prev_pos.append(positions[i])
                img_ids = selected_g_values

            input_ids = input_ids.view(input_shape)
            if len(img_ids) > 0:
                if self.img_processor is None:
                    # If `img_processor` is not available, stack the image tensors
                    input_embeds = np.stack([self.imgs[i - 1] for i in img_ids])
                    input_embeds = torch.tensor(input_embeds)
                else:
                    # If `img_processor` is available, decode and preprocess the images into tensors
                    if self.imgs is not None:
                        input_embeds = [self._decode_image(self.imgs[i - 1][1]) for i in img_ids]
                        input_embeds = self.img_processor.preprocess(input_embeds, return_tensors="pt")["pixel_values"]
                    else:
                        # If `img_processor` is available, decode and preprocess the images into tensors
                        if self.imgs is not None:
                            input_embeds = [self._decode_image(self.imgs[i - 1][1]) for i in img_ids]
                            input_embeds = self.img_processor.preprocess(input_embeds, return_tensors="pt")["pixel_values"]
                        else:
                            # logger.info(f'img_ids: {img_ids} ...')
                            # input_embeds = [self._decode_bytes(i) for i in img_ids]
                            input_embeds = []
                            for i in img_ids:
                                try:
                                    embed = self._decode_bytes(i)
                                    embed = self.img_processor.preprocess([embed], return_tensors="pt")["pixel_values"][0]
                                    # logger.info(f'embed size = {embed.shape}, type = {embed.dtype}')
                                except:
                                    # logger.info(f'error decoding images id = {i}')
                                    # embed = np.zeros((512, 512, 3), dtype=np.uint8)
                                    embed = torch.zeros((3, 224, 224), dtype=torch.float32)
                                input_embeds.append(embed)
                            # logger.info(f'input_embeds:  {len(input_embeds)}')
                            input_embeds = torch.stack(input_embeds)
                            # logger.info(f'input_embeds = {input_embeds.shape}, type={input_embeds.dtype}')
                            #input_embeds = self.img_processor.preprocess(input_embeds, return_tensors="pt")["pixel_values"]
        else:
            img_ids = []
            input_embeds = torch.tensor([])

        assert len(img_ids) == len(input_embeds), 'number of image ids and input embeds do not match!!'     
        return {
            "input_ids": input_ids,
            "input_embeds": input_embeds,
            "labels": labels,
        } 


if __name__ == '__main__':
    import numpy as np 
    from torch.utils.data import DataLoader

    input_arr = np.load('/data/datasets/llava1_5/train.npy', mmap_mode='r')
    image_file = "/data/datasets/llava1_5/bytes"
    img_processor = "openai/clip-vit-large-patch14"
    seq_len = 4096
    
    dataset = VLMDataset(input_arr, image_file, seq_len=seq_len, img_processor=img_processor)
    # image_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    for i in range(len(dataset)):
        item = dataset[i]

    
    import pdb 
    pdb.set_trace()
    # for i, batch in enumerate(image_loader):
    #     import pdb 
    #     pdb.set_trace()
    # for i in range(len(dataset)):
    #     item = dataset[i]
    #     import pdb 
    #     pdb.set_trace()