# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""helper function for activation checkpointing"""

from typing import Union, Dict, Callable
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
)


def validate_checkpointing_config(activation_checkpointing):
    """validate activation checkpointing configuration"""
    if isinstance(activation_checkpointing, str):
        assert activation_checkpointing in (
            "",
            "checkpoint",
            "offload",
        ), "activation_checkpointing has to be a dict or a str in ('', 'checkpoint', 'offload')."
    elif isinstance(activation_checkpointing, dict):
        assert activation_checkpointing.get("module", "transformer") in (
            "transformer",
            "attention",
        ), "module in activation_checkpointing has to be in ('transformer', 'attention')."
    else:
        raise ValueError("activation_checkpointing has to be a str or dict.")


def embedding_checkpoint_wrapper(
    activation_checkpointing: Union[str, Dict],
) -> Callable:
    """return encoder embedding activation checkpoint wrapper"""
    validate_checkpointing_config(activation_checkpointing)

    if isinstance(activation_checkpointing, str):
        if activation_checkpointing:
            if activation_checkpointing == "offload":
                return offload_wrapper
            return partial(checkpoint_wrapper)
        return lambda x: x

    if isinstance(activation_checkpointing, dict):
        enabled = activation_checkpointing.get("embed", False)
        if enabled:
            offloading = activation_checkpointing.get("offload", False)
            if offloading:
                return offload_wrapper
            impl = (
                CheckpointImpl.REENTRANT
                if activation_checkpointing.get("reentrant", False)
                else CheckpointImpl.NO_REENTRANT
            )
            return partial(checkpoint_wrapper, checkpoint_impl=impl)
        return lambda x: x
    raise ValueError("Invalid activation_checkpointing config")


def encoder_checkpoint_wrapper(
    activation_checkpointing: Union[str, Dict],
    layer_cls: type,
    idx: int = 0,
) -> Callable:
    """return encoder activation checkpoint wrapper"""
    validate_checkpointing_config(activation_checkpointing)

    if isinstance(activation_checkpointing, str):
        if activation_checkpointing:
            if activation_checkpointing == "offload":
                return offload_wrapper
            return partial(checkpoint_wrapper)
        return lambda x: x

    if isinstance(activation_checkpointing, dict):
        target_layer_cls = activation_checkpointing.get("module", "transformer")
        if target_layer_cls.lower() == "transformer":
            target_layer_cls = (
                "EncoderLayer",
                "ConformerEncoderLayer",
            )
        elif target_layer_cls.lower() == "attention":
            target_layer_cls = ("MultiHeadedAttention", "MultiHeadAttention")
        checkpointing_interval = activation_checkpointing.get("interval", 1)
        offloading = activation_checkpointing.get("offload", False)
        impl = (
            CheckpointImpl.REENTRANT
            if activation_checkpointing.get("reentrant", True)
            else CheckpointImpl.NO_REENTRANT
        )

        if idx % checkpointing_interval == 0 and layer_cls.__name__ in target_layer_cls:
            if offloading:
                return offload_wrapper
            return partial(checkpoint_wrapper, checkpoint_impl=impl)
        return lambda x: x

    raise ValueError("Invalid activation_checkpointing config")


def attn_checkpointing(activation_checkpointing: Union[str, Dict], i) -> Union[str, Dict]:
    """return activation checkpointing config for attention layer"""
    if isinstance(activation_checkpointing, str):
        return ""

    if isinstance(activation_checkpointing, dict):
        target_layer_cls = activation_checkpointing.get("module", "transformer")
        checkpointing_interval = activation_checkpointing.get("interval", 1)
        if target_layer_cls == "attention" and i % checkpointing_interval == 0:
            return activation_checkpointing
        return ""

    raise ValueError("Invalid activation_checkpointing config")

