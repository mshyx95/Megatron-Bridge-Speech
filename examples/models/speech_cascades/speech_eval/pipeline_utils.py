# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from torch.utils.data import Dataset


def example_generator(
    dataset: Dataset,
    mapping_fn: Optional[Callable[[Any], Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    **kwargs,
) -> Iterator[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Generate and optionally map examples over a dataset.

    Additional keyword arguments are passed to the ``mapping_fn``.

    Args:
        dataset: Dataset to generate examples from.
        mapping_fn: Mapping function to apply to each example.

    Yields:
        Iterator over examples.

    """

    for example in dataset:
        if mapping_fn is None:
            yield example
        else:
            yield mapping_fn(example, **kwargs)


def batch_example_generator(
    dataset: Dataset,
    batch_size: int,
    mapping_fn: Optional[Callable[[Any], Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    **kwargs,
) -> Iterator[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Generate and optionally map examples over a dataset. -> to a batch

    Additional keyword arguments are passed to the ``mapping_fn``.

    Args:
        dataset: Dataset to generate examples from.
        mapping_fn: Mapping function to apply to each example.

    Yields:
        Iterator over examples.

    """

    batch = {}

    for example in dataset:
        if isinstance(example, dict):
            example = mapping_fn(example, **kwargs) if mapping_fn is not None else example
            for key in example:
                batch[key] = batch.get(key, []) + [example[key]]
            assert "source" in batch, "source is required"
            if len(batch["source"]) == batch_size:
                yield batch
                batch = {}

        elif isinstance(example, list):
            # for multi-turn/ASR datasets with segments
            for ex in example:
                ex = mapping_fn(ex, **kwargs) if mapping_fn is not None else ex

                for key in ex:
                    batch[key] = batch.get(key, []) + [ex[key]]

                if batch_size == len(batch["source"]):
                    yield batch
                    batch = {}

            if len(batch) > 0:
                yield batch
                batch = {}

        else:
            raise ValueError(f"Unsupported example type: {type(example)}")

    if len(batch) > 0:
        yield batch
