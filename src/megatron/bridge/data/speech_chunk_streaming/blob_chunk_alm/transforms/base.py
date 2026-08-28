# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.
            {
                'image': image_tensor,
                'annotations': {}
            }

        Returns:
            dict: The result dict.
            {
                'image': image_tensor,
                'inputs': 
                    {
                        'input_ids': input_ids_tensor,
                        'attention_mask': attention_mask_tensor
                    }, # tokenized inputs
                'labels': label_tensor,
                'image_positions': image_positions_tensor,
            }
        """
