# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.qwen.checkpoint_inventory import inventory_qwen35_checkpoint


def _load_text_config(model_path: Path) -> Mapping[str, object]:
    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config.get("text_config", config)


def main() -> int:
    """Print a JSON inventory of a dense Qwen3.5 HF checkpoint; exit non-zero if invalid."""
    parser = argparse.ArgumentParser(description="Inventory dense Qwen3.5 HF checkpoint mappings")
    parser.add_argument("--hf-model-path", type=Path, required=True)
    args = parser.parse_args()

    config = _load_text_config(args.hf_model_path)
    tensor_keys = SafeTensorsStateSource(args.hf_model_path).get_all_keys()
    inventory = inventory_qwen35_checkpoint(config, tensor_keys)
    print(json.dumps({**asdict(inventory), "valid": inventory.valid}, indent=2))
    return 0 if inventory.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())