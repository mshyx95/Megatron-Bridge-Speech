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

"""Run independent Hindi VoiceAgentBench subsets on existing local speech servers."""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval_voiceagentbench_hindi import SUBSETS


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Launch and wait for one process per requested audio subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--subsets", choices=SUBSETS, nargs="+", required=True)
    args = parser.parse_args()
    if len(args.ports) != len(args.subsets) or len(set(args.ports)) != len(args.ports):
        parser.error("Specify one distinct server port for each subset")
    if any(not 1 <= port <= 65535 for port in args.ports):
        parser.error("Ports must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mirror_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("eval_voiceagentbench_hindi.py")

    def execute(subset: str, port: int) -> tuple[str, int]:
        log = args.output_dir / f"{subset}.log"
        command = [
            sys.executable,
            str(script),
            "--root",
            str(args.root),
            "--official-repo",
            str(args.official_repo),
            "--endpoint",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            args.model,
            "--subset",
            subset,
            "--output",
            str(args.output_dir / f"{subset}.json"),
            "--mirror",
            str(args.mirror_dir / f"{subset}.json"),
            "--resume",
        ]
        with log.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        LOGGER.info("%s finished with exit=%d (log=%s)", subset, completed.returncode, log)
        return subset, completed.returncode

    with ThreadPoolExecutor(max_workers=len(args.subsets)) as pool:
        results = list(pool.map(execute, args.subsets, args.ports))
    failures = [subset for subset, status in results if status]
    if failures:
        raise RuntimeError(f"Incomplete Hindi voice subsets: {', '.join(failures)}")
    LOGGER.info("Completed all %d Hindi voice subsets", len(results))


if __name__ == "__main__":
    main()
