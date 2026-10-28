# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import natsort


def find_checkpoints(
    checkpoint_dir: str,
    checkpoint_regex: Optional[Union[str, re.Pattern]] = "",
    n_checkpoints: Optional[int] = None,
    reverse: bool = True,
) -> List[str]:
    """Find available checkpoints in a given directory.

    This function finds available checkpoints in a given directory by looking
    for folders that match a regular expression for checkpoint names.

    Args:
        checkpoint_dir: Path to the directory that might contain checkpoints.
        checkpoint_regex: Regular expression to match the checkpoint folder name.
        n_checkpoints: Number of checkpoints to return.
            If ``None``, all checkpoints are returned.
        reverse: Whether to return the checkpoints in reverse order.

    """

    if not os.path.isdir(checkpoint_dir):
        return []

    content_dir = os.listdir(checkpoint_dir)
    checkpoints = [
        os.path.join(checkpoint_dir, path)
        for path in content_dir
        if re.search(checkpoint_regex, path) is not None and os.path.isdir(os.path.join(checkpoint_dir, path))
    ]
    checkpoints = natsort.natsorted(checkpoints, reverse=reverse)

    # If no sub-directories are found, assume that `checkpoint_dir` is the checkpoint
    if len(checkpoints) == 0:
        checkpoints = [checkpoint_dir]

    if n_checkpoints is not None:
        checkpoints = checkpoints[:n_checkpoints]

    return checkpoints


def get_basename(path: Union[str, os.PathLike], n_levels: int = 1) -> str:
    """Get the basename of a file or folder.

    Args:
        path: Path to the file or folder.
        n_levels: Number of levels to include in the basename.

    Returns:
        Basename of the file or folder.

    """

    path_parts = Path(path).parts

    if n_levels > len(path_parts):
        return path

    if n_levels <= 0:
        return ""

    return os.path.join(*path_parts[-n_levels:])


def get_checkpoints_info(
    model_name_or_path: str,
    checkpoint_regex: Optional[Union[str, re.Pattern]] = "",
    n_checkpoints: Optional[int] = None,
    reverse: bool = True,
) -> List[Dict[str, Union[str, int]]]:
    """Get information about available checkpoints in a given directory.

    Args:
        checkpoint_dir: Path to the directory that might contain checkpoints.
        checkpoint_regex: Regular expression to match the checkpoint folder name.
        n_checkpoints: Number of checkpoints to return.
            If ``None``, all checkpoints are returned.
        reverse: Whether to return the checkpoints in reverse order.

    Returns:
        List of dictionaries with checkpoint information.

    """

    if not os.path.isdir(model_name_or_path):
        return [{"checkpoint_path": model_name_or_path, "output_dir": "", "step": 0}]

    checkpoints = find_checkpoints(
        model_name_or_path,
        checkpoint_regex=checkpoint_regex,
        n_checkpoints=n_checkpoints,
        reverse=reverse,
    )

    checkpoints_info = []
    for checkpoint in checkpoints:
        checkpoint_basename = get_basename(checkpoint, n_levels=1)

        if checkpoint_basename.isdigit():
            step = int(checkpoint_basename)
        elif checkpoint_basename.split("-")[-1].isdigit():
            step = checkpoint_basename.split("-")[-1]
        elif checkpoint_basename.split("-")[-1] == "last":
            step = sys.maxsize
        else:
            step = 0

        checkpoints_info.append({"checkpoint_path": checkpoint, "output_dir": checkpoint_basename, "step": step})

    return checkpoints_info


def get_full_path(path: str, create_folder: bool = False) -> str:
    """Get the full path to a file or folder.

    Args:
        path: The path to the file or folder.
        create_folder: Whether to create the folder if it does not exist.

    Returns:
        The full path to the file or folder.

    """

    assert path, "`path` can not be empty."
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    if create_folder:
        os.makedirs(path, exist_ok=True)

    return path


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file (optionally compressed with ``gzip``).

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary-based contents of the file.

    """

    if str(file_path).endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            return json.load(f)

    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json_file(
    obj: Dict[str, Any], file_path: str, create_folder: bool = False, indent: int = 4, mode: str = "w", **kwargs
) -> None:
    """Save an object to a JSON file (optionally compressed with ``gzip``).

    Args:
        obj: Object to save.
        file_path: Path to the file.
        create_folder: Whether to create the folder if it does not exist.
        indent: Number of spaces to indent.
        mode: Mode to open the file.

    """

    file_path = get_full_path(file_path, create_folder=create_folder)

    if str(file_path).endswith(".gz"):
        assert mode == "wt", "`mode` must be `wt` when saving to a compressed file."

        with gzip.open(file_path, mode) as f:
            return json.dump(obj, f, indent=indent, **kwargs)

    with open(file_path, mode, encoding="utf8") as f:
        return json.dump(obj, f, indent=indent, **kwargs)


def save_jsonl_file(obj: List[Dict[str, Any]], file_path: str, create_folder: bool = False, mode: str = "w") -> None:
    """Save a list of objects to a JSONL file (optionally compressed with ``gzip``).

    Args:
        obj: List of objects to save.
        file_path: Path to the file.
        create_folder: Whether to create the folder if it does not exist.
        mode: Mode to open the file.

    """

    file_path = get_full_path(file_path, create_folder=create_folder)

    if str(file_path).endswith(".gz"):
        assert mode == "wt", "`mode` must be `wt` when saving to a compressed file."

        with gzip.open(file_path, mode) as f:
            for o in obj:
                f.write(json.dumps(o) + "\n")
        return

    with open(file_path, mode, encoding="utf8") as f:
        for o in obj:
            f.write(json.dumps(o) + "\n")


def load_text_file(file_path: str) -> str:
    """Load a text file (optionally compressed with ``gzip``).

    Args:
        file_path: Path to the file.

    Returns:
        Contents of the file.

    """

    if str(file_path).endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            return f.read()

    with open(file_path, "r", encoding="utf8") as f:
        return f.read()


def save_text_file(text: str, file_path: str, create_folder: bool = False, mode: str = "w") -> None:
    """Save a string to a text file (optionally compressed with ``gzip``).

    Args:
        text: String to save.
        file_path: Path to the file.
        create_folder: Whether to create the folder if it does not exist.
        mode: Mode to open the file.

    """

    file_path = get_full_path(file_path, create_folder=create_folder)

    if str(file_path).endswith(".gz"):
        assert mode == "wt", "`mode` must be `wt` when saving to a compressed file."

        with gzip.open(file_path, mode) as f:
            return f.write(text)

    with open(file_path, mode, encoding="utf8") as f:
        return f.write(text)


def validate_file_extension(file_path: Union[str, Path], extensions: Union[str, List[str]]) -> bool:
    """Validate whether a file has a valid extension.

    Args:
        file_path: Path to the file.
        extensions: Valid extension(s).

    Returns:
        Whether the file has a valid extension.

    """

    if not isinstance(file_path, (str, Path)):
        return False

    file_path = Path(file_path)
    extensions = extensions if isinstance(extensions, list) else [extensions]

    if file_path.suffix not in extensions:
        return False

    return True
