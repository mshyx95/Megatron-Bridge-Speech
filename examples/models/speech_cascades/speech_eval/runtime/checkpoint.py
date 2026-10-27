"""Read serving metadata without importing the model or loading tensor payloads."""

import json
import math
import re
import struct
from pathlib import Path


class CompatibilityError(ValueError):
    """The checkpoint or requested runtime does not satisfy the serving contract."""


def positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise CompatibilityError(f"{field} must be a positive integer")
    return value


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CompatibilityError(f"Cannot read JSON object: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CompatibilityError(f"Expected JSON object: {path}")
    return value


def exported_metadata(checkpoint: Path) -> tuple[dict, dict]:
    config = read_object(checkpoint / "config.json")
    if config.get("architectures") != ["Qwen3_5AudioForCausalLM"]:
        raise CompatibilityError("Expected exported architecture Qwen3_5AudioForCausalLM")
    if config.get("model_type") not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise CompatibilityError("Expected flat qwen3_5_text or qwen3_5_moe_text config from the speech exporter")
    embedding = config.get("embd_layer")
    if not isinstance(embedding, dict) or embedding.get("embedding_cls") != "audio":
        raise CompatibilityError("embd_layer.embedding_cls must be audio")
    index = read_object(checkpoint / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CompatibilityError("Export requires a nonempty safetensors weight_map")
    for name, filename in weight_map.items():
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise CompatibilityError(f"Invalid shard filename for {name}")
    return config, weight_map


def inspect_mtp(config: dict, weight_map: dict) -> dict:
    configured = config.get("mtp_num_hidden_layers", 0)
    if type(configured) is not int or configured not in (0, 1, 4):
        raise CompatibilityError("mtp_num_hidden_layers must be 0, 1, or 4")
    names = [name for name in weight_map if name.startswith("mtp.")]
    layer_ids = set()
    for name in names:
        if name.startswith("mtp.layers."):
            match = re.match(r"mtp\.layers\.(\d+)\.", name)
            if match is None:
                raise CompatibilityError(f"Malformed MTP layer tensor: {name}")
            layer_ids.add(int(match.group(1)))
    if names and (not configured or layer_ids != set(range(configured))):
        raise CompatibilityError("MTP weights disagree with configured physical layer count")
    for layer_id in layer_ids:
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.down_proj.weight",
        ):
            if f"mtp.layers.{layer_id}.{suffix}" not in weight_map:
                raise CompatibilityError(f"Incomplete MTP layer {layer_id}: missing {suffix}")
    retained = len(layer_ids)
    return {
        "configured_physical_layers": configured,
        "retained_physical_layers": retained,
        "variant": {0: "standard", 1: "NativeMTP1", 4: "ExpandedMTP4"}[retained],
        "stripped_by_export": configured > 0 and not names,
    }


def read_shapes(checkpoint: Path, weight_map: dict) -> dict:
    shapes = {}
    for filename in sorted(set(weight_map.values())):
        path = checkpoint / filename
        try:
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise CompatibilityError(f"Truncated safetensors header: {filename}")
                length = struct.unpack("<Q", prefix)[0]
                if not 2 <= length <= 100_000_000:
                    raise CompatibilityError(f"Invalid safetensors header length: {filename}")
                header = json.loads(stream.read(length))
                payload_size = path.stat().st_size - 8 - length
        except (OSError, ValueError) as error:
            raise CompatibilityError(f"Cannot read safetensors header {filename}: {error}") from error
        if not isinstance(header, dict):
            raise CompatibilityError(f"Invalid safetensors header: {filename}")
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in shapes or weight_map.get(name) != filename:
                raise CompatibilityError(f"Header/index mismatch or duplicate tensor: {name}")
            if not isinstance(tensor, dict):
                raise CompatibilityError(f"Invalid tensor metadata: {name}")
            shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
            if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
                raise CompatibilityError(f"Invalid tensor shape: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                raise CompatibilityError(f"Invalid tensor payload offsets: {name}")
            item_size = {
                "BF16": 2,
                "F16": 2,
                "F32": 4,
                "F64": 8,
                "I64": 8,
                "I32": 4,
                "I16": 2,
                "I8": 1,
                "U8": 1,
                "BOOL": 1,
            }.get(tensor.get("dtype"))
            if item_size is None or offsets[1] - offsets[0] != math.prod(shape) * item_size:
                raise CompatibilityError(f"Unsupported dtype or tensor byte size mismatch: {name}")
            shapes[name] = shape
    if set(shapes) != set(weight_map):
        raise CompatibilityError("Index tensors are missing from shard headers")
    return shapes


def check_tensor_group(prefix: str, expected: dict, weight_map: dict, shapes: dict | None) -> None:
    actual = {name for name in weight_map if name.startswith(prefix)}
    if actual != set(expected):
        raise CompatibilityError(
            f"{prefix} tensor names mismatch; missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    if shapes is not None:
        for name, expected_shape in expected.items():
            if shapes[name] != expected_shape:
                raise CompatibilityError(f"Shape mismatch: {name}: expected {expected_shape}, found {shapes[name]}")


def mtp_shapes(config: dict, depth: int) -> dict:
    hidden = positive_integer(config.get("hidden_size"), "hidden_size")
    heads = positive_integer(config.get("num_attention_heads"), "num_attention_heads")
    kv_heads = positive_integer(config.get("num_key_value_heads"), "num_key_value_heads")
    head_dim = positive_integer(config.get("head_dim"), "head_dim")
    intermediate = positive_integer(config.get("intermediate_size"), "intermediate_size")
    if heads % kv_heads:
        raise CompatibilityError("num_attention_heads must be divisible by num_key_value_heads")
    if config.get("mtp_use_dedicated_embeddings", False) or config.get("attention_bias", False):
        raise CompatibilityError("MTP requires shared embeddings/head and bias-free attention")
    if config.get("attn_output_gate", True) is not True or config.get("hidden_act", "silu") != "silu":
        raise CompatibilityError("Existing MTP layers require gated attention and SiLU MLP")
    if depth == 4:
        factor = config.get("partial_rotary_factor")
        epsilon = config.get("rms_norm_eps")
        if (
            type(factor) not in (int, float)
            or not 0 < factor <= 1
            or int(head_dim * factor) % 2
            or int(head_dim * factor) == 0
        ):
            raise CompatibilityError(
                "ExpandedMTP4 requires a valid partial_rotary_factor with positive even rotary dimension"
            )
        if (
            type(epsilon) not in (int, float)
            or not 0 < epsilon < 1
            or not isinstance(config.get("rope_parameters"), dict)
        ):
            raise CompatibilityError("ExpandedMTP4 requires rms_norm_eps and rope_parameters metadata")
    expected = {"mtp.fc.weight": [hidden, 2 * hidden]}
    for suffix in ("norm.weight", "pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight"):
        expected[f"mtp.{suffix}"] = [hidden]
    layer = {
        "self_attn.q_proj.weight": [2 * heads * head_dim, hidden],
        "self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
        "self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
        "self_attn.o_proj.weight": [hidden, heads * head_dim],
        "self_attn.q_norm.weight": [head_dim],
        "self_attn.k_norm.weight": [head_dim],
        "mlp.gate_proj.weight": [intermediate, hidden],
        "mlp.up_proj.weight": [intermediate, hidden],
        "mlp.down_proj.weight": [hidden, intermediate],
        "input_layernorm.weight": [hidden],
        "post_attention_layernorm.weight": [hidden],
    }
    for layer_id in range(depth):
        expected.update({f"mtp.layers.{layer_id}.{suffix}": shape for suffix, shape in layer.items()})
    return expected


def inspect_checkpoint(checkpoint: Path, *, index_only: bool = False) -> dict:
    checkpoint = checkpoint.resolve()
    config, weight_map = exported_metadata(checkpoint)
    hidden = positive_integer(config.get("hidden_size"), "hidden_size")
    layers = positive_integer(config.get("num_hidden_layers"), "num_hidden_layers")
    if config.get("quantization_config"):
        raise CompatibilityError("Quantized exports are outside the existing speech export contract")
    if config.get("n_embd", hidden) != hidden:
        raise CompatibilityError("n_embd and hidden_size disagree for the inherited audio embedding")
    processor = config.get("audio_processor")
    if not isinstance(processor, dict) or processor.get("name") != "cascades":
        raise CompatibilityError("audio_processor.name must be cascades for the existing adapter")
    encoder = processor.get("config")
    if not isinstance(encoder, dict):
        raise CompatibilityError("audio_processor.config must contain encoder structure")
    if encoder.get("input_size") != 80:
        raise CompatibilityError("The speech adapter extracts 80-bin filterbank features")
    blocks = positive_integer(encoder.get("num_blocks"), "audio_processor.config.num_blocks")
    audio_dim = positive_integer(encoder.get("attention_dim"), "audio_processor.config.attention_dim")
    time_reduction = positive_integer(encoder.get("time_reduction", 1), "time_reduction")
    embedding = config["embd_layer"]
    for option in (
        "use_qformer",
        "use_conv_downsample",
        "use_conv_temporal_downsample",
        "use_conv_bottleneck_downsample",
        "external_adapter",
        "zero_audio_projection",
    ):
        if embedding.get(option):
            raise CompatibilityError(f"Unsupported serving embedding transform: {option}")
    if embedding.get("ctc_weight", 0):
        raise CompatibilityError("CTC embedding heads are outside this serving contract")
    rate = positive_integer(embedding.get("downsample_rate", 1), "embd_layer.downsample_rate")
    input_dim = audio_dim * rate
    prefix = "model.embed_tokens_extend.audio_projection."
    projection = embedding.get("projection_cls")
    if projection == "lfg3":
        middle = positive_integer(embedding.get("projector_hidden", 4096), "embd_layer.projector_hidden")
        expected = {
            prefix + "norm.weight": [input_dim],
            prefix + "linear1.weight": [middle, input_dim],
            prefix + "linear2.weight": [hidden, middle],
        }
        public_name = "GatedProj"
        formula = "input_dim + input_dim * projector_hidden + projector_hidden * hidden_size"
    elif projection == "mlp":
        middle = hidden
        expected = {
            prefix + "0.weight": [hidden, input_dim],
            prefix + "0.bias": [hidden],
            prefix + "2.weight": [hidden, hidden],
            prefix + "2.bias": [hidden],
        }
        public_name = "MLP"
        formula = "input_dim * hidden_size + hidden_size * hidden_size + 2 * hidden_size"
    else:
        raise CompatibilityError("Supported projector metadata: projection_cls=lfg3 (GatedProj) or mlp (MLP)")
    for pattern, count, label in (
        (r"model\.embed_tokens_extend\.encoder\.encoders\.(\d+)\.", blocks, "encoder blocks"),
        (r"model\.layers\.(\d+)\.", layers, "backbone layers"),
    ):
        actual = {int(match.group(1)) for name in weight_map if (match := re.match(pattern, name))}
        if actual != set(range(count)):
            raise CompatibilityError(f"Index {label} disagree with config: expected {count}, found {sorted(actual)}")
    shapes = None if index_only else read_shapes(checkpoint, weight_map)
    for block in range(blocks):
        name = f"model.embed_tokens_extend.encoder.encoders.{block}.layer_norm.weight"
        if name not in weight_map or shapes is not None and shapes[name] != [audio_dim]:
            raise CompatibilityError(f"Encoder normalization shape or tensor missing: {name}; expected [{audio_dim}]")
    check_tensor_group(prefix, expected, weight_map, shapes)
    allowed_audio_prefixes = (prefix, "model.embed_tokens_extend.encoder.")
    unexpected_audio = [
        name
        for name in weight_map
        if name.startswith("model.embed_tokens_extend.") and not name.startswith(allowed_audio_prefixes)
    ]
    if unexpected_audio:
        raise CompatibilityError(f"Unsupported audio tensors: {unexpected_audio}")
    mtp = inspect_mtp(config, weight_map)
    if mtp["retained_physical_layers"]:
        check_tensor_group("mtp.", mtp_shapes(config, mtp["retained_physical_layers"]), weight_map, shapes)
    warnings = ["CPU metadata validation is not evidence of GPU startup, audio quality, or numerical parity."]
    if index_only:
        warnings.append("Index-only inspection: shard availability and tensor shapes have NOT been validated.")
    if mtp["stripped_by_export"]:
        warnings.append("MTP weights were stripped; configured training depth does not enable speculation.")
    return {
        "checkpoint": str(checkpoint),
        "architecture": config["architectures"][0],
        "backbone": {"model_type": config["model_type"], "hidden_size": hidden, "num_hidden_layers": layers},
        "encoder": {
            "implementation": "cascades",
            "num_blocks": blocks,
            "attention_dim": audio_dim,
            "input_size": 80,
            "time_reduction": time_reduction,
            "structural_profile": {(23, 1024, 80, 8): "Checkpoint22L", (24, 1024, 80, 8): "Phi4MM"}.get(
                (blocks, audio_dim, encoder["input_size"], time_reduction)
            ),
            "provenance": "unknown; structural profile does not establish source weight identity",
        },
        "projector": {
            "name": public_name,
            "parameters": sum(math.prod(shape) for shape in expected.values()),
            "count_evidence": "config_formula" if index_only else "safetensors_shapes_checked_against_config",
            "formula": formula,
            "input_dim": input_dim,
            "hidden_dim": middle,
            "output_dim": hidden,
            "downsample_rate": rate,
        },
        "mtp": mtp,
        "validation": "index_only" if index_only else "shard_headers",
        "warnings": warnings,
    }
