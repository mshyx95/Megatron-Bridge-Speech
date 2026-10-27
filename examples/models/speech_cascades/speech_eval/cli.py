# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluates a model.")

    parser.add_argument(
        "pre_trained_model_path",
        type=str,
        help="Pre-trained model name or path.",
    )

    parser.add_argument(
        "-t",
        "--pretrained_tokenizer_name_or_path",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="Pre-trained tokenizer name or path.",
    )

    parser.add_argument(
        "-tm",
        "--use_text_model",
        type=str,
        default="false",
        help="whether use text model only",
    )
    args, extra_args = parser.parse_known_args()

    parser.add_argument(
        "-a",
        "--audio_processor_name_or_path",
        type=str,
        default=None,
        help="Pre-trained tokenizer name or path.",
    )

    parser.add_argument(
        "-dm",
        "--device_map",
        type=str,
        default="cuda",
        choices=["auto", "balanced", "balanced_low_0", "cpu", "cuda", "sequential"],
        help="Device map to use for the model.",
    )

    parser.add_argument(
        "-th",
        "--token_healing",
        action="store_true",
        help="Use token healing.",
    )

    parser.add_argument(
        "-tn",
        "--task_name",
        type=str,
        default="human_eval",
        help="Name of the evaluation task.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "-st",
        "--sub_task",
        type=str,
        default="wer",
        help="sub task in wavllm",
    )
    parser.add_argument(
        "-nb",
        "--num_beams",
        type=int,
        default="1",
        help="num_beams for LLM generation",
    )
    parser.add_argument(
        "-tp",
        "--temperature",
        type=float,
        default="1.0",
        help="temperature for LLM generation",
    )
    parser.add_argument(
        "-nrns",
        "--no_repeat_ngram_size",
        type=int,
        default="0",
        help="no repeat n gram for LLM generation",
    )
    parser.add_argument(
        "-rp",
        "--repetition_penalty",
        type=float,
        default="1.0",
        help="repetition penalty for LLM generation",
    )
    parser.add_argument(
        "-ct",
        "--chat_template",
        type=str,
        default="phi_3",
        help="chat template",
    )
    parser.add_argument(
        "-if",
        "--instruction_first",
        type=str,
        default="false",
        help="whether to use instruction first evaluation, true=True, false=False",
    )
    parser.add_argument(
        "--bias_after_audio",
        type=str,
        default="false",
        help="whether to place biasing prompt after <audio> tag, true=True, false=False",
    )

    parser.add_argument(
        "-dp",
        "--dataset_path",
        type=str,
        help="path to the dataset file",
    )
    parser.add_argument(
        "-op",
        "--output_file_path",
        type=str,
        default="output.txt",
        help="path to the output file.",
    )
    parser.add_argument(
        "-pi",
        "--pad_features_inside",
        action="store_true",
        help="For whisper-p-all, pad to 30s for encoder forward and LLM attend to all frames",
    )
    parser.add_argument(
        "-po",
        "--pad_features_outside",
        action="store_true",
        help="For whisper-p-real, pad to 30s for encoder forward and LLM attend to real frames only",
    )
    (
        parser.add_argument(
            "-lid",
            "--use_lid",
            type=str,
            default="0",
            help="use_lid for ASR decoding, lid_mode (0,1,2), 0: no lid, 1: predict lid, 2: append lid to prompt",
        ),
    )
    parser.add_argument(
        "-ap",
        "--asr_prompt",
        type=str,
        default="default",
        help="ASR prompt mode: default, lfg3/asr-lfg3, verbatim/asr-verbatim, lexical/asr-lexical",
    )
    parser.add_argument(
        "-mt",
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "-mtp",
        "--use_mtp",
        type=bool,
        default=False,
        help="Whether to use MTP assistant.",
    )
    parser.add_argument(
        "-mtp_spec",
        "--use_mtp_speculative",
        type=bool,
        default=False,
        help="Whether to use MTP speculative decoding.",
    )
    parser.add_argument(
        "-bl",
        "--biasing_list_file",
        type=str,
        default=None,
        help="the list of biasing words for biasing decoding, separated by commas",
    )
    (
        parser.add_argument(
            "-cs",
            "--chunk_size",
            type=int,
            default=-1,
            help="Chunk size for Streaming decoding.",
        ),
    )
    (
        parser.add_argument(
            "-cdp",
            "--codec_decoder_path",
            type=str,
            default=None,
            help="codec_decoder_path for TTS decoding",
        ),
    )
    parser.add_argument(
        "-ow",
        "--output_wavdir",
        type=str,
        default=None,
        help="output wavdir for TTS decoding",
    )
    args, extra_args = parser.parse_known_args()

    return args, extra_args
