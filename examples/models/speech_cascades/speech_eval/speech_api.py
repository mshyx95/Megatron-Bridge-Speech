# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI audio transport preserving the reference evaluation request contract."""

import base64
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def encode_audio(waveform, sample_rate):
    """Encode native waveform samples as a float32 WAV without resampling."""
    import numpy as np
    from scipy.io import wavfile

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim not in (1, 2) or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite native waveform")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Expected positive native sample rate")
    stream = io.BytesIO()
    wavfile.write(stream, int(sample_rate), audio)
    return stream.getvalue()


class SpeechAPI:
    """Generate via vLLM only, retaining exact prompts and per-request evidence."""

    def __init__(self, base_url, model, audit_path, *, timeout=300, transport=None, concurrency=None):
        endpoints = [base_url] if isinstance(base_url, str) else list(base_url)
        if not endpoints or len(endpoints) != len(set(endpoints)):
            raise ValueError("Expected distinct nonempty speech endpoints")
        for endpoint in endpoints:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Expected an HTTP endpoint without embedded credentials")
            if not endpoint.endswith("/v1") or parsed.query or parsed.fragment:
                raise ValueError("Speech endpoint must end in /v1")
        self.base_url = endpoints[0]
        self.concurrency = len(endpoints) if concurrency is None else concurrency
        if not isinstance(self.concurrency, int) or self.concurrency < len(endpoints):
            raise ValueError("Concurrency must cover every endpoint")
        self._endpoints = Queue()
        for index in range(self.concurrency):
            self._endpoints.put(endpoints[index % len(endpoints)])
        self._audit_lock = Lock()
        self.model = model
        self.audit_path = Path(audit_path)
        self.timeout = timeout
        self.transport = transport

    def _post(self, payload, *, base_url=None):
        headers = {"Content-Type": "application/json"}
        if os.environ.get("SPEECH_EVAL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["SPEECH_EVAL_API_KEY"]
        request = Request(
            (base_url or self.base_url) + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Speech API HTTP {error.code}; no local-model fallback was attempted") from None

    def generate(self, source, audio, sample_rate, audio_id, generation_config, eos_token):
        """Submit audio and its already-rendered prompt without changing token boundaries."""
        if not isinstance(source, str):
            raise ValueError("Native source must be a string")
        audios = audio if isinstance(audio, (list, tuple)) else [audio]
        rates = sample_rate if isinstance(sample_rate, (list, tuple)) else [sample_rate]
        if not audios or len(audios) != len(rates) or source.count("<audio>") != len(audios):
            raise ValueError("Native audio placeholders, waveforms and sample rates must match exactly")
        if (
            generation_config.do_sample
            or generation_config.num_beams != 1
            or generation_config.repetition_penalty != 1.0
            or generation_config.no_repeat_ngram_size != 0
        ):
            raise ValueError("Speech API requires the native greedy decoding contract")
        encoded = [encode_audio(waveform, rate) for waveform, rate in zip(audios, rates)]
        stops = list(dict.fromkeys([*(generation_config.stop_tokens or []), eos_token]))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": "wav"},
                        }
                        for data in encoded
                    ],
                }
            ],
            "chat_template": "{{ delivery_prompt }}",
            "chat_template_kwargs": {"delivery_prompt": source},
            "add_generation_prompt": False,
            "add_special_tokens": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "seed": 0,
            "max_tokens": generation_config.max_new_tokens,
            "stop": stops,
            "repetition_penalty": 1.0,
            "skip_special_tokens": True,
            "spaces_between_special_tokens": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "min_tokens": 0,
            "ignore_eos": False,
            "include_stop_str_in_output": False,
        }
        audit = {
            "audio_id": audio_id,
            "prompt": source,
            "prompt_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "audio_sha256": [hashlib.sha256(data).hexdigest() for data in encoded],
            "sample_rates": [int(rate) for rate in rates],
            "model": self.model,
            "status": "failed",
        }
        if getattr(generation_config, "bad_words", None):
            payload["bad_words"] = generation_config.bad_words
        endpoint = self._endpoints.get()
        audit["endpoint"] = endpoint
        try:
            response = self.transport(payload) if self.transport else self._post(payload, base_url=endpoint)
            choices = response.get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "length"}:
                raise ValueError("Speech API must return one completed text choice")
            message = choices[0]["message"]
            if message.get("tool_calls") or message.get("reasoning_content") or message.get("reasoning"):
                raise ValueError("Disable server reasoning/tool parsing to preserve raw evaluation output")
            answer = message.get("content")
            if not isinstance(answer, str):
                raise ValueError("Speech API returned no text content")
            audit.update(
                status="ok",
                finish_reason=choices[0]["finish_reason"],
                response_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                usage=response.get("usage"),
            )
            return answer
        finally:
            self._endpoints.put(endpoint)
            with self._audit_lock:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a") as stream:
                    stream.write(json.dumps(audit, ensure_ascii=True) + "\n")


class SpeechAPIPipeline:
    """Native batch adapter that never constructs or loads a local model."""

    use_mtp_speculative = False

    def __init__(self, model, tokenizer, *, client, **kwargs):
        self.client = client
        self.eos_token = tokenizer.eos_token
        if not self.eos_token or kwargs.get("use_mtp") or kwargs.get("use_mtp_speculative"):
            raise ValueError("Native EOS is required; server owns speculative decoding")

    def __call__(self, inputs, *, generation_config, **kwargs):
        size = len(inputs["source"])
        if any(key not in inputs or len(inputs[key]) != size for key in ("source", "audio", "fs", "audio_id")):
            raise ValueError("Expected aligned native input lists")
        output = {"audio_ids": list(inputs["audio_id"]), "generated_texts": []}

        def generate(index):
            return self.client.generate(
                inputs["source"][index],
                inputs["audio"][index],
                inputs["fs"][index],
                inputs["audio_id"][index],
                generation_config,
                self.eos_token,
            )

        with ThreadPoolExecutor(max_workers=self.client.concurrency) as workers:
            output["generated_texts"] = list(workers.map(generate, range(size)))
        for source_key, target_key in (("label", "label"), ("segment", "segments")):
            if source_key in inputs:
                if len(inputs[source_key]) != size:
                    raise ValueError("Misaligned native metadata")
                output[target_key] = list(inputs[source_key])
        return output
