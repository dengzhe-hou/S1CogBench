"""P2 frozen generation control on complete original visible states.

This module changes the response interface only. Task loops own feedback,
original seeds, invalid-action handling and scoring. There is no answer retry,
reward imputation, model fitting, history truncation or paid API.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import platform
import re
import sys
import time

MAX_NEW_TOKENS = 1024
INSTRUCTION = (
    "Reason about the decision using only the information shown above. "
    "You may write reasoning before the answer. "
    "End your response with exactly one final line in this format: FINAL: X\n"
    "Replace X with one of the listed legal option letters. "
    "Do not put any text after that line."
)


def parse_answer(text, output_keys, truncated):
    """Strict documented final-line parser; no guessed/first-letter answer."""
    matches = re.findall(r"^\s*FINAL:\s*([A-Z])\s*$", text.strip(), flags=re.MULTILINE)
    choice = matches[0] if len(matches) == 1 and matches[0] in output_keys else None
    if choice is not None and not re.search(
        r"FINAL:\s*" + re.escape(choice) + r"\s*$", text.strip()
    ):
        choice = None
    return {
        "choice": None if truncated else choice,
        "parsed_choice_before_truncation_check": choice,
        "invalid": bool(truncated or choice is None),
        "invalid_reason": "max_new_tokens_reached" if truncated else
            ("missing_ambiguous_or_illegal_final_line" if choice is None else None),
    }


def content_for(state, question, output_keys):
    if len(output_keys) != len(set(output_keys)) or set(output_keys) != set(question["criteria"]):
        raise ValueError("Question criteria must exactly match distinct legal output keys.")
    if not all(isinstance(k, str) and len(k) == 1 and k.isupper() for k in output_keys):
        raise ValueError("P2 Horizon/WCST requires original single-letter action keys.")
    options = "\n".join(f"{k}: {question['criteria'][k]}" for k in output_keys)
    return state + "\n" + question["instructions"] + "\n" + options + "\n" + INSTRUCTION


LLAMA_HORIZON_RESPONSE = "Respond with **only** one character: 'I' or 'H'. No punctuation, no quotes, no explanation."
LLAMA_WCST_RESPONSE = "and reply accordingly with only one key card: 'A' or 'B' or 'C' or 'D', and nothing else!"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


def reasoning_raw_prompt(prompt):
    """Alter only the source output instruction, preserving original Llama headers."""
    if LLAMA_HORIZON_RESPONSE in prompt:
        return prompt.replace(LLAMA_HORIZON_RESPONSE, INSTRUCTION, 1)
    if LLAMA_WCST_RESPONSE in prompt:
        result = prompt.replace(LLAMA_WCST_RESPONSE, "and select one of the original key cards.", 1)
        header = result.rfind(ASSISTANT_HEADER)
        if header < 0:
            raise ValueError("Expected original Llama assistant header.")
        marker = result.rfind("<|eot_id|>", 0, header)
        if marker < 0:
            raise ValueError("Expected original WCST user-message end marker.")
        return result[:marker] + "\n" + INSTRUCTION + result[marker:]
    raise ValueError("Raw generation prompt must match original Horizon or WCST Llama output instruction.")


class GenerationAgent:
    """Greedy BF16 text generation, max 1024 new tokens, native EOS respected."""
    def __init__(self, model_name, device="cuda:0", root=None):
        import torch
        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("P2 neural runs require an admitted CUDA allocation.")
        self.root = Path(root or os.environ.get(
            "S1_PROBE_ROOT", str(Path(__file__).resolve().parents[1] / "root")
        )).resolve()
        sys.path.insert(0, str(self.root))
        started = time.perf_counter()
        if model_name == "llama":
            from src.hf_choice_models import FrozenDecoderChoice
            base = FrozenDecoderChoice(
                self.root / "models/llama-3.1-8b-instruct", device, "instruct_chat"
            )
            self.context_limit = int(base.model.config.max_position_embeddings)
        elif model_name == "qwen":
            from examples.qwen35_4b_choice_agent import Qwen35Choice
            base = Qwen35Choice(device)
            self.context_limit = int(base.max_context_tokens)
        else:
            raise ValueError("P2 model must be llama or qwen.")
        self.model = base.model
        self.tokenizer = base.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - started
        import transformers
        self.metadata = {
            "experiment": "P2_generation_control",
            "model": model_name,
            "dtype": sorted({str(p.dtype) for p in self.model.parameters()}),
            "model_directory": str((self.root / (
                "models/llama-3.1-8b-instruct" if model_name == "llama" else "models/qwen3.5-4b"
            )).resolve()),
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(self.device), "uuid", "unavailable")),
            "host": platform.node(),
            "linear_attention_kernel_modules": sorted({layer.chunk_gated_delta_rule.__module__
                for layer in self.model.modules() if hasattr(layer, "chunk_gated_delta_rule")}),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "add_special_tokens": model_name == "llama",
            "tokenization_relation": "Llama original raw-prompt tokenizer defaults include BOS postprocessing; Qwen typed adapter uses no additional special tokens.",
            "use_cache": True,
            "qwen_enable_thinking": model_name == "qwen",
            "instruction": INSTRUCTION,
            "parser": "Exactly one legal FINAL: X line at end; no fallback or retry.",
            "truncation": "A generated response reaching 1024 new tokens is marked invalid even if a final line is present.",
            "history": "Complete provided original visible state; no truncation or summary.",
            "invalid_transition": "Task runner must retain record and mark affected trajectory incomplete; never invent an action.",
            "probabilities": "No action probabilities or human NLL inferred from generated text.",
            "timing": "CUDA synchronized adapter generation, including tokenization and text parsing.",
            "generation_config": self.model.generation_config.to_dict(),
            "load_seconds": self.load_seconds,
            "base_adapter_metadata": base.metadata,
        }

    def _prompt(self, state, question, output_keys):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.model_name == "qwen":
            kwargs["enable_thinking"] = True
        if isinstance(state, list):
            messages = copy.deepcopy(state)
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("Original WCST message history must end in the current user stimulus.")
            for message in messages:
                if isinstance(message.get("content"), str):
                    message["content"] = message["content"].replace(
                        LLAMA_WCST_RESPONSE, "and select one of the original key cards."
                    )
            messages[-1]["content"] += "\n" + INSTRUCTION
        else:
            messages = [{"role": "user", "content": content_for(state, question, output_keys)}]
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def decide(self, state, question, output_keys):
        return self.decide_batch([(state, question, output_keys)])[0]

    def decide_messages(self, messages, question, output_keys):
        return self.decide(messages, question, output_keys)

    def decide_raw_prompt(self, prompt, output_keys):
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        if self.model_name != "llama":
            raise ValueError("Original manually formatted raw chat is used only for Llama.")
        prepared = reasoning_raw_prompt(prompt)
        return self._decide_rendered(
            [(prompt, {"instructions": "Original source response instruction replaced by reasoning and final line.", "criteria": dict.fromkeys(output_keys, "Original legal action")}, output_keys)],
            [prepared], started=started,
        )[0]

    def decide_batch(self, requests):
        """Return per-request raw records; batch size is disclosed, never hidden."""
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prompts = [self._prompt(state, question, keys) for state, question, keys in requests]
        return self._decide_rendered(requests, prompts, started=started)

    def _decide_rendered(self, requests, prompts, started=None):
        torch = self.torch
        if started is None:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=(self.model_name == "llama"), truncation=False)
        lengths = inputs["attention_mask"].sum(-1).tolist()
        if any(int(n) + MAX_NEW_TOKENS > self.context_limit for n in lengths):
            raise ValueError("Complete input plus generation allowance exceeds context; no truncation.")
        inputs = inputs.to(self.device)
        prefix_width = inputs["input_ids"].shape[1]
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                use_cache=True, return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - started
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
        records = []
        for i, (state, question, keys) in enumerate(requests):
            tokens = generated.sequences[i, prefix_width:].tolist()
            stop = next((j + 1 for j, token in enumerate(tokens) if token in eos), len(tokens))
            tokens = tokens[:stop]
            raw_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            truncated = len(tokens) >= MAX_NEW_TOKENS
            parsed = parse_answer(raw_text, keys, truncated)
            records.append({
                "state": state, "question": question, "output_keys": list(keys),
                "model_prompt": prompts[i], "raw_text": raw_text,
                "generated_token_ids": tokens, "input_tokens": int(lengths[i]),
                "output_tokens": len(tokens), "truncated": truncated,
                "generation_batch_size": len(requests),
                "generation_seconds": generation_seconds,
                "generation_timing_kind": "synchronized_batch_completion",
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                **parsed,
            })
        torch.cuda.synchronize(self.device)
        e2e = time.perf_counter() - started
        for record in records:
            record["inference_seconds"] = e2e
            record["timing_kind"] = "batch1_per_request" if len(requests) == 1 else "synchronous_batch_completion_not_unbatched_latency"
        return records


def self_test():
    keys = ["H", "I"]
    assert parse_answer("The first is better.\nFINAL: H", keys, False)["choice"] == "H"
    assert parse_answer("FINAL: I", keys, True)["invalid"]
    assert parse_answer("H", keys, False)["invalid"]
    assert parse_answer("FINAL: A", keys, False)["invalid"]
    assert parse_answer("FINAL: H\nFINAL: I", keys, False)["invalid"]
    assert parse_answer("FINAL: H\nmore", keys, False)["invalid"]
    assert parse_answer("<think>evidence</think>\nFINAL: I", keys, False)["choice"] == "I"
    text = content_for("original visible history", {"instructions": "Choose", "criteria": {"H": "arm H", "I": "arm I"}}, keys)
    assert text.startswith("original visible history\n") and "arm H" in text and "arm I" in text
    raw_horizon = "original history\n" + LLAMA_HORIZON_RESPONSE + "\n" + ASSISTANT_HEADER
    changed = reasoning_raw_prompt(raw_horizon)
    assert changed.startswith("original history\n") and ASSISTANT_HEADER in changed and INSTRUCTION in changed
    raw_wcst = LLAMA_WCST_RESPONSE + "<|eot_id|>stimulus<|eot_id|>" + ASSISTANT_HEADER
    changed = reasoning_raw_prompt(raw_wcst)
    assert "stimulus" in changed and changed.endswith(ASSISTANT_HEADER) and INSTRUCTION in changed
    return {"status": "passed", "parser_cases": 7, "original_raw_prompt_cases": 2, "visible_state_preserved": True,
            "neural_calls": 0, "paid_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use task runner for neural generation, or --self-test.")
    print(json.dumps(self_test()))
