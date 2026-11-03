"""Qwen3.5-4B same-backbone control in the benchmark's legal-choice track."""
import json
from pathlib import Path
import string

from src.choice_letter_tokens import ChoiceLetterTokens, IMPLEMENTATION

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/qwen3.5-4b"
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def format_choice_prompt(tokenizer, state, question, output_keys):
    if not output_keys or len(set(output_keys)) != len(output_keys):
        raise ValueError("Supply distinct ordered legal choice keys.")
    if set(output_keys) != set(question["criteria"]):
        raise ValueError("Question criteria and legal actions differ.")
    labels = (list(output_keys)
              if all(len(key) == 1 and key in string.ascii_uppercase for key in output_keys)
              else list(string.ascii_uppercase[:len(output_keys)]))
    if len(labels) != len(output_keys):
        raise ValueError("This benchmark comparison supports at most 26 legal actions.")
    options = "\n".join(f"{label}: {key} - {question['criteria'][key]}"
                        for label, key in zip(labels, output_keys))
    content = state + "\n" + question["instructions"] + "\n" + options
    content += "\nChoose one of the listed option letters."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return prompt, labels


class Qwen35Choice:
    def __init__(self, device):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.device = torch.device(device)
        config = json.loads((MODEL_DIR / "config.json").read_text())
        if config.get("quantization_config"):
            raise ValueError("This comparison requires the official unquantized checkpoint.")
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self._choice_letter_tokens = ChoiceLetterTokens(self.tokenizer)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        self.max_context_tokens = int(config["text_config"]["max_position_embeddings"])
        self.max_input_tokens = 0
        self.metadata = {
            "model_id": MODEL_ID, "revision": REVISION,
            "model_directory": str(MODEL_DIR),
            "backbone_role": "Official instruction-model backbone named by Intern-Decision-4B; not Qwen3.5-4B-Base.",
            "dtype": "bfloat16", "attention": "sdpa",
            "linear_attention_kernel_modules": sorted({
                layer.chunk_gated_delta_rule.__module__ for layer in self.model.modules()
                if hasattr(layer, "chunk_gated_delta_rule")}),
            "linear_attention_norm_classes": sorted({
                type(layer.norm).__module__ + "." + type(layer.norm).__name__
                for layer in self.model.modules() if hasattr(layer, "chunk_gated_delta_rule")}),
            "actual_parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()}),
            "training": "Frozen official weights; no benchmark fitting or probability calibration.",
            "probabilities": "Temperature-one softmax over legal next-letter logits.",
            "probability_semantics": "Constrained action-choice distribution, not confidence of success.",
            "native_choice": "Argmax in supplied legal-action order; no free-text generation.",
            "interface": "Explicit benchmark choice adaptation; official non-thinking chat template.",
            "prompt_relation": "Same task content and letter formatting as current Llama8B legal-choice adapter.",
            "history": "Complete provided state; context overflow raises, without truncation or summarization.",
            "thinking": "enable_thinking=False; no hidden reasoning tokens generated.",
            "choice_tokenization_implementation": IMPLEMENTATION,
            "choice_tokenization_timing": "CPU suffix cache and full-prompt suffix check are included in adapter latency; scalar model forward is unchanged.",
            "prior_training_overlap": "Not ruled out by frozen evaluation.",
        }

    def choice_probabilities(self, state, question, output_keys):
        import numpy as np
        import torch

        prompt, labels = format_choice_prompt(self.tokenizer, state, question, output_keys)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prefix = inputs["input_ids"][0].tolist()
        token_ids = self._choice_letter_tokens.token_ids(prompt, prefix, labels, qwen=True)
        if len(set(token_ids)) != len(output_keys):
            raise ValueError("Legal answer symbols must have distinct token ids.")
        self.last_input_tokens = len(prefix)
        if self.last_input_tokens > self.max_context_tokens:
            raise ValueError("Complete supplied history exceeds the model context; truncation is forbidden.")
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = prompt
        self.last_label_mapping = dict(zip(labels, output_keys))
        with torch.inference_mode():
            logits = self.model(**inputs.to(self.device), use_cache=False, logits_to_keep=1).logits[0, -1].float()
        selected = logits[token_ids].double()
        if not torch.isfinite(selected).all():
            raise ValueError("Non-finite legal choice logits.")
        probability = torch.softmax(selected, dim=-1).cpu().numpy()
        self.last_allowed_token_mass = float(torch.exp(
            torch.logsumexp(logits[token_ids], 0) - torch.logsumexp(logits, 0)))
        self.last_native_choice = output_keys[int(np.argmax(probability))]
        return probability


def create_agent(device):
    return Qwen35Choice(device)
