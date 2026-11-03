"""Official Intern-Decision-4B typed-choice inference, frozen BF16 checkpoint."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/intern-decision-4b"
MODEL_ID = "internlm/Intern-Decision-4B"
REVISION = "1e65a988dacbcf00b84aee7366f9e47563407000"


def load_official_inference():
    """Import the inference.py distributed with this exact official checkpoint."""
    name = "s1cogbench_intern_decision_4b_official"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODEL_DIR / "inference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def choice_request(state, question, output_keys):
    if not output_keys or len(set(output_keys)) != len(output_keys):
        raise ValueError("Supply distinct ordered legal choice keys.")
    if set(output_keys) != set(question["criteria"]):
        raise ValueError("Question criteria and legal actions differ.")
    return {
        "state": state,
        "questions": {
            "action": {
                "type": "choice",
                "instructions": question["instructions"],
                "criteria": {key: question["criteria"][key] for key in output_keys},
            }
        },
    }


class InternDecisionChoice:
    def __init__(self, device):
        import torch

        config = json.loads((MODEL_DIR / "config.json").read_text())
        if config.get("quantization_config"):
            raise ValueError("This comparison requires the official unquantized checkpoint.")
        torch.set_num_threads(4)
        self.official = load_official_inference()
        self.engine = self.official.DecisionEngine(
            checkpoint=MODEL_DIR, device=device, dtype="bfloat16",
            attn_implementation="sdpa", max_length=8192,
        )
        self.max_input_tokens = 0
        self.metadata = {
            "model_id": MODEL_ID, "revision": REVISION,
            "model_directory": str(MODEL_DIR),
            "backbone": "Qwen/Qwen3.5-4B",
            "backend": "Official checkpoint inference.py DecisionEngine / native HF",
            "dtype": "bfloat16", "attention": "sdpa",
            "linear_attention_kernel_modules": sorted({
                layer.chunk_gated_delta_rule.__module__ for layer in self.engine.backend.model.modules()
                if hasattr(layer, "chunk_gated_delta_rule")}),
            "linear_attention_norm_classes": sorted({
                type(layer.norm).__module__ + "." + type(layer.norm).__name__
                for layer in self.engine.backend.model.modules() if hasattr(layer, "chunk_gated_delta_rule")}),
            "actual_parameter_dtypes": sorted({str(p.dtype) for p in self.engine.backend.model.parameters()}),
            "training": "Frozen official weights; no benchmark training or prompt tuning.",
            "probabilities": "Official legal-symbol distribution with checkpoint default temperature 1.99241824.",
            "probability_semantics": "Action-choice probabilities, not probability of task success.",
            "native_choice": "Official highest-probability choice with lexical tie-breaking.",
            "interface": "Original masked assistant JSON skeleton; one causal forward; no free-text generation.",
            "max_input_tokens": 8192,
            "history": "Full supplied state; official encoder raises above 8192 tokens without truncation.",
            "calibration": "Published checkpoint calibration, not fitted on S1CogBench; not transferred to Qwen.",
            "numerical_normalization": "Divide official probabilities by their sum; raw values retained. No recalibration.",
            "prior_training_overlap": "Not ruled out by public model documentation.",
        }

    def choice_probabilities(self, state, question, output_keys):
        import numpy as np

        request = choice_request(state, question, output_keys)
        response = self.engine.predict(request)
        answer = response["answers"]["action"]
        probability = np.asarray([answer["probabilities"][key] for key in output_keys], dtype=float)
        if not np.isfinite(probability).all() or np.any(probability < 0) or not np.isclose(probability.sum(), 1.0):
            raise ValueError("Official engine returned invalid legal-choice probabilities.")
        self.last_raw_probabilities = probability.tolist()
        self.last_native_choice = answer["choice"]
        self.last_input_tokens = int(response["usage"]["input_tokens"])
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        compiled = self.official.compile_row(request)
        self.last_prompt = self.engine.tokenizer.apply_chat_template(
            compiled.messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True,
        )
        self.last_label_mapping = dict(zip(compiled.symbols["action"], output_keys))
        self.last_response = response
        return probability / probability.sum()


def create_agent(device):
    return InternDecisionChoice(device)
