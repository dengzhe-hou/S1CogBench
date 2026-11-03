"""Laya's documented input-budget override for complete CogBench histories."""
from pathlib import Path


def create_agent(device):
    from examples.laya_choice_agent import create_agent as load_default

    agent = load_default(device)
    default_max_len = agent.agent.cfg['max_len']
    agent.agent.cfg['max_len'] = 1024
    agent.metadata = {
        'model': 'Laya-English',
        'checkpoint': str(Path(__file__).resolve().parents[1] / 'models/laya-english'),
        'training': 'Frozen checkpoint; no task fitting or calibration.',
        'sdk_default_max_len': default_max_len,
        'max_len': 1024,
        'head_max_len': agent.agent.cfg['head_max_len'],
        'forward_dtype': str(agent.agent.dtype),
        'actual_parameter_dtypes': sorted({str(p.dtype) for p in agent.agent.model.parameters()}),
        'temperature_config': agent.agent.cfg.get('temperature'),
        'temperature_by_options': agent.agent.cfg.get('temperature_by_options'),
        'input_processing': 'Complete original visible history; assert identical tokens to untruncated build_sequence.',
        'budget_source': 'Official external/laya/README.md input budget override and Agent.predict overrides.',
        'protocol_adaptation': 'Typed choice and a 1024-token input cap; original weights, head budget and temperatures retained.',
    }
    return agent
