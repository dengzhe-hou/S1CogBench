"""Describe existing modes and inspect an adapter without importing or calling it."""
import ast
from pathlib import Path


def effective_action_rule(mode, requested):
    if mode.family == 'restless':
        return 'source_native' if requested == 'native' else 'unsupported'
    if (mode.family == 'wcst' and mode.mode == 'autonomous_generative') or mode.family in {
            'twostep', 'horizon_generative', 'mouselab_generative', 'bandit', 'expansion',
            'extension_autonomous'}:
        return requested
    return 'not_applied'


def validate_action_rules(modes, requested):
    unsupported = [m.key for m in modes if effective_action_rule(m, requested) == 'unsupported']
    if unsupported:
        raise SystemExit('--action-rule sample is not implemented by the source-native runner for '
                         + ', '.join(unsupported) + '. Use its native rule, or select modes that support sampling.')


def inspect_adapter(path):
    path = Path(path).resolve()
    tree = ast.parse(path.read_text(), filename=str(path))
    factory = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_agent'), None)
    if factory is None:
        raise SystemExit(f'{path} must define create_agent(device). This check does not import the adapter.')
    args = factory.args
    positional = args.posonlyargs + args.args
    required = len(positional) - len(args.defaults)
    if (not positional and args.vararg is None) or required > 1 or any(d is None for d in args.kw_defaults):
        raise SystemExit(f'{path}: create_agent must accept one positional device without other required arguments.')
    return {'file': str(path), 'check': 'syntax_and_factory_signature_only',
            'imported': False, 'agent_created': False, 'model_API_calls': 0}


def comparison_context(mode, requested):
    context = {k: v.format(rule=requested) for k, v in mode.primary.items()}
    if effective_action_rule(mode, requested) == 'sample' and '{rule}' not in mode.primary['protocol']:
        context['protocol'] += '_sample'
    if mode.family == 'mouselab_generative':
        context['scope'] = f'{requested}_autonomous_full'
    return context


def describe(modes, requested):
    rows = []
    for m in modes:
        rule = effective_action_rule(m, requested)
        methods = list(m.interface)
        if m.family == 'wcst' and rule == 'sample':
            methods.remove('last_native_choice')
        if m.family == 'bandit' and rule == 'native':
            methods.append('last_native_choice')
        context = comparison_context(m, requested)
        rows.append({'key': m.key, 'label': m.label, 'track': m.track, 'adapter_interface': methods,
                     'source_family': m.family, 'expected_input_units': m.inputs_expected,
                     'expected_scored_units': m.scored_expected,
                     'input_paths': list(m.inputs), 'requested_action_rule': requested,
                     'effective_action_rule': rule, 'comparison_context': context, 'note': m.note})
    return rows
