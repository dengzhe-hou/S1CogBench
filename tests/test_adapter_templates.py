"""Offline checks of the two API adapter templates; no network, no key.

    python tests/test_adapter_templates.py

The letter cases are the self-test of the retained FINAL-line parser
(generation_agent.py of the accepted Claude runs). The rest check the
generalization to other legal keys and the Jev-style readout.
"""
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'examples/templates/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = load('generative_api_choice')
D = load('decision_api_choice')


def test_retained_letter_parser():
    keys = ['H', 'I']
    p = G.parse_answer
    assert p('<think>FINAL: H</think>\nFINAL: I', keys, False)['choice'] == 'I'
    assert p('<think>FINAL: H</think>\nFINAL: I', keys, True)['invalid']
    assert p('<think>FINAL: I', keys, False)['invalid']
    assert p('The first is better.\nFINAL: H', keys, False)['choice'] == 'H'
    assert p('FINAL: I', keys, True)['invalid']
    assert p('H', keys, False)['invalid']
    assert p('FINAL: A', keys, False)['invalid']
    assert p('FINAL: H\nFINAL: I', keys, False)['invalid']
    assert p('FINAL: H\nmore', keys, False)['invalid']
    assert p('<think>evidence</think>\nFINAL: I', keys, False)['choice'] == 'I'


def test_generalized_keys():
    p = G.parse_answer
    assert p('reasoning\nFINAL: 2', ['1', '2'], False)['choice'] == '2'
    assert p('FINAL: 3', ['1', '2'], False)['invalid']
    integers = [str(i) for i in range(101)]
    assert p('so\nFINAL: 57', integers, False)['choice'] == '57'
    assert p('so\nFINAL: 100', integers, False)['choice'] == '100'
    assert p('so\nFINAL: 101', integers, False)['invalid']
    assert p('FINAL: 5\nFINAL: 57', integers, False)['invalid']
    names = ['Video A', 'Video B']
    assert p('FINAL: Video B', names, False)['choice'] == 'Video B'
    assert p('FINAL: Video C', names, False)['invalid']
    assert G.content_for('s', {'instructions': 'q', 'criteria': {'1': 'a', '2': 'b'}}, ['1', '2']).endswith(
        'Replace X with one of the listed legal option numbers. Do not put any text after that line.')


def test_generative_agent():
    replies = iter([{'content': [{'type': 'thinking', 'thinking': 'FINAL: H'},
                                 {'type': 'text', 'text': 'I is better.\nFINAL: I'}], 'stop_reason': 'end_turn'},
                    {'content': [{'type': 'text', 'text': 'FINAL: I'}], 'stop_reason': 'max_tokens'},
                    {'content': [{'type': 'text', 'text': 'about\nFINAL: 0.25'}], 'stop_reason': 'end_turn'}])
    agent = G.GenerativeAPIChoice(post=lambda body: next(replies))
    q = {'instructions': 'Choose.', 'criteria': {'H': 'arm H', 'I': 'arm I'}}
    assert agent.choice_probabilities('state', q, ['H', 'I']) == [0.0, 1.0] and agent.last_native_choice == 'I'
    assert agent.choice_probabilities('state', q, ['H', 'I']) == [.5, .5] and agent.last_native_choice == G.INVALID
    assert agent.posterior_probability('state', {'instructions': 'it is left'}) == .25


def test_decision_readout():
    keys = ['C', 'N']
    choice, raw, p = D.parse_choice({'type': 'choice', 'choice': 'N', 'probabilities': {'C': .14, 'N': .87}}, keys)
    assert choice == 'N' and raw == [.14, .87] and p == [.14 / 1.01, .87 / 1.01]
    choice, raw, p = D.parse_choice({'type': 'choice', 'choice': 'C', 'probabilities': {'C': 1.0, 'N': 0.0}}, keys)
    assert p == [1.0, 0.0]  # no floor
    for bad in ({'type': 'choice', 'choice': 'X', 'probabilities': {'C': .5, 'N': .5}},
                {'type': 'choice', 'choice': 'C', 'probabilities': {'C': .5}},
                {'type': 'choice', 'choice': 'C', 'probabilities': {'C': .5, 'N': .6}},
                {'type': 'noul', 'noul': .3}):
        try:
            D.parse_choice(bad, keys)
        except D.InvalidDecision:
            continue
        raise AssertionError(bad)
    assert D.parse_event({'type': 'noul', 'noul': .94}) == .94
    sent = []
    agent = D.DecisionAPIChoice(post=lambda body: (sent.append(body), {
        'answers': {'answer': {'type': 'choice', 'choice': 'C', 'probabilities': {'C': .7, 'N': .3}}},
        'usage': {'input_tokens': 10}})[1])
    q = {'type': 'choice', 'instructions': 'Choose.', 'criteria': {'C': 'c', 'N': 'n'}}
    assert agent.choice_probabilities('state', q, keys) == [.7, .3] and agent.last_native_choice == 'C'
    assert sent == [{'model': D.MODEL, 'state': 'state', 'questions': {'answer': q}}]
    assert math.isclose(sum(agent.choice_probabilities('state', q, keys)), 1)


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('passed', name)
