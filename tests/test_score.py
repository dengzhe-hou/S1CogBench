import math

import pytest

from s1cogbench.score import Units, interval


def units(rows):
    A = Units()
    for uid, num, inf, unum in rows:
        A.add(uid, num, 1, inf, unum, 1)
    return A


def test_clearly_better_target_is_reliably_better():
    # False-belief style accuracy: 40 story blocks, 90% correct against a 50% Uniform.
    A = units([(i, 1. if i % 10 else 0., 0, .5) for i in range(40)])
    r = interval('false_belief_story_qa', A, 0.9)
    assert r['class'] == 'better' and r['lo'] > 0 and math.isclose(r['diff'], 40.0)


def test_constant_units_and_infinite_records():
    A = units([(i, .5, 0, .5) for i in range(10)])
    assert interval('false_belief_story_qa', A, 0.5)['class'] == 'constant'
    A = units([(0, 0., 1, math.log(2)), (1, .3, 0, math.log(2))])
    assert interval('multi_attribute_choice', A, math.inf)['class'] == 'infinite'


def test_point_must_match_mode_result():
    A = units([(i, 1., 0, .5) for i in range(5)])
    with pytest.raises(SystemExit):
        interval('false_belief_story_qa', A, 0.8)
