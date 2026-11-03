"""Exhaustive initial/one-reveal Exp1 states for presentation and cost probes.

These are constructed mechanism states, not human data or rollout samples.
"""
import numpy as np
from mouselab_task import new_game, choice_question, one_step_net_voi, visible_belief, VALUES

PRESENTATIONS = ('original', 'order_only', 'label_only')
CONSTRUCTION_SEED = 260926200


def state_pool():
    """Every first legal node and possible value, plus the initial state."""
    env = new_game(CONSTRUCTION_SEED, 1.)
    return ([dict(state_id='initial', reveals=[])] +
            [dict(state_id=f'node{node}_value{value}', reveals=[[node, value]])
             for node in env.tree[0] for value in VALUES])


def make_state(spec, cost):
    env = new_game(CONSTRUCTION_SEED, cost)
    for node, value in spec['reveals']:
        assert value in VALUES
        env.ground_truth[node] = value
        _, reward, done, _ = env.step(node)
        assert reward == -cost and not done
    return env


def presentation_question(actions, term_action, presentation):
    """Return the question and keys in CANONICAL physical-action order.

    order_only changes insertion/display order, preserving letter meanings.
    label_only exchanges the STOP/last-reveal letters, preserving row order.
    """
    question, keys = choice_question(actions, term_action)
    assert actions[0] == term_action
    if presentation == 'original':
        return question, keys
    if presentation == 'order_only':
        shown_keys = keys.copy()
        shown_keys[0], shown_keys[-1] = shown_keys[-1], shown_keys[0]
        question['criteria'] = {key: question['criteria'][key] for key in shown_keys}
        return question, keys
    if presentation == 'label_only':
        labels = keys.copy()
        labels[0], labels[-1] = labels[-1], labels[0]
        descriptions = list(question['criteria'].values())
        question['criteria'] = dict(zip(labels, descriptions))
        return question, labels
    raise ValueError(presentation)


def choice_metrics(env, actions, cost, probabilities):
    """Score the executed argmax against the reveal-once-then-stop family."""
    p = np.asarray(probabilities, dtype=float)
    assert len(p) == len(actions) and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(), 1., atol=1e-6)
    gains = one_step_net_voi(env, actions, cost)
    idx = int(np.argmax(p))
    best = float(gains.max())
    stop_idx = actions.index(env.term_action)
    return dict(action=int(actions[idx]), selected_index=idx,
                stop_probability=float(p[stop_idx]), stop=bool(idx == stop_idx),
                one_step_net_gains=gains.tolist(), best_one_step_gain=best,
                chosen_one_step_gain=float(gains[idx]),
                chosen_one_step_gap=best-float(gains[idx]),
                stop_with_positive_one_step_gain=bool(idx == stop_idx and best > 0),
                reveal_with_negative_one_step_gain=bool(idx != stop_idx and gains[idx] < 0),
                top_probability_tie=bool(np.count_nonzero(p == p.max()) > 1))


def boundary_reference(env, cost):
    """Proven STOP/continue requirements for the existing 17 shallow states.

    This returns a conservative loss bound for violating the requirement,
    not the optimal value of every action or a closed-loop return.
    """
    assert cost in (0., 31.)
    b = visible_belief(env._state)
    roots = env.tree[0]
    assert len(roots) == 4 and b[0] == 0
    assert sum(b[r] is not None for r in roots) <= 1
    for root in roots:
        assert b[root] is None or b[root] in VALUES
        assert len(env.tree[root]) == 1
        middle = env.tree[root][0]
        assert b[middle] is None and len(env.tree[middle]) == 2
        assert all(b[leaf] is None and not env.tree[leaf] for leaf in env.tree[middle])
    stop_value = float(env.expected_term_reward(env._state))
    assert stop_value >= 0
    maximum_route_value = 3 * max(VALUES)
    leaf_max_expectation = float(np.mean([max(a, z) for a in VALUES for z in VALUES]))
    if cost == 31.:
        continuation_upper_bound = maximum_route_value - cost
        assert stop_value > continuation_upper_bound
        return dict(required_stop=True, stop_value=stop_value,
                    violating_action_loss_lower_bound=stop_value-continuation_upper_bound,
                    maximum_route_value=maximum_route_value,
                    continuation_value_upper_bound=continuation_upper_bound,
                    argument='Any reveal costs more than the largest possible route payoff')
    known_nonnegative = [r for r in roots if b[r] is not None and b[r] >= 0]
    root = known_nonnegative[0] if known_nonnegative else next(r for r in roots if b[r] is None)
    branch_mean = b[root] if b[root] is not None else 0.
    feasible_value_lower_bound = branch_mean + leaf_max_expectation
    assert feasible_value_lower_bound > stop_value
    return dict(required_stop=False, stop_value=stop_value,
                violating_action_loss_lower_bound=feasible_value_lower_bound-stop_value,
                maximum_route_value=maximum_route_value,
                feasible_continuation_value_lower_bound=feasible_value_lower_bound,
                feasible_branch_root=root, leaf_max_expectation=leaf_max_expectation,
                argument='Freely revealing a full legal branch strictly improves expected payoff')
