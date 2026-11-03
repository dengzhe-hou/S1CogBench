"""Current-decision text view of the published Hybrid task's original X rows.

The author X CSV contains current objective/remembered statistics. Identifiers,
future mouse movement, final outcome, hidden dot probabilities and fitted VoI
estimates are not part of the model state.
"""

ACTIONS = ['sample_left', 'sample_right', 'select_left', 'select_right']
X_COLUMNS = ['subject', 'session', 'trial', 'nL', 'nR', 'nB', 'rL', 'rR', 'μL', 'μR',
             'gL', 'gR', 'first_visit', 'after_first_visit', 't']
QUESTION = {
    'type': 'choice',
    'instructions': 'Predict the participant’s next action from the current task statistics.',
    'criteria': {
        'sample_left': 'Continue gathering dots from the left patch; move there if needed.',
        'sample_right': 'Continue gathering dots from the right patch; move there if needed.',
        'select_left': 'Stop gathering dots and finally choose the left patch.',
        'select_right': 'Stop gathering dots and finally choose the right patch.',
    },
}


def _whole(value):
    number = float(value)
    if not number.is_integer():
        raise ValueError(f'Expected an integer-valued task statistic, got {value}.')
    return int(number)


def state_from_author_x(row):
    if list(row) != X_COLUMNS:
        raise ValueError('Unexpected published Hybrid X column order.')
    gaze_left, gaze_right = _whole(row['gL']), _whole(row['gR'])
    first, after = _whole(row['first_visit']), _whole(row['after_first_visit'])
    if (gaze_left, gaze_right) not in [(1, 0), (0, 1)] or (first, after) not in [(1, 0), (0, 1)]:
        raise ValueError('Current cursor and visit indicators must each select exactly one state.')
    current = 'left' if gaze_left else 'right'
    lines = [
        'Task: Two available patches contain red and black dots. Gather information by sampling '
        'a patch, or stop and finally select one. A third patch is blocked.',
        f'Current cursor: {current} patch.',
        'Visit phase: first attended patch, before the first switch.' if first else
        'Visit phase: after the first switch; earlier revealed dots in the unattended patch '
        'must be remembered because they are covered again.',
        f'Blocked patch: {_whole(row["nB"])} initially green-covered dots; it cannot be sampled '
        'or selected. This X row does not encode its red/black colors.',
        f'Event steps since the latest switch: {_whole(row["t"])}. This is not elapsed seconds.',
    ]
    for side, gaze in [('left', gaze_left), ('right', gaze_right)]:
        letter = 'L' if side == 'left' else 'R'
        count, red = _whole(row['n' + letter]), _whole(row['r' + letter])
        posterior = row['μ' + letter]
        if first and not gaze:
            if red != 0 or abs(float(posterior) - 0.5) > 1e-6:
                raise ValueError('Unvisited first-visit side must have no observed colors and prior mean 0.5.')
            lines.append(f'{side.title()} patch: not yet visited in this sampling sequence. '
                         f'Its X statistic records an initial reveal count of {count} dots, '
                         'but no red/black evidence for this side is encoded in this row. '
                         'Current encoded Beta(1,1) prior mean: 0.5.')
        else:
            if red < 0 or red > count:
                raise ValueError('Observed red count is outside observed dot count.')
            source = 'currently attended and visible' if gaze else 'previously revealed and remembered'
            lines.append(f'{side.title()} patch ({source}): {count} revealed dots, '
                         f'{red} red and {count-red} black; '
                         f'Beta(1,1) posterior mean of red proportion: {posterior}.')
    return '\n'.join(lines)
