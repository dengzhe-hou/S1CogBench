KEYS = ["participant_id", "block", "game", "trial_num"]

def verify_rollout(timeline, result):
    """Detect omitted trials, reward misrouting, or human-history leakage."""
    import ast
    import numpy as np
    import pandas as pd
    expected = timeline.rename(columns={'trial_num_block': 'trial_num'}).set_index(KEYS).sort_index()
    actual = result.set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    if not actual.choice.isin(['H', 'I']).all():
        raise ValueError('Illegal action in full rollout.')
    free = actual.is_free.astype(bool)
    np.testing.assert_array_equal(free, expected.type.eq('free'))
    rewards = np.where(actual.choice.eq('H'), expected.reward_H, expected.reward_I)
    np.testing.assert_array_equal(actual.loc[free, 'reward'], rewards[free])
    np.testing.assert_array_equal(actual.loc[~free, 'choice'], expected.loc[~free, 'choice'])
    np.testing.assert_array_equal(actual.loc[~free, 'reward'], expected.loc[~free, 'reward'])
    for _, game in result.groupby(['participant_id', 'block', 'game'], sort=False):
        game = game.sort_values('trial_num')
        history = []
        cumulative = 0
        for row in game.itertuples():
            cumulative += row.reward
            if row.is_free:
                text_list = ast.literal_eval(row.prompt.removesuffix('You press <<'))
                observed_history = text_list[9:]
                if observed_history != history:
                    raise ValueError('Prompt is not the exact own-action/observed-reward history.')
                if row.cumulative_reward != cumulative:
                    raise ValueError('Incorrect free-trial cumulative reward.')
                history.append(f'You press <<{row.choice}>> and get {row.reward} points.')
            else:
                history.append(f'You are instructed to press {row.choice} and get {row.reward} points.')
