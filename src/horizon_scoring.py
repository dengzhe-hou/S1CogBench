"""CogBench Horizon scoring, with explicit failures instead of zero effects.

The regression uses latent reward means for scoring only. Its design, coefficient
definitions and normal-approximation CI half widths follow the public
Experiments/HorizonTask/store.py. Reward means include forced observations.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError, PerfectSeparationWarning,
)


def _game_frames(trials):
    columns = ['run', 'trial', 'choice', 'reward', 'mean0', 'mean1']
    missing = set(columns) - set(trials.columns)
    if missing:
        raise ValueError(f'Missing Horizon columns: {sorted(missing)}')
    if trials.empty or trials['run'].isna().any():
        raise ValueError('Horizon scoring requires complete games with run IDs.')
    if not np.isfinite(trials[columns[1:]].to_numpy(dtype=float)).all():
        raise ValueError('Horizon trials contain nonfinite numeric values.')
    if not trials['choice'].isin([0, 1]).all():
        raise ValueError('Horizon choices must be physical arm 0 or 1.')
    games = []
    for run, frame in trials.groupby('run', sort=False):
        frame = frame.sort_values('trial')
        if len(frame) not in (5, 10) or not np.array_equal(
                frame['trial'].to_numpy(), np.arange(len(frame))):
            raise ValueError(f'Run {run}: expected trials 0..4 or 0..9.')
        if any(frame[column].nunique() != 1 for column in ('mean0', 'mean1')):
            raise ValueError(f'Run {run}: latent means changed within a game.')
        counts = np.bincount(frame.iloc[:4]['choice'].to_numpy(dtype=int), minlength=2)
        if sorted(counts.tolist()) not in ([1, 3], [2, 2]):
            raise ValueError(f'Run {run}: forced observations must be 3:1 or 2:2.')
        games.append(frame)
    return games


def _design(games, condition):
    if condition not in ('equal', 'unequal'):
        raise ValueError("condition must be 'equal' or 'unequal'.")
    predictors, choices = [], []
    for frame in games:
        observed_arm1 = int(frame.iloc[:4]['choice'].sum())
        equal = observed_arm1 == 2
        if equal != (condition == 'equal'):
            continue
        difference = float(frame.iloc[0]['mean1'] - frame.iloc[0]['mean0'])
        choice = int(frame.iloc[4]['choice'])
        if not equal and observed_arm1 == 3:
            # Recode outcome and reward difference toward the less-observed arm.
            difference, choice = -difference, 1 - choice
        horizon = 1.0 if len(frame) == 10 else -1.0
        predictors.append([difference, horizon, difference * horizon, 1.0])
        choices.append(choice)
    return np.asarray(predictors, dtype=float).reshape(-1, 4), np.asarray(choices, dtype=int)


def exploration_design(trials: pd.DataFrame, condition: str):
    """Return (X, y) for first free choices in equal or unequal information.

    X columns are latent mean difference, horizon (-1/+1), their interaction,
    and intercept. Unequal choices and differences refer to the less-seen arm.
    Complete games are required; input row order does not matter.
    """
    return _design(_game_frames(trials), condition)


def _fit_effect(design, choices, coefficient, sign):
    result = dict(value=None, ci95_half_width=None, status='not_fitted', games=len(choices))
    if not len(choices):
        result['status'] = 'no_games'
        return result
    if np.unique(choices).size == 1:
        result['status'] = 'constant_outcome'
        return result
    if np.linalg.matrix_rank(design) < 4:
        result['status'] = 'rank_deficient_design'
        return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted = sm.Logit(choices, design).fit(disp=False)
        if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
            result['status'] = 'perfect_separation'
        elif not fitted.mle_retvals['converged']:
            result['status'] = 'not_converged'
        elif not (np.isfinite(fitted.params).all() and np.isfinite(fitted.bse).all()):
            result['status'] = 'nonfinite_estimate'
        else:
            result.update(value=float(sign * fitted.params[coefficient]),
                          ci95_half_width=float(1.96 * fitted.bse[coefficient]),
                          status='ok')
    except PerfectSeparationError:
        result['status'] = 'perfect_separation'
    except np.linalg.LinAlgError:
        result['status'] = 'singular_fit'
    return result


def score_horizon(trials: pd.DataFrame) -> dict:
    """Score complete Horizon games in CogBench's published column convention.

    Required columns: run, trial, choice, reward, mean0, mean1. Each run has
    trials 0..4 or 0..9; the first four rows are forced observations. Returns
    JSON-safe scalars and nested effect results. Unestimable effects are null,
    with a status explaining why, rather than the official scorer's fallback 0.
    The all-trial reward CI deliberately retains the official trial-level SE;
    it is not a paired game-level comparison interval.
    """
    games = _game_frames(trials)
    rewards = trials['reward'].to_numpy(dtype=float)
    equal_design, equal_choices = _design(games, 'equal')
    unequal_design, unequal_choices = _design(games, 'unequal')
    return dict(
        games=len(games), trials=len(trials),
        mean_reward_all_trials=float(rewards.mean()),
        mean_reward_all_trials_ci95_half_width=float(
            1.96 * rewards.std(ddof=1) / np.sqrt(len(rewards))),
        directed_exploration=_fit_effect(unequal_design, unequal_choices, 1, 1.0),
        random_exploration=_fit_effect(equal_design, equal_choices, 2, -1.0),
    )
