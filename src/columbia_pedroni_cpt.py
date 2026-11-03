"""Pedroni et al. 2018 Appendix A DU-CPT-1; not Centaur's ten-parameter model."""
import numpy as np
from scipy.special import expit


def logits(features, parameters):
    """features=(gain, loss, loss_cards, cards_remaining); parameters=(alpha, eta, lambda).

    Published final CPT-1 fixes theta=10, delta=1; value and weighting parameters
    are shared between gains and losses. Reference point updates after each card.
    """
    alpha, eta, loss_aversion = parameters
    gain, loss, loss_cards, remaining = np.asarray(features, dtype=float).T
    p_loss = loss_cards / remaining
    p_win = 1 - p_loss
    # exp(-(-log p)**eta) has its usual continuous endpoint values 0 and 1.
    with np.errstate(divide='ignore'):
        weighted_win = np.exp(-(-np.log(p_win)) ** eta)
        weighted_loss = np.exp(-(-np.log(p_loss)) ** eta)
    value = weighted_win * gain ** alpha - weighted_loss * loss_aversion * loss ** alpha
    return 10 * value


def probabilities(features, parameters):
    return expit(logits(features, parameters))


def sufficient_statistic_nll(parameters, features, count_turn, count_stop):
    """Exact full-data likelihood grouped by identical visible states; no subsampling."""
    alpha, eta, loss_aversion = parameters
    if not (0 < alpha < 3 and 0 < eta < 3 and 0 < loss_aversion < 10):
        return np.inf
    z = logits(features, parameters)
    losses = np.asarray(count_turn) * np.logaddexp(0, -z) + np.asarray(count_stop) * np.logaddexp(0, z)
    return float(losses.sum() / (np.asarray(count_turn).sum() + np.asarray(count_stop).sum()))
