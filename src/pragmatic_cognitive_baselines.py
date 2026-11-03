"""Source-grounded readouts for the published Yoon and Kao pragmatic models.

No task test labels are used in forward(). Human priors / author fits are explicitly
calibration data, not independent held-out predictions.
"""
import csv
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'literature/baseline_resolution_20261017'
PRICES = (50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001)
QUDS = ('price', 'valence', 'priceValence', 'approxPrice', 'approxPriceValence')


def read_webppl_priors(kind='human'):
    """Read the source's exact human comment block or active Meta-Llama constants."""
    source = (SOURCES / 'number_interpretation/rsa.wppl').read_text()
    price_source, valence_source = source.split('var valencePrior = function(state) {')
    result = {}
    for obj in ('kettle', 'watch', 'laptop'):
        pattern = r"(?:if|else if) \(argv\['object'\] == '" + obj + r"'\) \{(.*?)(?=else if|\n}\n|\Z)"
        price = re.search(pattern, price_source, re.S).group(1)
        valence = re.search(pattern, valence_source, re.S).group(1)
        if kind == 'human':
            # The first commented parameter block in each object is explicitly 'human'.
            price = price.split('// human', 1)[1] if obj != 'kettle' else price.split('//  ps: [ // human', 1)[1]
            price = price.split('// gpt-', 1)[0]
            if obj == 'kettle':
                price = '[' + price
            price = re.sub(r'//', '', price)
            vals = re.search(r'\[([^\]]+)\]', price, re.S).group(1)
            price_values = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', vals)]
            valence = valence.split('// human', 1)[1].split('// gpt-', 1)[0]
            valence = re.sub(r'//', '', valence)
        elif kind == 'meta-llama':
            # Only uncommented assignments are active in the released current file.
            price = re.sub(r'//[^\n]*', '', price)
            vals = re.search(r'ps:\s*\[([^\]]+)\]', price, re.S).group(1)
            price_values = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', vals)]
            valence = re.sub(r'//[^\n]*', '', valence)
        else:
            raise ValueError('Only source-verified human and active Meta-Llama constants are implemented.')
        pairs = re.findall(r'(\d+)\s*:\s*(\d+(?:\.\d+)?)', re.search(r'var probs\s*=\s*\{(.*?)\}', valence, re.S).group(1))
        valence_values = {int(k): float(v) for k, v in pairs}
        if len(price_values) != 10 or set(valence_values) != set(PRICES):
            raise ValueError('Original source prior extraction did not recover all ten prices.')
        total = sum(price_values)  # WebPPL categorical() normalizes supplied weights.
        result[obj] = {'price': dict(zip(PRICES, [p / total for p in price_values])),
                       'affect': valence_values}
    return result


def qud_value(qud, price, affect):
    approximate = 10 * math.floor(price / 10 + 0.5)  # JS Math.round on these positive inputs.
    return {'price': (price,), 'valence': (affect,), 'priceValence': (price, affect),
            'approxPrice': (approximate,), 'approxPriceValence': (approximate, affect)}[qud]


class KaoRSA:
    """Finite exact enumeration of the released RSA, with published fixed parameters.

    cost=3.4 and power=.25 are Kao et al. (2014), Materials and Methods,
    final halo-fit values, not a fit to this benchmark. The published power
    transform applies to joint meaning (price, affect) before marginalization.
    """
    def __init__(self, priors='human', cost=3.4, power=.25, qud_source='kao2014'):
        self.priors = read_webppl_priors(priors)
        self.cost, self.power = cost, power
        # Kao 2014 M&M explicitly has 2 precision × 3 projection goals, uniform.
        # The two affect-only goals are equivalent but each keeps its prior mass.
        # Tsvilodub's newer WebPPL instead lists five distinct uniform QUDs.
        if qud_source == 'kao2014':
            self.quds = QUDS + ('valence',)
        elif qud_source == 'tsvilodub2025':
            self.quds = QUDS
        else:
            raise ValueError('Unknown source QUD prior.')

    def joint(self, obj, utterance, power=None):
        priors = self.priors[obj]
        prior = {(s, a): priors['price'][s] * (priors['affect'][s] if a else 1 - priors['affect'][s])
                 for s in PRICES for a in (False, True)}
        utterance_weights = {u: math.exp(-(1 if u % 10 == 0 else self.cost)) for u in PRICES}
        # L0 is a distribution on each QUD's value, conditioned on literal price=u.
        literal = {}
        for qud in self.quds:
            for u in PRICES:
                dist = {}
                for a in (False, True):
                    key = qud_value(qud, u, a)
                    dist[key] = dist.get(key, 0) + (priors['affect'][u] if a else 1 - priors['affect'][u])
                literal[qud, u] = dist
        posterior = {}
        for (s, a), prior_mass in prior.items():
            speaker_likelihood = 0
            for qud in self.quds:
                meaning = qud_value(qud, s, a)
                denom = sum(utterance_weights[u] * literal[qud, u].get(meaning, 0) for u in PRICES)
                likelihood = utterance_weights[utterance] * literal[qud, utterance].get(meaning, 0) / denom
                speaker_likelihood += likelihood / len(self.quds)
            posterior[s, a] = prior_mass * speaker_likelihood
        total = sum(posterior.values())
        power = self.power if power is None else power
        powered = {key: (value / total) ** power for key, value in posterior.items()}
        total = sum(powered.values())
        return {key: value / total for key, value in powered.items()}

    def price_distribution(self, obj, utterance, power=None):
        joint = self.joint(obj, utterance, power)
        return {s: joint[s, False] + joint[s, True] for s in PRICES}

    def forward(self, trial):
        obj = 'kettle' if 'electric kettle' in trial.state else 'watch' if 'watch' in trial.state else 'laptop'
        price = int(re.search(r'It cost \$(\d+)', trial.state).group(1))
        if trial.experiment.endswith('/exp3b'):
            return {'reported_probability': self.priors[obj]['affect'][price],
                    'baseline_role': 'human_affect_prior_input_reference_not_prediction'}
        target = int(re.fullmatch(r'state_prob_(\d+)', trial.query_tag).group(1))
        return {'reported_probability': self.price_distribution(obj, price)[target],
                'baseline_role': 'published_fixed_parameter_rsa_prediction',
                'original_utterance': price, 'original_price_query': target, 'original_object': obj}


class YoonAuthorPosterior:
    """Normalize the author's rounded posterior means; retain unmodified values."""
    def __init__(self, model='full'):
        folder = SOURCES / 'polite_speaker'
        self.literal = {(r['state'], r['utterance']): r for r in json.loads((folder / '02_analysis/01_data/literal_semantics.json').read_text())}
        path = folder / '03_model/01_posterior_predictives/03_output_processed/postpred_probability_mean.csv'
        self.joint = {}
        self.raw_joint = {}
        self.raw_sums = {}
        # Original posterior uses initial letters (informational/social/presentational);
        # the author's manuscript table expands these to the names used below.
        canonical = {'isp': 'full', 'is': 'inf_soc', 'ip': 'inf_pres', 'sp': 'soc_pres',
                     'i': 'inf', 's': 'soc', 'p': 'pres'}
        goals = {'i': 'informative', 's': 'social', 'b': 'both'}
        polarities = {'y': 'yes', 'n': 'not'}
        words = {'t': 'terrible', 'b': 'bad', 'g': 'good', 'a': 'amazing'}
        self.source_model_labels = set()
        for row in csv.DictReader(path.read_text().splitlines()):
            if canonical.get(row['model'], row['model']) == model:
                self.source_model_labels.add(row['model'])
                key = (goals.get(row['goal'], row['goal']), int(row['true_state']),
                       polarities.get(row['positivity'], row['positivity']),
                       words.get(row['utterance'], row['utterance']))
                self.joint[key] = float(row['prob'])
                self.raw_joint[key] = float(row['raw_rounded_prob'])
                self.raw_sums[key[:2]] = float(row['raw_probability_sum'])
        if len(self.joint) != 96:
            raise ValueError('Author posterior requires all 3 goals × 4 states × 8 utterances.')
        for goal in ('both', 'informative', 'social'):
            for state in range(4):
                total = sum(v for (g, s, _, _), v in self.joint.items() if (g, s) == (goal, state))
                if not math.isclose(total, 1, abs_tol=1e-8):
                    raise ValueError(f'Author posterior probability means are not normalized: {total}')

    def forward(self, trial):
        if trial.experiment.endswith('/exp1'):
            state = int(trial.metadata['condition'].split('=')[1])
            word = re.search(r'(terrible|bad|good|amazing)\?', trial.original_prompt).group(1)
            positivity = 'not' if "wasn't" in trial.original_prompt else 'yes'
            row = self.literal[state, positivity + '_' + word]
            p = row['posterior_b1'] / (row['posterior_b1'] + row['posterior_b2'])
            ps = [p if option == 'Yes' else 1-p for option in trial.options]
            return {'probabilities': ps, 'baseline_role': 'literal_semantics_calibration_prior_not_prediction'}
        goal, state, _ = trial.metadata['condition'].split(':')
        state = int(state)
        if trial.query_tag == 'posneg_choice':
            def marginal(joint):
                return [sum(p for (g, s, polarity, word), p in joint.items()
                          if g == goal and s == state and polarity == ('not' if option == "wasn't" else 'yes')) for option in trial.options]
        else:
            def marginal(joint):
                return [sum(p for (g, s, polarity, word), p in joint.items()
                          if g == goal and s == state and word == option) for option in trial.options]
        ps, raw = marginal(self.joint), marginal(self.raw_joint)
        total = self.raw_sums[goal, state]
        return {'probabilities': ps, 'raw_rounded_marginals': raw,
                'raw_rounded_probability_sum': total,
                'normalization_correction_max_abs_probability': max(abs(a-b) for a,b in zip(ps, raw)),
                'normalization_nll_change_nats': math.log(total),
                'baseline_role': 'published_rounded_archive_normalized_readout',
                'source_posterior_model_labels': sorted(self.source_model_labels),
                'source_posterior_goal_label': {'informative': 'i', 'social': 's', 'both': 'b'}[goal],
                'source_positivity_dictionary': {'y': 'yes', 'n': 'not'},
                'source_utterance_dictionary': {'t': 'terrible', 'b': 'bad', 'g': 'good', 'a': 'amazing'},
                'marginalization': 'Mean of published rounded eight-utterance values -> explicit p/sum(p) interface conversion -> source-order polarity/adjective marginal; raw values retained, no fitting or smoothing.'}


class TsvilodubAuthorPrediction:
    """Exactly rescore the author's released human-prior RSA posterior, no refit."""
    def __init__(self):
        path = SOURCES / 'number_interpretation/human_data/best_model_preds_humanPriors.csv'
        self.price_marginals = {}
        self.source_labels = set()
        count = 0
        for row in csv.DictReader(path.read_text().splitlines()):
            key = row['object'], int(row['utterance']), int(row['state'])
            self.price_marginals[key] = self.price_marginals.get(key, 0) + float(row['probability'])
            self.source_labels.add(row['model']);count += 1
        if count != 600 or len(self.price_marginals) != 300:
            raise ValueError('Author full posterior must have 600 price×affect rows.')
        for obj in ('kettle', 'watch', 'laptop'):
            for utterance in PRICES:
                if not math.isclose(sum(self.price_marginals[obj, utterance, s] for s in PRICES), 1, abs_tol=1e-12):
                    raise ValueError('Released author posterior is not normalized.')
        self.priors = read_webppl_priors('human')

    def forward(self, trial):
        obj = 'kettle' if 'electric kettle' in trial.state else 'watch' if 'watch' in trial.state else 'laptop'
        utterance = int(re.search(r'It cost \$(\d+)', trial.state).group(1))
        if trial.experiment.endswith('/exp3b'):
            return {'reported_probability': self.priors[obj]['affect'][utterance],
                    'baseline_role': 'human_affect_prior_input_reference_not_prediction'}
        target = int(re.fullmatch(r'state_prob_(\d+)', trial.query_tag).group(1))
        return {'reported_probability': self.price_marginals[obj, utterance, target],
                'baseline_role': 'released_author_rsa_predictive_rescore',
                'artifact_model_column': sorted(self.source_labels),
                'artifact_label_note': 'Filename and author notebook cell 15 identify human-prior RSA; CSV model column says gpt-4o-mini. Source inconsistency retained, not silently relabeled.'}
