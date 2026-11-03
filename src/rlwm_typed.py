"""CF12 original data with a declared block-history, stimulus-ID text interface.

All original rows are retained. Model scoring uses the author's RT > .15 mask;
history retains earlier recorded events in the current independent block,
including fast answers and omissions. This is not a visual/timed replication.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

from src.psych_category_tasks import CategoryTrial
from src.psych_typed import score_prediction
from src.psych_extended import summarize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'data/rlwm/CollinsFrank_2012_EJoN.csv'
PROTOCOL_VERSION = 'cf12-stimulus-id-complete-block-history-v1'
KEYS = ('1', '2', '3')
QUESTION = {'type': 'choice', 'instructions': 'Which button do you choose for the current stimulus?',
            'criteria': {k: f'Choose button {k}.' for k in KEYS}}


@dataclass(frozen=True)
class RLWMTrial(CategoryTrial):
    source_metadata: dict

    @property
    def session_id(self):
        return f'CF12:participant={self.participant}:block={self.source_metadata["block"]}'

    @property
    def id(self):
        return f'CF12:source_row={self.source_row}'

    def identifiers(self):
        row = super().identifiers()
        row.update(self.source_metadata)
        row.update(session_id_origin='original participant and block',
                   trial_id_origin='zero-based original CSV data-row position')
        return row


def read_source(path=DEFAULT_SOURCE):
    rows = []
    with Path(path).open(newline='') as stream:
        for index, raw in enumerate(csv.DictReader(stream)):
            row = {key: float(value) for key, value in raw.items()}
            for key in ('subno', 'block', 'ns', 'time', 'stimseq', 'imageseq', 'folderseq',
                        'iterseq', 'corAseq', 'choice', 'key', 'cor', 'rew', 'pcor'):
                row[key] = int(row[key])
            row['source_row'] = index
            row['author_rt_eligible'] = row['rt'] > .15
            row['legal_response'] = row['choice'] in (1, 2, 3)
            if row['author_rt_eligible'] and not row['legal_response']:
                raise ValueError('Author-eligible row has no legal recorded response.')
            if row['choice'] == -1:
                if (row['cor'], row['rew'], row['rt']) != (-1, -1, -1):
                    raise ValueError('Unrecognized omission feedback in original CF12 data.')
            elif row['cor'] != int(row['choice'] == row['corAseq']) or row['rew'] != row['cor']:
                raise ValueError('Original action, deterministic correctness and reward disagree.')
            rows.append(row)
    return rows


def build_trials(rows):
    trials, history, last_session = [], [], None
    previous_stimulus = {}
    last_time = 0
    for row in rows:
        participant, block, stimulus = row['subno'], row['block'], row['stimseq']
        session = (participant, block)
        if session != last_session:
            history, previous_stimulus, last_time = [], {}, 0
            last_session = session
        if row['time'] != last_time + 1 or not 1 <= stimulus <= row['ns']:
            raise ValueError('Original block order/stimulus range is inconsistent.')
        last_time = row['time']
        before = ('Learn which button is correct for each stimulus. Available buttons are 1, 2, 3. '
                  'Each stimulus has one fixed correct button in this block. '
                  'Different stimuli may have the same correct button; learning one mapping does not reveal another. '
                  'Correct feedback means reward 1; incorrect feedback means reward 0. '
                  'This block has a new set of stimuli: ' + ', '.join(f'S{i}' for i in range(1, row['ns'] + 1)) + '.\n'
                  'The following is the complete recorded history of this block. No response means no response or feedback was recorded.\n')
        state = before + ''.join(history) + f'Trial {row["time"]}: stimulus S{stimulus}. Choose button '
        if row['author_rt_eligible']:
            metadata = {key: row[key] for key in ('block', 'ns', 'time', 'stimseq', 'imageseq',
                'folderseq', 'iterseq', 'pcor', 'key', 'rt')}
            metadata.update(author_rt_eligible=True,
                source_delay_since_last_correct=None if math.isnan(row['delay']) else row['delay'],
                trials_since_previous_stimulus=(row['time'] - previous_stimulus[stimulus]
                                                if stimulus in previous_stimulus else None),
                history_events=len(history), history_scope='all previous original rows within this block')
            trials.append(RLWMTrial('rlwm', 'CollinsFrank_2012_EJoN', participant, row['source_row'],
                row['time'] - 1, 'learning', 'three_button_choice', row['iterseq'] - 1,
                len(state), state, KEYS, str(row['choice']), str(row['corAseq']), QUESTION,
                f'S{stimulus}', {'correct': row['cor'], 'reward': row['rew']}, metadata))
        # Append only after the current prediction boundary; hidden corAseq is never rendered.
        if row['legal_response']:
            feedback = 'correct' if row['cor'] else 'incorrect'
            history.append(f'Trial {row["time"]}: stimulus S{stimulus}, button {row["choice"]}, feedback {feedback}.\n')
        else:
            history.append(f'Trial {row["time"]}: stimulus S{stimulus}, no response.\n')
        previous_stimulus[stimulus] = row['time']
    return trials


def load_trials(source=DEFAULT_SOURCE):
    return build_trials(read_source(source))


def inventory(rows, trials):
    return {'task': 'rlwm', 'protocol_version': PROTOCOL_VERSION,
        'source_rows': len(rows), 'participants': len({r['subno'] for r in rows}),
        'sessions': len({(r['subno'], r['block']) for r in rows}), 'responses': len(trials),
        'excluded_from_author_scoring': sum(not r['author_rt_eligible'] for r in rows),
        'omissions': sum(not r['legal_response'] for r in rows),
        'fast_recorded_answers': sum(r['legal_response'] and not r['author_rt_eligible'] for r in rows),
        'set_sizes': sorted({r['ns'] for r in rows}),
        'history': 'Complete ordered current-block history, including all fast responses and omissions. Reset at original independent block boundaries. No within-block truncation.',
        'split': 'No official ML split exists. Frozen zero-shot evaluation on all author-eligible CF12 rows; no fitting or resplitting.',
        'source_retention': 'All original CSV rows and exclusion reasons preserved in source_rows.jsonl.',
        'adaptation': 'New stimulus-ID textual three-action interface; not original visual/timed presentation.',
        'baseline_scope': 'Archived seven-model fits are in-sample and use RT-filtered histories. Separate reference; not equal-training or identical-history superiority comparison.',
        'interpretation': 'External-history-assisted response prediction; not human working-memory capacity or autonomous reward.'}


def score(trial, prediction):
    row = score_prediction(trial, prediction)
    row.update(native_rating_absolute_error=None,
               objective_target_semantics='Hidden original correct action, scored after prediction only.')
    return row


def save_source_rows(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            value = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                     for k, v in row.items()}
            value['scoring_status'] = ('author_eligible' if row['author_rt_eligible'] else
                                       'omission' if not row['legal_response'] else 'fast_answer')
            stream.write(json.dumps(value, allow_nan=False) + '\n')
