"""The numeric report parser, and the Minitaur nonliteral-number reports it re-reads."""
import csv
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('social_numeric_models', ROOT / 'src/social_numeric_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parse = module.parse_percentage


def test_plain_reports():
    for text, want in [('50', .5), ('50%', .5), (' 12.5 % ', .125), ('100', 1.0), ('0', 0.0)]:
        assert parse(text) == want


def test_closing_marker_ends_the_report():
    for text in ['50>>', '50.0%»', '50.0%».\n\nPlease provide the probability as a percentage (0-100']:
        assert parse(text) == .5


def test_other_text_stays_invalid():
    for text in ['abc', '50 percent', 'about 50', '101', '', '050', '50 and then >>']:
        assert parse(text) is None


def test_minitaur_reports_reproduce_the_table():
    rows = list(csv.DictReader(open(ROOT / 'results/aggregate/minitaur_nonliteral_reparsed.csv', newline='')))
    assert len(rows) == 330
    errors = [abs(100 * parse(r['completion']) - float(r['human_mean'])) for r in rows]
    assert math.isclose(sum(errors) / len(errors), 28.52972727272727, abs_tol=1e-9)
