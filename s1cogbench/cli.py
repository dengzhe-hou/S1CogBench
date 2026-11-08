"""python -m s1cogbench {list,plan,doctor,eval}

  list    the 27 evaluated modes, their adapter methods and inputs
  plan    selected modes, adapter contract and action rules without importing the adapter
  doctor  which modes the local data and runtimes can run, and why not
  eval    run modes through their original runners and scorers
          (--smoke: a few calls per mode, checked against stored values)
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from s1cogbench import ROOT, VERSION, results, smoke
from s1cogbench.modes import MODES, select
from s1cogbench.planning import describe, effective_action_rule, inspect_adapter, validate_action_rules

UNIFORM = ROOT / 'examples/uniform_reference_agent.py'
LINKED = ('data', 'external', 'literature', '.tools')
PYTHON_PACKAGES = ['numpy', 'pandas', 'scipy', 'statsmodels', 'torch', 'gym', 'gymnasium', 'matplotlib',
                   'pydantic', 'litellm', 'transformers']


def now():
    return datetime.now(timezone.utc).isoformat()


def source_root(path):
    """The directory linked from a data root: data/<x>, external/<x>, literature/<x>, .tools/<x>."""
    return '/'.join(Path(path).parts[:2])


def locate(path, data_root):
    """Where an input resolves: the repository tree first, then the data root."""
    for base in (ROOT, data_root):
        if (base / path).exists():
            return base / path
    return None


def link_data_root(data_root, modes):
    """Link each needed source directory of the data root into the repository tree.

    The original runners resolve inputs relative to the repository root. A data
    root holds the same layout (data/, external/, literature/, .tools/); only
    missing entries are linked, and an existing different entry stops the run.
    """
    data_root = data_root.resolve()
    if data_root == ROOT:
        return []
    made = []
    for name in sorted({source_root(p) for m in modes for p in m.inputs}):
        target, link = data_root / name, ROOT / name
        if not target.exists():
            continue
        if link.is_symlink() or link.exists():
            if link.resolve() != target.resolve():
                raise SystemExit(f'{link} exists and is not {target}. Remove one of them or '
                                 'drop --data-root for this source.')
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=target.is_dir())
        made.append(name)
    return made


def extra_path():
    """Directories in S1COGBENCH_EXTRA_PATH, for adapter code outside the active environment."""
    return [p for p in os.environ.get('S1COGBENCH_EXTRA_PATH', '').split(os.pathsep) if p]


def adapter_module(path):
    sys.path[:0] = [p for p in extra_path() if p not in sys.path]
    spec = importlib.util.spec_from_file_location('s1cogbench_adapter_check', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not callable(getattr(m, 'create_agent', None)):
        raise SystemExit(f'{path} must define create_agent(device).')
    return m


def cmd_plan(args):
    modes = select(args.tasks)
    report = {'selected_modes': len(modes), 'model_API_calls': 0,
              'adapter': inspect_adapter(args.adapter) if args.adapter else None,
              'modes': describe(modes, args.action_rule),
              'status': 'plan_only',
              'comparison': 'New adapter runs are user-supplied results. Check checkpoint, prompt, '
                            'readout, history, action rule and score context before comparing with paper rows.'}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for m in report['modes']:
            print(f"{m['key']:45s} rule={m['effective_action_rule']:13s} "
                  f"needs={'+'.join(m['adapter_interface'])} scored_units={m['expected_scored_units']}")
        print('Plan only. No adapter import, model/API call, dataset read, download or evaluation output.')
        print('Use --json for original input paths and full comparison contexts.')
    validate_action_rules(modes, args.action_rule)


# ---------------------------------------------------------------- list / doctor
def cmd_list(args):
    for m in MODES:
        print(f'{m.key:45s} {m.label:26s} {"+".join(m.interface):45s} n={m.scored_expected}')


def julia_problem(data_root):
    julia = locate('.tools/julia-1.4.1/bin/julia', data_root)
    if julia is None:
        return 'Julia 1.4.1 not found at .tools/julia-1.4.1/bin/julia'
    try:
        version = subprocess.run([str(julia), '--version'], capture_output=True, text=True, timeout=60).stdout
    except OSError as exc:
        return f'Julia 1.4.1 does not run: {exc}'
    if 'julia version 1.4.1' not in version:
        return f'Julia at {julia} reports {version.strip()!r}, not 1.4.1'
    depot = locate('.tools/julia-depot-1.4.1', data_root)
    if depot is not None and not os.access(depot, os.W_OK):
        return f'Julia depot {depot} is not writable'
    return None


def diagnose(modes, data_root):
    packages = {}
    for name in PYTHON_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    missing_packages = [k for k, v in packages.items() if v is None]
    julia = None
    report = []
    for m in modes:
        reasons = [f'missing {p}' for p in m.inputs if locate(p, data_root) is None
                   and not p.startswith('.tools/')]
        if m.family == 'mouselab_generative':
            julia = julia if julia is not None else (julia_problem(data_root) or '')
            if julia:
                reasons.append(julia)
        if missing_packages:
            reasons.append('Python packages missing: ' + ', '.join(missing_packages))
        report.append({'key': m.key, 'label': m.label, 'runnable': not reasons, 'reasons': reasons})
    return {'python': sys.version.split()[0], 'packages': packages, 'modes': report}


def cmd_doctor(args):
    modes = select(args.tasks)
    if args.adapter:
        a = adapter_module(args.adapter)
        print(f'adapter {args.adapter}: create_agent found; probability readout '
              f'{getattr(a, "PROBABILITY_READOUT", True)}')
    d = diagnose(modes, args.data_root.resolve())
    print(f"python {d['python']}; " + ', '.join(f'{k} {v or "MISSING"}' for k, v in d['packages'].items()))
    for r in d['modes']:
        print(f"{'ok ' if r['runnable'] else 'NO '} {r['key']:45s} {'; '.join(r['reasons'])}")
    ok = sum(r['runnable'] for r in d['modes'])
    print(f'{ok}/{len(d["modes"])} modes runnable with data root {args.data_root.resolve()}')
    print('See docs/DATA.md for how to obtain each missing input.')
    raise SystemExit(0 if ok == len(d['modes']) else 1)


# ---------------------------------------------------------------- eval
def cmd_eval(args):
    modes = select(args.tasks)
    validate_action_rules(modes, args.action_rule)
    adapter = Path(args.adapter or UNIFORM).resolve()
    if not args.smoke and args.adapter is None:
        raise SystemExit('--adapter is required (or use --smoke).')
    probabilities = getattr(adapter_module(adapter), 'PROBABILITY_READOUT', True)
    if not probabilities and any(effective_action_rule(m, args.action_rule) == 'sample' for m in modes):
        raise SystemExit('Categorical sampling requires actual legal-option probabilities. '
                         'This adapter declares PROBABILITY_READOUT=False; its text-parser placeholders '
                         'cannot be sampled as model probabilities.')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit(f'--output must be inside the repository ({ROOT}), for example runs/my-model: '
                         'the original runners refuse to write elsewhere.')
    out.mkdir(parents=True, exist_ok=False)
    data_root = args.data_root.resolve()
    linked = link_data_root(data_root, modes)
    d = diagnose(modes, data_root)
    blocked = {r['key']: r['reasons'] for r in d['modes'] if not r['runnable']}
    env = os.environ.copy()
    threads = str(args.threads)
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               NUMEXPR_NUM_THREADS=threads, JULIA_NUM_THREADS=threads, TF_NUM_INTRAOP_THREADS=threads,
               TF_NUM_INTEROP_THREADS=threads, PYTHONDONTWRITEBYTECODE='1',
               LITELLM_LOCAL_MODEL_COST_MAP='True', PYTHONPATH=os.pathsep.join([str(ROOT), *extra_path()]))
    env.setdefault('MPLCONFIGDIR', str(out / '.runtime/matplotlib'))
    config = {'s1cogbench_version': VERSION, 'started_utc': now(), 'adapter': str(adapter),
              'model_label': args.model_label, 'device': args.device, 'action_rule': args.action_rule,
              'threads': args.threads, 'smoke': args.smoke, 'data_root': str(data_root),
              'linked_into_repository': linked, 'modes': [m.key for m in modes],
              'blocked_before_start': blocked, 'python': d['python'], 'packages': d['packages'],
              'probability_readout': probabilities}
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    tasks = list(dict.fromkeys(m.task for m in modes if m.key not in blocked))
    # Same order as the CPU validation driver: the slow Instrumental scorer last.
    tasks = [t for t in tasks if t != 'instrumental_learning'] + (
        ['instrumental_learning'] if 'instrumental_learning' in tasks else [])
    for task in tasks:
        keys = [m.key for m in modes if m.task == task and m.key not in blocked]
        (out / task).mkdir()
        command = [sys.executable, '-B', '-m', 's1cogbench.worker', '--modes', ','.join(keys),
                   '--adapter', str(adapter), '--output', str(out), '--device', args.device,
                   '--action-rule', args.action_rule, '--model-label', args.model_label,
                   '--threads', threads] + (['--smoke', '--smoke-calls', str(args.smoke_calls)] if args.smoke else [])
        started = time.monotonic()
        with (out / task / 'worker.log').open('w') as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f'{task}: exit {code} after {time.monotonic() - started:.1f}s', flush=True)
    finish(out, modes, blocked, args, adapter, probabilities)


def finish(out, modes, blocked, args, adapter, probabilities):
    states, rows = [], []
    for m in modes:
        path = out / m.task / m.mode / 'mode_result.json'
        if m.key in blocked:
            r = {'key': m.key, 'status': 'not_run', 'error': '; '.join(blocked[m.key])}
        elif path.exists():
            r = json.loads(path.read_text())
        else:
            r = {'key': m.key, 'status': 'failed', 'error': f'worker ended before writing {path}; '
                                                            f'see {m.task}/worker.log'}
        r.setdefault('device', args.device)
        r.setdefault('action_rule', args.action_rule)
        r.setdefault('adapter', str(adapter))
        states.append(r)
        if not args.smoke:
            rows.append(results.row(m, r, args.model_label, probabilities))
    summary = {'finished_utc': now(), 'modes': len(modes),
               'complete': sum(s['status'] == 'complete' for s in states),
               'results': [{k: s.get(k) for k in ('key', 'status', 'inputs', 'scored', 'primary',
                                                    'seconds', 'error')} for s in states]}
    if args.smoke:
        summary['smoke'] = smoke_check(out, modes, states, adapter, args.write_smoke_expected, args.smoke_calls)
    else:
        results.write(out, rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=True) + '\n')
    for s in summary['results']:
        p = s.get('primary') or {}
        print(f"{s['status']:9s} {s['key']:45s} {p.get('value_status', '')} {p.get('value', '')} "
              f"{s.get('error') or ''}")
    print(f"{summary['complete']}/{len(modes)} modes complete; outputs in {out}")
    if args.smoke:
        print(summary['smoke']['verdict'])
    ok = summary['complete'] == len(modes) and (not args.smoke or summary['smoke']['passed'])
    raise SystemExit(0 if ok else 1)


def smoke_check(out, modes, states, adapter, write_expected, calls):
    values = {}
    for m, s in zip(modes, states):
        p = s.get('primary') or {}
        values[m.key] = smoke.observed(s, {'value': p.get('value'), 'n': p.get('n')} if p else None)
    if write_expected:
        failed = [m.key for m, s in zip(modes, states) if s['status'] != 'complete']
        if failed or len(modes) != 27:
            raise SystemExit(f'Expected values need all 27 modes complete; not complete: {failed}')
        smoke.EXPECTED.write_text(json.dumps({
            'adapter': 'examples/uniform_reference_agent.py', 'limit_per_mode': smoke.LIMIT,
            'limits': smoke.LIMITS, 'recorded_utc': now(), 'modes': values}, indent=1, allow_nan=False) + '\n')
    if adapter != UNIFORM.resolve() or calls != smoke.LIMIT:
        return {'passed': True, 'compared': 0,
                'verdict': 'Smoke run finished. Stored values exist only for the uniform reference adapter at the '
                           f'default {smoke.LIMIT} calls, so nothing was compared.'}
    expected = json.loads(smoke.EXPECTED.read_text())['modes']
    problems, compared = [], 0
    for m, s in zip(modes, states):
        if s['status'] == 'not_run':
            continue
        compared += 1
        problems += smoke.compare(m.key, values[m.key], expected)
    (out / 'smoke_check.json').write_text(json.dumps({'compared_modes': compared, 'problems': problems},
                                                    indent=2) + '\n')
    verdict = (f'SMOKE PASSED: {compared} modes match the stored values.' if not problems else
               f'SMOKE FAILED: {len(problems)} differences in {compared} modes; see smoke_check.json.')
    return {'passed': not problems, 'compared': compared, 'problems': problems, 'verdict': verdict}


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m s1cogbench', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List the 27 modes.').set_defaults(func=cmd_list)
    q = sub.add_parser('plan', help='Inspect modes and adapter syntax without model or data access.')
    q.add_argument('--tasks', nargs='+', default=['all'])
    q.add_argument('--adapter', type=Path)
    q.add_argument('--action-rule', choices=['native', 'sample'], default='native')
    q.add_argument('--json', action='store_true', help='Print input paths, targets and comparison contexts.')
    q.set_defaults(func=cmd_plan)
    for name, func, text in (('doctor', cmd_doctor, 'Check which modes the local data can run.'),
                             ('eval', cmd_eval, 'Run modes through the original runners.')):
        q = sub.add_parser(name, help=text)
        q.add_argument('--tasks', nargs='+', default=['all'],
                       help='all, task IDs or task:mode keys (see `list`).')
        q.add_argument('--data-root', type=Path, default=ROOT,
                       help='Directory with data/, external/, literature/, .tools/ (default: repository).')
        q.add_argument('--adapter', type=Path, help='Python file defining create_agent(device).')
        q.set_defaults(func=func)
        if name == 'eval':
            q.add_argument('--output', type=Path, required=True, help='New output directory.')
            q.add_argument('--device', default='cpu')
            q.add_argument('--model-label', default='user_model')
            q.add_argument('--action-rule', choices=['native', 'sample'], default='native',
                           help='Autonomous modes: native choice (argmax when none) or sampling.')
            q.add_argument('--threads', type=int, default=1)
            q.add_argument('--smoke', action='store_true',
                           help='A few calls per mode; default adapter is the uniform reference.')
            q.add_argument('--smoke-calls', type=int, default=smoke.LIMIT,
                           help='Adapter calls per mode in a smoke run.')
            q.add_argument('--write-smoke-expected', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.func(args)
