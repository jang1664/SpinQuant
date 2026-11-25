"""Persistent local supervisor for the approved free-generation metric sweep."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(args, stage, state='running', **extra):
    data = dict(updated=now(), state=state, stage=stage, supervisor_pid=os.getpid(),
        smoke_completed=len(list(Path(args.smoke).glob('*/r*/complete.json'))),
        main_completed=len(list(Path(args.main).glob('*/r*/complete.json'))),
        expected_joint_runs_per_model=18, runtime_batch_settings_per_joint_run=4, **extra)
    Path(args.status).write_text(json.dumps(data, indent=2)+'\n')


def event(args, message):
    line = f'- {now()}: {message}'
    print(line, flush=True)
    with Path(args.progress).open('a') as stream:
        stream.write(line+'\n')


def logged(args, label, command):
    logs = Path(args.logs)
    event(args, f'Start {label}: `{subprocess.list2cmdline(command)}`')
    with (logs / f'{label}.stderr.log').open('x') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err)
        tee = subprocess.Popen(['tee', str(logs / f'{label}.stdout.log')], stdin=process.stdout)
        process.stdout.close()
        while process.poll() is None:
            write_status(args, label, child_pid=process.pid)
            time.sleep(10)
        code, tee_code = process.wait(), tee.wait()
    if code or tee_code:
        raise RuntimeError(f'{label}: program exit={code}, tee exit={tee_code}; see {logs}')
    event(args, f'Completed {label}; program exit=0, tee exit=0.')


def work(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    Path(args.logs).mkdir(parents=True, exist_ok=True)
    for key in ['smoke', 'main']:
        root = Path(getattr(args, key))
        manifest = json.loads((root / 'manifest.json').read_text())
        expected = (8,128,16,3) if key == 'smoke' else (128,512,128,3)
        if tuple(manifest[k] for k in ['windows','prompt_length','positions','repeats']) != expected:
            raise ValueError('approved workload changed')
        if manifest['batch_sizes'] != [1,2,4,8] or len(manifest['conditions']) != 6:
            raise ValueError('approved runtime/condition sweep changed')
        destination = Path(args.reports) / root.name
        for repeat in range(3):
            logged(args, f'{key}_r{repeat}', [sys.executable, 'measure_aqp_common_prefix.py', 'run',
                '--output', str(root), '--repeats', str(repeat)])
            logged(args, f'{key}_analysis_r{repeat}', [sys.executable, 'summarize_aqp_common_prefix.py',
                '--experiment', str(root), '--output', str(destination)])
        summary = json.loads((destination / 'summary.json').read_text())
        if summary['status'] != 'complete':
            raise RuntimeError(f'{key} analysis incomplete')
        event(args, f'{key} common-prefix sweep and analysis complete: `{destination / "REPORT.md"}`.')
    write_status(args, 'finished', 'complete')
    event(args, 'Free-generation common-prefix metric complete for 3B validation and full 8B workload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['smoke','main','reports','progress','status','logs']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        Path(args.logs).mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve())]+[x for x in sys.argv[1:] if x != '--detach']
        with (Path(args.logs) / 'supervisor.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid, status=args.status)), flush=True)
        return
    try:
        work(args)
    except Exception as error:
        write_status(args, 'failed', 'failed', error=str(error))
        event(args, f'Common-prefix execution failed: {error}.')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
