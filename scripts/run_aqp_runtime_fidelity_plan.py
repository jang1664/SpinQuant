"""Persistently execute the approved full plan, check exits, analyze, archive evidence.

This is a local process supervisor, not a scheduler or remote service. It stops on
any failed stage and leaves status.json plus stderr/stdout logs for diagnosis.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def completed(root):
    return len(list(root.glob('*/r*/b*/complete.json')))


def event(args,message):
    line=f'- {stamp()}: {message}'
    print(line,flush=True)
    with Path(args.progress).open('a') as f: f.write(line+'\n')


def status(args,stage,state='running',**extra):
    data=dict(updated=stamp(),pid=os.getpid(),stage=stage,state=state,
              smoke_completed=completed(Path(args.smoke)),main_completed=completed(Path(args.main)),
              expected_runs_per_model=72,**extra)
    Path(args.status).write_text(json.dumps(data,indent=2)+'\n')


def run_logged(args,label,command):
    logs=Path(args.logs); logs.mkdir(parents=True,exist_ok=True)
    status(args,label); event(args,f'Start {label}: `{subprocess.list2cmdline(command)}`')
    with (logs/f'{label}.stderr.log').open('x') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=err)
        tee=subprocess.Popen(['tee',str(logs/f'{label}.stdout.log')],stdin=process.stdout)
        process.stdout.close()
        last=0
        while process.poll() is None:
            if time.monotonic()-last>60:
                status(args,label,child_pid=process.pid)
                last=time.monotonic()
            time.sleep(10)
        exit_code=process.wait(); tee_code=tee.wait()
    if exit_code or tee_code:
        raise RuntimeError(f'{label}: program exit={exit_code}, tee exit={tee_code}; see {logs}')
    event(args,f'Completed {label}; program exit=0, tee exit=0.')


def wait_existing(args,pid,label,needle):
    if not pid: return
    event(args,f'Waiting for existing {label} process {pid}.')
    while True:
        try: command=Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError: break
        if needle.encode() not in command: break
        status(args,'waiting_'+label,child_pid=pid)
        time.sleep(30)


def archive(args,root,analysis_id):
    analysis=root/analysis_id
    destination=Path(args.reports)/root.name/analysis_id
    destination.mkdir(parents=True,exist_ok=False)
    for name in ['REPORT.md','summary.json','summary.csv','ablation_summary.csv','margin_flip_rates.csv',
                 'runtime_std.csv','cancellation.csv','condition_validation.json','analysis_source.py','paired_fidelity_metrics.py']:
        shutil.copy2(analysis/name,destination/name)
    for name in ['manifest.json','weight_validation.json']:
        if (root/name).exists(): shutil.copy2(root/name,destination/name)
    shutil.copytree(analysis/'figures',destination/'figures')
    index={str(p.relative_to(root)):json.loads(p.read_text()) for p in sorted(root.glob('*/r*/b*/complete.json'))}
    (destination/'artifact_index.json').write_text(json.dumps({'raw_root':str(root),'runs':index},indent=2)+'\n')
    (destination/'RAW_DATA.md').write_text(f'Full-vocabulary logits and fixed corpus: `{root}`.\n\n'
        'SHA256 hashes for every raw shard are in artifact_index.json. The corpus hash is in manifest.json.\n')
    event(args,f'Archived reviewable report, figures, summaries, manifests, and raw shard hashes at `{destination}`.')


def work(args):
    smoke,main=Path(args.smoke),Path(args.main)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    wait_existing(args,args.wait_smoke_pid,'smoke','measure_aqp_runtime_fidelity.py')
    # Also validates W4 metadata, then skips already completed immutable outputs.
    run_logged(args,'smoke_finish',[sys.executable,'measure_aqp_runtime_fidelity.py','run','--output-dir',str(smoke)])
    if completed(smoke)!=72: raise RuntimeError('smoke sweep is incomplete')
    run_logged(args,'smoke_analysis',[sys.executable,'summarize_aqp_runtime_fidelity.py','--experiment',str(smoke),'--analysis-id','final-analysis'])
    archive(args,smoke,'final-analysis')
    wait_existing(args,args.wait_prepare_pid,'main_prepare','measure_aqp_runtime_fidelity.py')
    if not (main/'manifest.json').exists(): raise RuntimeError('main preparation did not produce a manifest')
    manifest=json.loads((main/'manifest.json').read_text())
    if (manifest['windows'],manifest['prompt_length'],manifest['positions'],manifest['repeats'],manifest['batch_sizes'])!=(128,512,128,3,[1,2,4,8]):
        raise ValueError('approved full main scope changed')
    run_logged(args,'main_c0',[sys.executable,'measure_aqp_runtime_fidelity.py','run','--output-dir',str(main),
                             '--conditions','AQP16,A8,AQ8,AQP8','--batch-sizes','1','--repeat-ids','0'])
    run_logged(args,'main_c0_analysis',[sys.executable,'summarize_aqp_runtime_fidelity.py','--experiment',str(main),'--analysis-id','c0-analysis'])
    archive(args,main,'c0-analysis')
    run_logged(args,'main_sweep',[sys.executable,'measure_aqp_runtime_fidelity.py','run','--output-dir',str(main)])
    if completed(main)!=72: raise RuntimeError('main sweep is incomplete')
    run_logged(args,'main_analysis',[sys.executable,'summarize_aqp_runtime_fidelity.py','--experiment',str(main),'--analysis-id','final-analysis'])
    result=json.loads((main/'final-analysis/summary.json').read_text())
    if result['status']!='complete': raise RuntimeError('analysis reported missing runs')
    archive(args,main,'final-analysis')
    event(args,'Full 3B validation sweep and approved 8B main experiment completed; reports and figures archived. No manuscript edits.')
    status(args,'finished','complete')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['smoke','main','progress','reports','logs','status']: p.add_argument('--'+name,required=True)
    p.add_argument('--wait-smoke-pid',type=int)
    p.add_argument('--wait-prepare-pid',type=int)
    p.add_argument('--detach',action='store_true')
    args=p.parse_args()
    if args.detach:
        log=Path(args.logs); log.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve())]+[s for s in sys.argv[1:] if s!='--detach']
        with (log/'supervisor.log').open('x') as stdout:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stdout,start_new_session=True)
        print(json.dumps({'supervisor_pid':process.pid,'status':args.status,'log':str(log/'supervisor.log')}),flush=True)
        return
    try: work(args)
    except Exception as error:
        event(args,f'Execution stopped: {type(error).__name__}: {error}. Incomplete work is not marked complete.')
        status(args,'failed','failed',error=str(error))
        traceback.print_exc()
        raise


if __name__=='__main__': main()
