"""Run the approved fresh zero-shot sweep and publish its local report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=['fp_base','aqp16','aqp8']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def status(args,stage,state='running',**extra):
    Path(args.status).write_text(json.dumps(dict(updated=stamp(),state=state,stage=stage,
        supervisor_pid=os.getpid(),
        smoke_completed=sum((Path(args.smoke)/c/'complete.json').exists() for c in CONDITIONS),
        main_completed=sum((Path(args.main)/c/'complete.json').exists() for c in CONDITIONS),
        expected_conditions=3,**extra),indent=2)+'\n')


def event(args,message):
    text=f'- {stamp()}: {message}'
    print(text,flush=True)
    with Path(args.progress).open('a') as stream: stream.write(text+'\n')


def logged(args,label,command):
    logs=Path(args.logs)
    event(args,f'Start {label}: `{subprocess.list2cmdline(command)}`')
    with (logs/(label+'.stderr.log')).open('x') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=err)
        tee=subprocess.Popen(['tee',str(logs/(label+'.stdout.log'))],stdin=child.stdout)
        child.stdout.close()
        while child.poll() is None:
            status(args,label,child_pid=child.pid)
            time.sleep(10)
        code,tee_code=child.wait(),tee.wait()
    if code or tee_code:
        raise RuntimeError(f'{label}: program exit={code}, tee exit={tee_code}')
    event(args,f'Completed {label}; program exit=0, tee exit=0.')


def work(args):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    for phase in ['smoke','main']:
        root=Path(getattr(args,phase))
        m=json.loads((root/'manifest.json').read_text())
        if m['model_name']!='meta-llama/Llama-3.1-8B' or m['batch_size']!=4 or m['max_length']!=2048:
            raise ValueError('approved model/runtime changed')
        if m['eval_limit'] != (2 if phase=='smoke' else None):
            raise ValueError('unexpected evaluation limit')
        for condition in CONDITIONS:
            logged(args,f'{phase}_{condition}',[sys.executable,'measure_aqp_zero_shot.py','run',
                '--output',str(root),'--condition',condition])
            complete=json.loads((root/condition/'complete.json').read_text())
            if complete['questions'] != (10 if phase=='smoke' else 15357):
                raise ValueError('not all expected questions evaluated')
            h=hashlib.sha256((root/(condition+'.json')).read_bytes()).hexdigest()
            if h!=complete['output_sha256']:
                raise ValueError('result checksum mismatch')
        event(args,f'{phase}: all three fresh model conditions completed and checked.')
    logged(args,'main_analysis',[sys.executable,'summarize_zero_shot_top1_fidelity.py',
        '--input-root',args.main,'--output',args.reports])
    destination=Path(args.reports)
    shutil.copy2(Path(args.main)/'manifest.json',destination/'execution_manifest.json')
    for condition in CONDITIONS:
        shutil.copy2(Path(args.main)/condition/'validation.json',destination/(condition+'_validation.json'))
        shutil.copy2(Path(args.main)/condition/'complete.json',destination/(condition+'_complete.json'))
    status(args,'finished','complete')
    event(args,f'Llama-3.1 8B zero-shot top-1 probability comparison complete: `{destination / "REPORT.md"}`.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['smoke','main','logs','status','progress','reports']: p.add_argument('--'+name,required=True)
    p.add_argument('--detach',action='store_true')
    args=p.parse_args(); Path(args.logs).mkdir(parents=True,exist_ok=True)
    if args.detach:
        command=[sys.executable,str(Path(__file__).resolve())]+[x for x in sys.argv[1:] if x!='--detach']
        with (Path(args.logs)/'supervisor.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,status=args.status)),flush=True)
    else:
        try: work(args)
        except Exception as error:
            status(args,'failed','failed',error=str(error))
            event(args,f'Zero-shot execution failed: {error}')
            traceback.print_exc()
            raise
