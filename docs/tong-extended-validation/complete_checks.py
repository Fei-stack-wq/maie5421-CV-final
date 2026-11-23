"""Finish read-only checks on completed CPU runs after a blocked worker test.

Historical recovery helper for the original interrupted run ONLY.
It requires runs/tong_extended_validation_v1/B_workers2_5421.txt and the
completed checkpoints from that original run. New reproductions use only
run_checks.py; do not run this helper afterward.
Never retries the multiprocessing operation or overwrites model checkpoints.
"""
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models import BinaryResNet18
from src.train import load_config, seed_everything
from src.training.checkpoint import load_model, read_checkpoint
from src.training.data import PREPROCESS, ManifestPilotDataset, balanced_subset, read_manifest, validate_train_val
from src.training.engine import validate

OUT=Path('runs/tong_extended_validation_v1')
report=json.loads((OUT/'report.json').read_text())
torch.set_num_threads(2)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def identical(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):
            raise AssertionError(f'Tensor mismatch: {path}')
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:
            identical(a[key],b[key],f'{path}.{key}')
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):
            identical(x,y,f'{path}[{i}]')
    elif a!=b:
        raise AssertionError(f'Value mismatch: {path}')

def compare_runs(first,second):
    a,b=read_checkpoint(first/'last.pt'),read_checkpoint(second/'last.pt')
    fields=('model_state','optimizer_state','scaler_state','best_epoch','best_val_auc','bad_epochs','epoch')
    for key in fields:
        identical(a[key],b[key],key)
    ha=[{k:v for k,v in row.items() if k!='seconds'} for row in a['history']]
    hb=[{k:v for k,v in row.items() if k!='seconds'} for row in b['history']]
    identical(ha,hb,'history_without_wall_time')
    return {'equal':True,'fields':[*fields,'history_without_wall_time']}

report['checks']['real_resnet_epoch_resume']=compare_runs(OUT/'A_5421',OUT/'A_resumed_5421')
worker_log=(OUT/'B_workers2_5421.txt').read_text()
if 'PermissionError: [Errno 1] Operation not permitted' not in worker_log:
    raise ValueError('Expected worker blocker evidence not found; inspect before recording status')
report['blocked_checks']={
    'num_workers_2':{'status':'blocked_by_environment','operation':'PyTorch multiprocessing resource_sharer socket',
        'error':'PermissionError: [Errno 1] Operation not permitted',
        'attempt_stopped':True,'training_completed':False,
        'not_verified':['workers=0/2 full training equality','workers=0/2 sample/view equality'],
        'supported_setting':'num_workers=0'},
    'cuda':{'status':'unavailable','not_verified':['GPU AMP','GPU memory','GPU throughput','GPU determinism']},
    'formal_training':{'status':'pending_team_integration','final_A_B_models_completed':0,
        'requires':['OUYANG formal data and paired views','shared seeds and training settings','target-machine integration']}}
rows=read_manifest('Images/manifest.csv','Images')
val_rows=validate_train_val(rows)['val']
for name,entry in report['runs'].items():
    folder=OUT/name
    config=load_config(folder/'config.yaml')
    assert config.run_kind=='pilot' and config.dataset_factory is None
    val_data=ManifestPilotDataset(balanced_subset(val_rows,config.val_limit,config.seed),config.seed)
    model,metadata=load_model(folder/'best.pt')
    assert metadata['preprocessing']==PREPROCESS
    metrics,predictions=validate(model,DataLoader(val_data,batch_size=config.batch),torch.device('cpu'))
    with (folder/'val_predictions.csv').open() as f:
        saved=list(csv.DictReader(f))
    assert len(predictions)==len(saved)
    assert [p[0] for p in predictions]==[p['image_id'] for p in saved]
    difference=max(abs(p[2]-float(old['p_ai'])) for p,old in zip(predictions,saved))
    assert difference==0
    entry['reload_check']={'images':len(predictions),'max_absolute_probability_difference':difference,
        'best_checkpoint_sha256':sha(folder/'best.pt'),'validation_metrics':metrics}
    seed_everything(config.seed,config.deterministic)
    initial=BinaryResNet18(weights=config.weights)
    initial_state=initial.state_dict()
    final=read_checkpoint(folder/'last.pt')['model_state']
    frozen=[key for key in initial_state if not key.startswith(('backbone.layer4.','backbone.fc.'))
            or key.endswith(('running_mean','running_var','num_batches_tracked'))]
    for key in frozen:
        assert torch.equal(initial_state[key],final[key]),key
    assert not torch.equal(initial_state['backbone.fc.weight'],final['backbone.fc.weight'])
    entry['freeze_check']={'unchanged_frozen_tensors':len(frozen),'all_bn_buffers_fixed':True,
        'head_weights_updated':True}
    del model,initial,initial_state,final
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Reload and freeze checks passed: {name}',flush=True)
report['status']='passed_available_single_process_CPU_checks_with_documented_blockers'
report['ok']=False  # Do not present blocked multiprocessing/CUDA as a full pass.
report['available_cpu_checks_passed']=True
report['completed_pilot_runs']=len(report['runs'])
report['completed_model_reloads']=len(report['runs'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'available_cpu_checks_passed':True,'pilot_runs':len(report['runs']),'blocked':['multiprocessing','CUDA','formal integration']}),flush=True)
