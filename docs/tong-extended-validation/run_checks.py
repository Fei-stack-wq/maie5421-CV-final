"""Reproducible CPU integration checks, not formal project experiments.

Run from repository root: python - < docs/tong-extended-validation/run_checks.py
Creates a new ignored runs/ directory. Defaults to the original run name;
use --output-dir runs/a_fresh_name to avoid overwriting prior results.
This safe reproduction driver omits the IPC test that was blocked in the original run.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.models import BinaryResNet18
from src.train import TrainConfig, load_config, make_datasets, seed_everything
from src.training.checkpoint import load_model, read_checkpoint
from src.training.data import PREPROCESS, read_manifest, verify_declared_hashes
from src.training.engine import validate

ROOT = Path.cwd()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', default='runs/tong_extended_validation_v1')
args = parser.parse_args()
OUT = (ROOT / args.output_dir).resolve()
if not OUT.is_relative_to(ROOT / 'runs'):
    raise ValueError('validation output must be inside the ignored runs/ directory')
if OUT.exists():
    raise FileExistsError(f'Choose a new output directory; refusing to overwrite {OUT.name}')
OUT.mkdir(parents=True)
started = time.monotonic()
torch.set_num_threads(2)
report = {'purpose': 'CPU workflow/integration validation only; not formal experiments',
          'formal_models_completed': 0, 'protocol_confirmed': False, 'checks': {}, 'runs': {},
          'environment': {'scope': 'current cloud execution environment, not the user personal computer',
             'python': platform.python_version(), 'torch': str(torch.__version__),
             'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
             'pillow': PIL.__version__, 'pyyaml': yaml.__version__,
             'cuda_available': torch.cuda.is_available(), 'torch_cuda_runtime': torch.version.cuda,
             'cpu_count': os.cpu_count(),
             'ram_gib': round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3, 1),
             'initial_disk_free_gib': round(shutil.disk_usage(ROOT).free / 1024**3, 1)},
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_report():
    (OUT/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

def command(args, log):
    before = time.monotonic()
    with log.open('w') as f:
        result = subprocess.run([sys.executable, *args], stdout=f, stderr=subprocess.STDOUT, timeout=300)
    entry = {'command': ['python', *args], 'returncode': result.returncode,
             'seconds': time.monotonic()-before}
    if result.returncode != 0:
        raise RuntimeError(f'Command failed; inspect {log.relative_to(ROOT)}')
    return entry

def identical(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError(f'Tensor mismatch: {path}')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f'Key mismatch: {path}')
        for key in a:
            identical(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'Length mismatch: {path}')
        for i, (x, y) in enumerate(zip(a, b)):
            identical(x, y, f'{path}[{i}]')
    elif a != b:
        raise AssertionError(f'Value mismatch: {path}: {a} != {b}')

def compare_runs(first, second):
    a, b = read_checkpoint(first/'last.pt'), read_checkpoint(second/'last.pt')
    checked = ('model_state','optimizer_state','scaler_state','best_epoch','best_val_auc','bad_epochs','epoch')
    for key in checked:
        identical(a[key], b[key], key)
    history_a = [{k:v for k,v in row.items() if k!='seconds'} for row in a['history']]
    history_b = [{k:v for k,v in row.items() if k!='seconds'} for row in b['history']]
    identical(history_a, history_b, 'history_without_wall_time')
    return {'equal': True, 'fields': [*checked, 'history_without_wall_time']}

def run_case(name, config, resume=False):
    config.validate()
    conf = OUT/f'{name}.yaml'
    conf.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    destination = OUT/name
    args = ['-m','src.train','--config',str(conf.relative_to(ROOT)),
            '--output-dir',str(destination.relative_to(ROOT))]
    if resume:
        args += ['--resume',str((destination/'last.pt').relative_to(ROOT))]
    log = OUT/f'{name}{"_resume" if resume else ""}.txt'
    execution = command(args, log)
    report['runs'].setdefault(name, {'executions': []})['executions'].append(execution)
    report['runs'][name]['summary'] = json.loads((destination/'summary.json').read_text())
    save_report()
    print(f'Completed {name}{" resume" if resume else ""}', flush=True)
    return destination

report['checks']['unit_tests'] = command(['-m','unittest','discover','-s','tests','-v'], OUT/'unit-tests.txt')
report['checks']['compileall'] = command(['-m','compileall','-q','src','tests'], OUT/'compileall.txt')
report['checks']['cli_help'] = command(['-m','src.train','--help'], OUT/'cli-help.txt')

# Run the existing dataset verifier in an isolated mirror to preserve FEI's report.
original_audit_hash = sha(ROOT/'Images/verification.json')
audit_root = OUT/'dataset-audit'
(audit_root/'Images').mkdir(parents=True)
(audit_root/'tools/dataset').mkdir(parents=True)
for source in (ROOT/'Images').iterdir():
    if source.name != 'verification.json':
        (audit_root/'Images'/source.name).symlink_to(source, target_is_directory=source.is_dir())
shutil.copy2(ROOT/'tools/dataset/verify_dataset.py', audit_root/'tools/dataset/verify_dataset.py')
report['checks']['full_dataset_audit'] = command(
    [str((audit_root/'tools/dataset/verify_dataset.py').relative_to(ROOT))], OUT/'dataset-audit.txt')
audit = json.loads((audit_root/'Images/verification.json').read_text())
if not audit['ok'] or audit['bad_file_count'] or sha(ROOT/'Images/verification.json') != original_audit_hash:
    raise AssertionError('Dataset audit failed or original report changed')
report['checks']['full_dataset_audit'].update({'ok': True, 'files_checked': audit['files_checked'],
    'bad_files': audit['bad_file_count'], 'cross_split_shared_images': audit['cross_split_shared_images'],
    'original_report_unchanged': True, 'perceptual_near_duplicates_checked': False})

fixture = read_manifest(ROOT/'Images/fixture/manifest_fixture.csv', ROOT/'Images')
fixture_audit = verify_declared_hashes(fixture, require_hashes=True)
for row in fixture:
    with Image.open(row['resolved_path']) as im:
        im.load()
        assert im.size == (int(row['width']),int(row['height']))
        assert im.format == row['format']
formal_rows = read_manifest(ROOT/'Images/manifest.csv', ROOT/'Images')
assert not ({r['sha256'] for r in fixture} & {r['sha256'] for r in formal_rows})
fixture_audit.update({'decoded': len(fixture), 'exact_overlap_with_formal': 0})
report['checks']['fixture_audit'] = fixture_audit
save_report()
print('Completed full dataset and fixture audit', flush=True)

base = replace(load_config('examples/tong/pilot_a.yaml'), epochs=2, amp=False, device='cpu')
for seed in (5421, 2026):
    for condition in ('A','B','C'):
        run_case(f'{condition}_{seed}', replace(base, seed=seed, condition=condition))

run_case('C_zero_5421', replace(base, condition='C', consistency_weight=0.0))
report['checks']['B_equals_C_lambda_zero'] = compare_runs(OUT/'B_5421', OUT/'C_zero_5421')
run_case('A_resumed_5421', replace(base, epochs=1))
run_case('A_resumed_5421', base, resume=True)
report['checks']['real_resnet_epoch_resume'] = compare_runs(OUT/'A_5421', OUT/'A_resumed_5421')
# Multiprocessing was blocked by this cloud environment's IPC socket permissions.
# Do not retry or change security settings here. Use num_workers=0.
report['unverified'] = ['num_workers>0', 'CUDA AMP/memory/throughput', 'formal team integration']

# All actual run outputs must reload to exactly the saved best validation CSV.
for name, entry in report['runs'].items():
    folder=OUT/name
    config=load_config(folder/'config.yaml')
    _, val_data, _, _ = make_datasets(config)
    model, metadata=load_model(folder/'best.pt')
    assert metadata['preprocessing'] == PREPROCESS
    metrics, predictions=validate(model,DataLoader(val_data,batch_size=config.batch),torch.device('cpu'))
    with (folder/'val_predictions.csv').open() as f:
        recorded=list(csv.DictReader(f))
    assert len(predictions) == len(recorded)
    assert [p[0] for p in predictions] == [p['image_id'] for p in recorded]
    difference=max(abs(p[2]-float(old['p_ai'])) for p,old in zip(predictions,recorded))
    assert difference == 0
    entry['reload_check']={'images':len(predictions),'max_absolute_probability_difference':difference,
        'best_checkpoint_sha256':sha(folder/'best.pt'),'validation_metrics':metrics}
    # Trainable affine parameters are not frozen, but every BN buffer is.
    seed_everything(config.seed,config.deterministic)
    initial=BinaryResNet18(weights=config.weights)
    final=read_checkpoint(folder/'last.pt')['model_state']
    initial_state=initial.state_dict()
    frozen=[key for key in initial_state if not key.startswith(('backbone.layer4.','backbone.fc.'))
            or key.endswith(('running_mean','running_var','num_batches_tracked'))]
    for key in frozen:
        assert torch.equal(initial_state[key],final[key]),key
    assert not torch.equal(initial_state['backbone.fc.weight'],final['backbone.fc.weight'])
    entry['freeze_check']={'unchanged_frozen_tensors':len(frozen),'all_bn_buffers_fixed':True,
                           'head_weights_updated':True}
    del model, initial, final, initial_state
    save_report()

report['available_cpu_checks_passed']=True
report['ok']=False  # CUDA, multiprocessing and formal integration remain unverified.
report['elapsed_seconds']=time.monotonic()-started
save_report()
print(json.dumps({'available_cpu_checks_passed':True,'runs':len(report['runs']),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
