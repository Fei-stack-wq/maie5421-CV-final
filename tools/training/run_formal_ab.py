"""Plan, preflight and sequentially execute TONG's four formal A/B runs.

The default is a read-only plan. --execute requires an exact clean Git commit,
fresh data QA, matching readiness evidence, cached weights and CUDA. No scheduler
jobs, downloads, account operations or automatic retries are performed here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models import ResNet18_Weights

from src.train import load_config, make_datasets
from tools.training.check_received_qa import check_reports
from tools.training.verify_formal_ab import check_protocol

ROOT = Path(__file__).resolve().parents[2]
RUN_IDS = ('A_5421', 'B_5421', 'A_2026', 'B_2026')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_runs(requested=None):
    selected = list(requested) if requested else list(RUN_IDS)
    if len(selected) != len(set(selected)) or any(x not in RUN_IDS for x in selected):
        raise ValueError('select distinct run IDs from A_5421, B_5421, A_2026, B_2026')
    return selected


def build_plan(runs_root, selected=None, repo=ROOT):
    repo = Path(repo).resolve()
    runs_root = Path(runs_root).resolve()
    if runs_root == repo / 'runs' or not runs_root.is_relative_to(repo / 'runs'):
        raise ValueError('runs-root must be strictly inside this checkout\'s ignored runs/ directory')
    entries = []
    for run_id in select_runs(selected):
        condition, seed = run_id.split('_')
        config_path = repo / 'configs/tong' / f'{condition.lower()}_{seed}.yaml'
        config = load_config(config_path)
        reference = load_config(repo / 'configs/wang' / f'c_{seed}.yaml')
        check_protocol(config, run_id, reference)
        output = runs_root / run_id
        if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            raise FileExistsError(f'{run_id} already has output; never overwrite or silently retrain it')
        entries.append({'run_id': run_id, 'config_path': str(config_path),
                        'config_sha256': sha256(config_path), 'config': asdict(config),
                        'output_dir': str(output),
                        'command': [sys.executable, '-m', 'src.train', '--config', str(config_path),
                                    '--output-dir', str(output)]})
    return {'purpose': 'formal A/B execution plan; no completed models implied',
            'selected_runs': [x['run_id'] for x in entries], 'runs_root': str(runs_root),
            'formal_models_completed': 0, 'entries': entries}


def check_git(expected_commit, repo=ROOT):
    if not expected_commit or not re.fullmatch(r'[0-9a-f]{40}', expected_commit):
        raise ValueError('--execute requires --expected-commit with the full reviewed Git SHA')
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if actual != expected_commit or dirty:
        raise ValueError('checkout must be clean and at exactly --expected-commit')
    return actual


def check_data_evidence(config, qa_path, ready_path, repo=ROOT):
    if not qa_path or not ready_path:
        raise ValueError('--execute requires --qa and --data-ready from this machine\'s data preparation')
    qa = json.loads(Path(qa_path).read_text())
    ready = json.loads(Path(ready_path).read_text())
    ref_path = Path(repo) / 'artifacts/ouyang/qa_received.json'
    reference = json.loads(ref_path.read_text())
    review = json.loads((Path(repo) / 'artifacts/ouyang/qa_visual_review.json').read_text())
    manifest = Path(config.data_manifest).read_bytes()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    if ready.get('ready_for_training') is not True or ready.get('verified_images') != 12100:
        raise ValueError('full data readiness evidence is required')
    if ready.get('fresh_qa_sha256') != sha256(qa_path):
        raise ValueError('readiness evidence does not describe the supplied QA file')
    if ready.get('manifest_sha256') != manifest_hash or qa.get('manifest_sha256') != manifest_hash:
        raise ValueError('QA/readiness evidence does not match the current manifest bytes')
    if sha256(ref_path) != review['source_qa_sha256']:
        raise ValueError('versioned QA reference hash mismatch')
    if hashlib.sha256(manifest.replace(b'\r\n', b'\n')).hexdigest() != review['manifest_sha256']:
        raise ValueError('manifest differs from the reviewed data version')
    check_reports(qa, reference, review)
    return {'manifest_sha256': manifest_hash, 'qa_sha256': sha256(qa_path),
            'readiness_sha256': sha256(ready_path)}


def check_environment(repo=ROOT):
    if not torch.cuda.is_available():
        raise RuntimeError('formal A/B execution requires an authorized CUDA GPU; plan mode needs none')
    if sys.version_info[:2] != (3, 11):
        raise ValueError('use Python 3.11 to match the recorded formal C environment')
    versions = {}
    for line in (Path(repo) / 'requirements-training.txt').read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith('#'):
            continue
        name, expected = requirement.split('==')
        actual = importlib.metadata.version(name)
        versions[name] = actual
        if actual.split('+')[0] != expected:
            raise ValueError(f'{name} must match requirements-training.txt ({expected}); found {actual}')
    if torch.version.cuda != '12.1':
        raise ValueError('use the shared cu121 PyTorch runtime before comparing with the completed C models')
    filename = Path(urlparse(ResNet18_Weights.IMAGENET1K_V1.url).path).name
    weights = Path(torch.hub.get_dir()) / 'checkpoints' / filename
    prefix = filename.rsplit('-', 1)[1].split('.')[0]
    if not weights.is_file() or not sha256(weights).startswith(prefix):
        raise ValueError('verified ImageNet weights must already be present in TORCH_HOME; prepare them first')
    return {'python': sys.version.split()[0], 'packages': versions, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'weights_sha256': sha256(weights)}


def preflight(plan, expected_commit, qa_path, ready_path):
    if Path.cwd().resolve() != ROOT:
        raise ValueError('execute this module from the repository root')
    commit = check_git(expected_commit)
    environment = check_environment()
    config = load_config(plan['entries'][0]['config_path'])
    evidence = check_data_evidence(config, qa_path, ready_path)
    # Rehash the actual files now; a historical QA JSON alone cannot establish readiness.
    train, val, metadata, audit = make_datasets(config)
    if len(train) != 6000 or len(val) != 1500 or audit.get('sha256_verified') != 12000:
        raise ValueError('formal runs require all 6000 train, 1500 clean-val and 12000 verified manifest images')
    return {'source_commit': commit, 'environment': environment, 'data': evidence,
            'audit': audit, 'preprocessing': metadata}


def save_status(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def execute_plan(plan, evidence):
    # The caller has completed preflight before any output or process is created.
    root = Path(plan['runs_root'])
    attempt = root.parent / (root.name + '-attempts') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    status_path = attempt / 'execution.json'
    status = {'purpose': 'formal A/B execution receipt', 'ok': False,
              'plan': plan, 'preflight': evidence, 'runs': {}, 'verified_models': 0}
    save_status(status_path, status)
    try:
        for entry in plan['entries']:
            run_id = entry['run_id']
            # Detect changes made after the initial preflight, before each run.
            check_git(evidence['source_commit'])
            if sha256(entry['config_path']) != entry['config_sha256']:
                raise ValueError('config changed after preflight')
            output = Path(entry['output_dir'])
            if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
                raise FileExistsError(f'{run_id} output appeared after preflight')
            status['runs'][run_id] = {'status': 'training'}
            save_status(status_path, status)
            with (attempt / f'{run_id}.train.txt').open('x') as stream:
                subprocess.run(entry['command'], check=True, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status['runs'][run_id]['status'] = 'trained_pending_verification'
            save_status(status_path, status)
            verification = attempt / f'{run_id}.verification.json'
            command = [sys.executable, '-m', 'tools.training.verify_formal_ab', '--runs-root', str(root),
                       '--run-id', run_id, '--device', 'cuda', '--output', str(verification)]
            with (attempt / f'{run_id}.verify.txt').open('x') as stream:
                subprocess.run(command, check=True, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            report = json.loads(verification.read_text())
            if report.get('selected_verified') is not True:
                raise ValueError('verification report did not verify the selected run')
            budget_review = report['runs'][run_id].get('budget_review_required', False)
            status['runs'][run_id] = {'status': 'verified', 'verification': str(verification),
                                      'budget_review_required': budget_review}
            status['verified_models'] += 1
            save_status(status_path, status)
        status['ok'] = True
        status['budget_review_required'] = any(r.get('budget_review_required', False) for r in status['runs'].values())
        status['all_four_verified_in_this_attempt'] = set(plan['selected_runs']) == set(RUN_IDS)
    except Exception as exc:
        status['error'] = f'{type(exc).__name__}: {exc}'
        save_status(status_path, status)
        raise
    save_status(status_path, status)
    return status_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=Path('runs/tong_formal_ab'))
    parser.add_argument('--run-id', action='append', choices=RUN_IDS, help='repeat to select unstarted runs; default all four')
    parser.add_argument('--execute', action='store_true', help='run the selected formal experiments after all preflight checks')
    parser.add_argument('--expected-commit', help='full SHA of the reviewed, clean checkout; required for execution')
    parser.add_argument('--qa', type=Path)
    parser.add_argument('--data-ready', type=Path)
    args = parser.parse_args()
    plan = build_plan(args.runs_root, args.run_id)
    if not args.execute:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    evidence = preflight(plan, args.expected_commit, args.qa, args.data_ready)
    print(execute_plan(plan, evidence))


if __name__ == '__main__':
    main()
