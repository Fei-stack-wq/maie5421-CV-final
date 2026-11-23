"""Formal A/B preparation must fail closed before starting any training."""
import json
from unittest.mock import patch

import pytest

from tools.training import run_formal_ab as runner


def test_four_configs_align_with_completed_c_protocol():
    plan = runner.build_plan(runner.ROOT / 'runs/plan-test')
    assert plan['selected_runs'] == list(runner.RUN_IDS)
    assert plan['formal_models_completed'] == 0
    for entry in plan['entries']:
        cfg = entry['config']
        assert cfg['run_kind'] == 'formal' and cfg['protocol_confirmed']
        assert cfg['train_limit'] is None and cfg['val_limit'] is None
        assert cfg['consistency_weight'] == 0 and cfg['amp_init_scale'] == 1024
        assert cfg['batch'] * cfg['accumulation_steps'] == 16


@pytest.mark.parametrize('selected', [['C_5421'], ['A_5421', 'A_5421']])
def test_bad_run_selection(selected):
    with pytest.raises(ValueError):
        runner.select_runs(selected)


def test_outside_output_rejected(tmp_path):
    with pytest.raises(ValueError, match='inside'):
        runner.build_plan(tmp_path)


def test_bare_runs_root_rejected():
    with pytest.raises(ValueError, match='strictly'):
        runner.build_plan(runner.ROOT / 'runs')


def test_existing_outputs_rejected(tmp_path):
    repo = tmp_path / 'repo'
    out = repo / 'runs/ab/A_5421'
    out.mkdir(parents=True)
    (out / 'best.pt').write_text('existing')
    with patch.object(runner, 'load_config', return_value=object()), patch.object(runner, 'check_protocol'):
        with pytest.raises(FileExistsError):
            runner.build_plan(repo / 'runs/ab', ['A_5421'], repo=repo)


def test_symlink_output_rejected(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'runs/ab').mkdir(parents=True)
    other = tmp_path / 'other'
    other.mkdir()
    (repo / 'runs/ab/A_5421').symlink_to(other, target_is_directory=True)
    with patch.object(runner, 'load_config', return_value=object()), patch.object(runner, 'check_protocol'):
        with pytest.raises(FileExistsError):
            runner.build_plan(repo / 'runs/ab', ['A_5421'], repo=repo)


def test_default_cli_never_preflights_or_executes(capsys):
    with patch('sys.argv', ['run_formal_ab', '--runs-root', str(runner.ROOT / 'runs/plan-test')]), \
            patch.object(runner, 'preflight') as preflight, patch.object(runner, 'execute_plan') as execute:
        runner.main()
    preflight.assert_not_called()
    execute.assert_not_called()
    assert json.loads(capsys.readouterr().out)['formal_models_completed'] == 0


def test_execute_requires_clean_exact_commit():
    with pytest.raises(ValueError, match='full reviewed'):
        runner.check_git('short')
    with patch.object(runner.subprocess, 'check_output', side_effect=['a' * 40, ' M src/train.py']):
        with pytest.raises(ValueError, match='clean'):
            runner.check_git('a' * 40)
    with patch.object(runner.subprocess, 'check_output', side_effect=['b' * 40, '']):
        with pytest.raises(ValueError, match='exactly'):
            runner.check_git('a' * 40)


def test_execute_requires_cuda():
    with patch.object(runner.torch.cuda, 'is_available', return_value=False):
        with pytest.raises(RuntimeError, match='CUDA'):
            runner.check_environment()


def test_data_evidence_requires_matching_qa_hash(tmp_path):
    repo = tmp_path / 'repo'
    artifacts = repo / 'artifacts/ouyang'
    artifacts.mkdir(parents=True)
    (artifacts / 'qa_received.json').write_text('{}')
    (artifacts / 'qa_visual_review.json').write_text('{}')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('a,b\n')
    qa = tmp_path / 'qa.json'
    qa.write_text('{}')
    ready = tmp_path / 'ready.json'
    ready.write_text(json.dumps({'ready_for_training': True, 'verified_images': 12100,
                                'fresh_qa_sha256': 'wrong'}))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='supplied QA'):
        runner.check_data_evidence(SimpleNamespace(data_manifest=str(manifest)), qa, ready, repo)


def test_execution_failure_preserves_receipt_and_does_not_retry(tmp_path):
    entry = {'run_id': 'A_5421', 'config_path': 'config.yaml', 'config_sha256': 'sha',
             'output_dir': str(tmp_path / 'A_5421'), 'command': ['training-command']}
    plan = {'runs_root': str(tmp_path), 'selected_runs': ['A_5421'], 'entries': [entry]}
    with patch.object(runner, 'check_git'), patch.object(runner, 'sha256', return_value='sha'), \
            patch.object(runner.subprocess, 'run', side_effect=RuntimeError('mock training failed')) as run:
        with pytest.raises(RuntimeError, match='mock training'):
            runner.execute_plan(plan, {'source_commit': 'a' * 40})
    assert run.call_count == 1
    receipt = json.loads(next((tmp_path.parent / (tmp_path.name + '-attempts')).glob('*/execution.json')).read_text())
    assert not receipt['ok'] and receipt['verified_models'] == 0
    assert 'mock training failed' in receipt['error']


def test_successful_attempt_reports_stay_outside_models(tmp_path):
    root = tmp_path / 'runs/ab'
    entry = {'run_id': 'A_5421', 'config_path': 'config.yaml', 'config_sha256': 'sha',
             'output_dir': str(root / 'A_5421'), 'command': ['training-command']}
    plan = {'runs_root': str(root), 'selected_runs': ['A_5421'], 'entries': [entry]}
    def fake_run(command, **kwargs):
        if '--output' in command:
            output = runner.Path(command[command.index('--output') + 1])
            assert not output.is_relative_to(root)
            output.write_text(json.dumps({'selected_verified': True, 'runs': {
                'A_5421': {'budget_review_required': True}}}))
    with patch.object(runner, 'check_git'), patch.object(runner, 'sha256', return_value='sha'), \
            patch.object(runner.subprocess, 'run', side_effect=fake_run):
        receipt_path = runner.execute_plan(plan, {'source_commit': 'a' * 40})
    receipt = json.loads(receipt_path.read_text())
    assert receipt['ok'] and receipt['verified_models'] == 1
    assert receipt['budget_review_required'] is True
    assert not receipt['all_four_verified_in_this_attempt']
