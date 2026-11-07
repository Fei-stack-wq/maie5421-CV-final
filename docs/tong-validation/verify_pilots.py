import csv, hashlib, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.train import load_config, make_datasets
from src.training.checkpoint import load_model
from src.training.engine import validate

torch.set_num_threads(2)
for condition in ('A', 'B'):
    directory = Path(f'runs/tong_verified_{condition}_5421')
    config = load_config(directory / 'config.yaml')
    _, val_data, _, audit = make_datasets(config)
    model, metadata = load_model(directory / 'best.pt')
    metrics, actual = validate(model, DataLoader(val_data, batch_size=config.batch), torch.device('cpu'))
    with (directory / 'val_predictions.csv').open() as stream:
        expected = list(csv.DictReader(stream))
    assert len(actual) == len(expected)
    assert [row[0] for row in actual] == [row['image_id'] for row in expected]
    difference = max(abs(row[2] - float(saved['p_ai'])) for row, saved in zip(actual, expected))
    assert difference == 0.0
    digest = hashlib.sha256()
    with (directory / 'best.pt').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    result = {'condition': condition, 'purpose': 'workflow validation only; not formal results',
              'reloaded_images': len(actual), 'max_absolute_probability_difference': difference,
              'checkpoint_sha256': digest.hexdigest(), 'checkpoint_bytes': (directory/'best.pt').stat().st_size,
              'validation_metrics': metrics, 'data_audit': audit,
              'code_commit': metadata['provenance']['commit'],
              'working_tree_dirty_during_training': metadata['provenance']['working_tree_dirty']}
    (directory / 'reload_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
