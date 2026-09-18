"""Verify executed outputs, policy consistency, image links and local-data exclusions."""
from pathlib import Path
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import pandas as pd
from fraud_utils import file_sha256, DATA_PATH, metrics


def main():
    execution = json.loads((ROOT / 'reports/execution.json').read_text())
    assert execution['all_completed'], 'A full ordered execution is required.'
    notebook_names = ['01_exploratory_analysis.ipynb', '02_fraud_model.ipynb', '03_risk_monitoring.ipynb']
    assert [row['file'] for row in execution['notebooks']] == notebook_names
    for row in execution['notebooks']:
        assert file_sha256(ROOT / row['file']) == row['sha256'], 'Notebook changed after recorded execution.'
    for name in notebook_names:
        nb = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(nb)
        cells = [c for c in nb.cells if c.cell_type == 'code']
        assert [c.execution_count for c in cells] == list(range(1, len(cells) + 1))
        assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            package, version = line.split('==')
            assert importlib.metadata.version(package) == version, f'Environment differs for {package}'
    frozen = json.loads((ROOT / 'config/frozen_policy.json').read_text())
    manifest = json.loads((ROOT / 'reports/prediction_manifest.json').read_text())
    data_manifest = json.loads((ROOT / 'reports/data_manifest.json').read_text())
    assert file_sha256(DATA_PATH) == frozen['data_sha256'] == data_manifest['sha256']
    assert file_sha256(ROOT / 'config/frozen_policy.json') == manifest['policy_sha256']
    assert file_sha256(ROOT / 'fraud_utils.py') == manifest['shared_code_sha256']
    assert file_sha256(ROOT / 'artifacts/frozen_model.joblib') == manifest['model_sha256']
    prediction_path = ROOT / 'artifacts/frozen_predictions.csv.gz'
    assert file_sha256(prediction_path) == manifest['predictions_sha256']
    predictions = pd.read_csv(prediction_path, float_precision='round_trip')
    assert predictions.row_id.is_unique
    current = predictions.loc[predictions.period.eq('Historical replay')]
    result = metrics(current.isFraud, current.score, frozen['threshold'])
    assert np.array_equal(current.alert, current.score.ge(frozen['threshold']))
    reported = json.loads((ROOT / 'reports/model_summary.json').read_text())['historical_test']
    for key in ['precision', 'recall', 'F1', 'AP', 'TP', 'FP', 'FN', 'TN']:
        assert np.isclose(result[key], reported[key]), key
    for key, value in frozen['expected_test_counts'].items():
        assert result[key] == value
    periods = pd.read_csv(ROOT / 'reports/tables/monitoring_periods.csv')
    by_type = pd.read_csv(ROOT / 'reports/tables/monitoring_by_type.csv')
    for key in ['transactions', 'fraud_count', 'alerts', 'TP', 'FP', 'FN', 'TN']:
        assert periods[key].sum() == by_type[key].sum() == result[key]
    rolling = pd.read_csv(ROOT / 'reports/tables/development_rolling_results.csv')
    assert rolling.evaluation_end.le(377).all()
    assert (rolling.fit_end < rolling.threshold_end).all()
    assert (rolling.threshold_end < rolling.evaluation_end).all()
    readme = (ROOT / 'README.md').read_text()
    image_links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', readme)
    assert 2 <= len(image_links) <= 4
    for relative_path in image_links:
        image = ROOT / relative_path
        assert image.is_file() and image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    # Check local links from each Markdown document's own directory.
    markdown_documents = [ROOT / 'README.md', ROOT / 'data/README.md', *sorted((ROOT / 'docs').glob('*.md'))]
    for document in markdown_documents:
        for destination in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            if destination.startswith(('http://', 'https://', 'mailto:', '#')):
                continue
            relative_path = destination.split('#', 1)[0]
            assert (document.parent / relative_path).exists(), f'Broken link in {document.name}: {destination}'
    # Test the actual ignore rules in an isolated empty Git directory, without initialising this project.
    with tempfile.TemporaryDirectory(prefix='fraud-ignore-check-') as temporary:
        temporary_root = Path(temporary)
        subprocess.run(['git', 'init', '--quiet', str(temporary_root)], check=True)
        (temporary_root / '.gitignore').write_text((ROOT / '.gitignore').read_text())
        for relative_path in ['data/PS_20174392719_1491204439457_log.csv', 'artifacts/frozen_predictions.csv.gz',
                              'artifacts/frozen_model.joblib']:
            result = subprocess.run(['git', '-C', str(temporary_root), 'check-ignore', '--quiet', relative_path])
            assert result.returncode == 0, relative_path
        for relative_path in ['data/README.md', 'reports/figures/monitoring_overview.png', 'README.md']:
            result = subprocess.run(['git', '-C', str(temporary_root), 'check-ignore', '--quiet', relative_path])
            assert result.returncode == 1, relative_path
    report = {'status': 'passed', 'checks': [
        'Three notebooks executed in order from fresh kernels; outputs contain no errors',
        'Installed dependency versions match requirements.txt',
        'Dataset, fixed policy, shared code, model and prediction fingerprints match',
        'Frozen confusion counts and metrics reproduce; group totals reconcile',
        'Rolling fit/threshold/evaluation periods remain in development',
        'README contains four valid local chart links' if len(image_links) == 4 else 'README chart links are valid',
        'Local Markdown links resolve in README, data instructions and analysis documents',
        'Git ignore rules exclude raw data and transaction-level artefacts',
    ], 'notebook_sha256': {name: file_sha256(ROOT / name) for name in notebook_names},
        'fresh_environment_install_tested': False}
    (ROOT / 'reports/verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
