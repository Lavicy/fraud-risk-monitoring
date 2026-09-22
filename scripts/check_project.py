"""Check the executed learning notebooks and their saved results."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_helpers import evaluate


def main():
    execution = json.loads((ROOT / 'reports/execution.json').read_text())
    assert execution['all_completed']
    names = ['01_exploratory_analysis.ipynb', '02_fraud_model.ipynb', '03_risk_monitoring.ipynb']
    assert [row['file'] for row in execution['notebooks']] == names
    for row in execution['notebooks']:
        path = ROOT / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1))
        assert not any(o.output_type == 'error' for cell in cells for o in cell.outputs)
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            package, version = line.split('==')
            assert importlib.metadata.version(package) == version
    windows = pd.read_csv(ROOT / 'reports/tables/model_windows.csv')
    assert windows.first_step.tolist() == [1, 324]
    assert windows.last_step.tolist() == [323, 377]
    policy = json.loads((ROOT / 'reports/model_summary.json').read_text())
    assert policy['features'] == ['log_amount', 'is_transfer']
    predictions = pd.read_csv(ROOT / 'artifacts/learning_validation_predictions.csv.gz', float_precision='round_trip')
    assert predictions.source_row.is_unique and predictions.step.between(324, 377).all()
    assert np.array_equal(predictions.alert, predictions.score.ge(policy['threshold']))
    result = evaluate(predictions.isFraud, predictions.score, policy['threshold'])
    for key, value in result.items():
        assert np.isclose(value, policy['validation'][key], equal_nan=True), key
    thresholds = pd.read_csv(ROOT / 'reports/tables/threshold_comparison.csv')
    best = thresholds.sort_values(['F1', 'threshold'], ascending=False).iloc[0]
    assert best.threshold == policy['threshold']
    candidates = pd.read_csv(ROOT / 'reports/tables/model_comparison.csv')
    assert candidates.loc[candidates.validation_AP.idxmax(), 'model'] == policy['model']
    for name in ['monitoring_blocks.csv', 'monitoring_by_type.csv']:
        table = pd.read_csv(ROOT / 'reports/tables' / name)
        for key in ['transactions', 'fraud', 'alerts', 'TP', 'FP', 'FN', 'TN']:
            assert table[key].sum() == result[key]
    documents = [ROOT / 'README.md', ROOT / 'data/README.md', *sorted((ROOT / 'docs').glob('*.md'))]
    for document in documents:
        for destination in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            if not destination.startswith(('https://', 'http://', '#')):
                assert (document.parent / destination.split('#')[0]).exists(), destination
    for path in ['data/PS_20174392719_1491204439457_log.csv',
                 'artifacts/learning_validation_predictions.csv.gz', '.local_history/advanced-2026-09-18/README.md']:
        assert subprocess.run(['git', 'check-ignore', '--quiet', path], cwd=ROOT).returncode == 0
    report = {'status': 'passed', 'core_tests': 4, 'notebooks': 3,
              'checks': ['Clean-kernel execution and saved outputs', 'Pinned installed versions',
                         'Original non-overlapping development boundaries', 'Two permitted model features',
                         'Validation-only predictions and consistent threshold/metrics',
                         'Monitoring totals', 'Local documentation links', 'Data and backup exclusions'],
              'independent_new_test': False, 'fresh_environment_install_tested': False}
    (ROOT / 'reports/verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
