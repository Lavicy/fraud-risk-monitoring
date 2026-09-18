"""Execute the portfolio sequentially, each notebook in a fresh current-Python kernel."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['01_exploratory_analysis.ipynb', '02_fraud_model.ipynb', '03_risk_monitoring.ipynb']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebooks', nargs='+', choices=NOTEBOOKS, default=NOTEBOOKS,
                        help='Optional recovery run; the default executes all three in order.')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
    import nbformat
    from nbclient import NotebookClient
    run_report = {'python': sys.version.split()[0], 'notebooks': [], 'all_completed': False}
    report_path = ROOT / 'reports' / 'execution.json'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(run_report, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='fraud-notebook-') as temporary:
        temp = Path(temporary)
        kernel = temp / 'kernels' / 'fraud-project'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python (fraud portfolio)', 'language': 'python',
        }))
        os.environ['JUPYTER_PATH'] = str(temp) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temp / 'runtime')
        os.environ['IPYTHONDIR'] = str(temp / 'ipython')
        os.environ['MPLCONFIGDIR'] = str(temp / 'matplotlib')
        for name in args.notebooks:
            path = ROOT / name
            notebook = nbformat.read(path, as_version=4)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    compile(cell.source, f'{name}:cell-{index}', 'exec')
                    cell.outputs = []
                    cell.execution_count = None
            start = time.perf_counter()

            def progress(cell, cell_index, **kwargs):
                if cell.cell_type == 'code':
                    print(f'{name}: executing cell {cell_index}', flush=True)

            def completed(cell, cell_index, **kwargs):
                if cell.cell_type == 'code':
                    for output in cell.outputs:
                        if output.output_type == 'stream':
                            print(output.text[:1500], flush=True)
                nbformat.write(notebook, path)

            client = NotebookClient(notebook, timeout=1800, kernel_name='fraud-project',
                                    resources={'metadata': {'path': str(ROOT)}},
                                    on_cell_start=progress, on_cell_executed=completed)
            try:
                client.execute()
            finally:
                nbformat.write(notebook, path)
            nbformat.validate(notebook)
            code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
            assert all(cell.execution_count is not None for cell in code_cells)
            assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
            run_report['notebooks'].append({'file': name, 'code_cells': len(code_cells),
                                            'seconds': round(time.perf_counter() - start, 2),
                                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                            'status': 'passed'})
            report_path.write_text(json.dumps(run_report, indent=2) + '\n')
            print(f'{name}: complete', flush=True)
    run_report['all_completed'] = args.notebooks == NOTEBOOKS
    report_path.write_text(json.dumps(run_report, indent=2) + '\n')
    print('Requested notebook execution completed. See reports/execution.json.', flush=True)


if __name__ == '__main__':
    main()
