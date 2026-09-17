"""Exercise notebook logging with real child/grandchild output, without network."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def check_stage_output(tmp_path, exit_code):
    notebook = json.loads((ROOT / 'notebooks/TaxaLens_v0.9_Multisource_PoC_Colab.ipynb').read_text())
    setup = next(''.join(c['source']) for c in notebook['cells'] if 'def stage(' in ''.join(c['source']))
    stage = next(n for n in ast.parse(setup).body if isinstance(n, ast.FunctionDef) and n.name == 'stage')
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    child = "import sys; print('GBIF progress', flush=True); print('Причина: пример ошибки', file=sys.stderr); sys.exit(" + str(exit_code) + ")"
    (repo / 'scripts/run_multisource_poc_v09.py').write_text(
        'import subprocess, sys\n'
        f'sys.exit(subprocess.run([sys.executable, "-u", "-c", {child!r}]).returncode)\n'
    )
    data = tmp_path / 'data'
    data.mkdir()
    manifest = data / 'bioscan.parquet'
    manifest.write_bytes(b'previous BIOSCAN manifest')
    env = dict(Path=Path, sys=sys, os=os, subprocess=subprocess, REPO=str(repo), DATA_ROOT=str(data),
               BIOSCAN_ROOT=str(data / 'raw'), BIOSCAN_MANIFEST=str(manifest))
    exec(compile(ast.Module(body=[stage], type_ignores=[]), '<notebook-stage>', 'exec'), env)
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        for _ in range(2):
            if exit_code:
                try:
                    env['stage']('download', '--source', 'gbif', '--reuse-existing-bioscan')
                except RuntimeError as error:
                    assert 'Причина: пример ошибки' in str(error)
                else:
                    raise AssertionError('Failed subprocess did not stop the stage')
            else:
                env['stage']('download', '--source', 'gbif', '--reuse-existing-bioscan')
    logs = list(data.glob('taxalens_download_*.log'))
    assert len(logs) == 2
    for log in logs:
        assert 'GBIF progress' in log.read_text() and 'Причина: пример ошибки' in log.read_text()
    output = output_buffer.getvalue()
    assert 'GBIF progress' in output and 'Причина: пример ошибки' in output
    assert ('✓ Этап завершён:' in output) == (exit_code == 0)
    assert manifest.read_bytes() == b'previous BIOSCAN manifest'


class NotebookStageOutputTests(unittest.TestCase):
    def test_success(self):
        with tempfile.TemporaryDirectory() as temp:
            check_stage_output(Path(temp), 0)

    def test_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            check_stage_output(Path(temp), 1)


if __name__ == '__main__':
    unittest.main()
