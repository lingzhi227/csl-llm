"""Owner evidence/reset failure tests only, without importing the SDK."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import model_generation_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def exercise(self, directory, *, stop_failure=False, repeat_failure=False):
        (directory/'out').mkdir()
        closed, calls = [], []
        refs = [SimpleNamespace(steps=[{}], close=lambda: closed.append(True)) for _ in range(4)]
        def stop():
            calls.append('stop')
            (directory/'out.core').write_bytes(b'fault-injection core placeholder, not SDK evidence')
            if stop_failure:
                raise RuntimeError('stop failure')
        runner = SimpleNamespace(load=lambda: calls.append('load'), run=lambda: calls.append('run'),
            launch=lambda *a, **k: calls.append('initialize'), stop=stop)
        def schedule(io, references, *, precision, record):
            for request in range(4):
                actual = {name: np.array([0.0], np.float32) for name in
                          ('layer_outputs', 'model_input', 'generated', 'logits', 'winner', 'best')}
                if repeat_failure and request == 3:
                    actual['logits'][0] = -0.0
                io.save(actual)
                record(request, 0, actual, {'normal_only': True})
            return dict(requests=[], invocation=4, runtime_accepted=False)
        with patch.object(runtime, 'open_references', return_value=refs), \
                patch.object(runtime, 'ModelRuntimeIO', side_effect=lambda *a, **kw: SimpleNamespace(save=kw['save'])), \
                patch.object(runtime, 'run_requests', side_effect=schedule):
            result = runtime.run(runner, directory=directory, reference_directory=directory,
                precision={}, memcpy_data_type=None, memcpy_order=None, stage=lambda _: None)
        return result, calls, closed

    def test_all_steps_and_stop_evidence_preserved_without_claiming_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, calls, closed = self.exercise(root)
            self.assertFalse(result['runtime_accepted'])
            self.assertEqual(calls, ['load', 'run', 'initialize', 'stop'])
            self.assertEqual(len(closed), 4)
            receipt = json.loads((root/'runtime-evidence.json').read_text())
            self.assertFalse(receipt['runtime_accepted'])
            self.assertEqual(len(receipt['files']), 10)
            for name, digest in receipt['files'].items():
                self.assertEqual(runtime.sha(root/name), digest)

    def test_stop_failure_still_records_core_and_normal_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, 'stop failure'):
                self.exercise(root, stop_failure=True)
            receipt = json.loads((root/'runtime-evidence.json').read_text())
            self.assertIn('out.core', receipt['files'])
            self.assertIn('request-03/step-0000/normal-report.json', receipt['files'])
            self.assertFalse(receipt['runtime_accepted'])

    def test_bitwise_reset_failure_preserves_partial_readback_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'not bit-exact'):
                self.exercise(root, repeat_failure=True)
            receipt = json.loads((root/'runtime-evidence.json').read_text())
            self.assertIn('request-03/step-0000/normal-d2h.npz', receipt['files'])
            self.assertNotIn('request-03/step-0000/normal-report.json', receipt['files'])
            self.assertNotIn('requests.json', receipt['files'])
            self.assertIn('out.core', receipt['files'])

    def test_primary_failure_survives_stop_logging_and_reference_close_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'out').mkdir()
            calls = []
            def fail(message):
                raise RuntimeError(message)
            refs = [SimpleNamespace(close=lambda: fail('first close failed')),
                    SimpleNamespace(close=lambda: calls.append('second closed'))]
            runner = SimpleNamespace(load=lambda: fail('original load failure'),
                stop=lambda: calls.append('SDK stop attempted'))
            def stage(name):
                if name == 'stop':
                    raise OSError('stop log unavailable')
            with patch.object(runtime, 'open_references', return_value=refs):
                with self.assertRaisesRegex(RuntimeError, 'original load failure'):
                    runtime.run(runner, directory=root, reference_directory=root, precision={},
                        memcpy_data_type=None, memcpy_order=None, stage=stage)
            self.assertEqual(calls, ['SDK stop attempted', 'second closed'])
            receipt = json.loads((root/'runtime-evidence.json').read_text())
            self.assertEqual(receipt['primary_error']['message'], 'original load failure')
            self.assertEqual([e['operation'] for e in receipt['cleanup_errors']],
                             ['stage-stop', 'reference-close-0'])


if __name__ == '__main__':
    unittest.main()
