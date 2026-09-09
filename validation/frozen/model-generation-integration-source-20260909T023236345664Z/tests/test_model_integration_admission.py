"""Receipt mutation tests with synthetic files; no SDK acceptance evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import model_integration_admission as admission


def write(path, value):
    path.write_text(json.dumps(value)+'\n')


class AdmissionTests(unittest.TestCase):
    def fixture(self, root):
        (root/'runtime').mkdir()
        (root/'driver.py').write_text('synthetic test fixture, not SDK source')
        pins = {'driver.py': admission.sha(root/'driver.py')}
        config = dict(prompt=[151644], generation_limit=1, assembly_execution_sha256='assembly')
        write(root/'config.json', config)
        for name in ['normal-d2h.npz', 'normal-report.json', 'actual.npz', 'protocol.json', 'out.core']:
            (root/'runtime'/name).write_text('synthetic receipt mutation fixture')
        write(root/'runtime/runtime-evidence.json', dict(primary_error=None, cleanup_errors=[]))
        result = dict(success=True, normal=dict(passed=True), core=dict(passed=True, coordinates=40530),
            assembly_execution_sha256='assembly',
            output_sha256={p.name: admission.sha(p) for p in (root/'runtime').iterdir()})
        write(root/'results.json', result)
        write(root/'manifest.json', dict(sdk_sha256='sdk', files=dict(pins, **{'config.json':admission.sha(root/'config.json')})))
        execution = dict(success=True, sdk_sha256='sdk', manifest_sha256=admission.sha(root/'manifest.json'),
            results_sha256=admission.sha(root/'results.json'), observed_process_identities={'7':'100'},
            after_cleanup_identities={'7':None})
        write(root/'execution.json', execution)
        return pins

    def call(self, root, pins, **kw):
        with patch.object(admission, 'FIRST_FILES', pins):
            return admission.first_single_token(root,
                assembly_execution_sha256=kw.get('assembly', 'assembly'), sdk_sha256='sdk')

    def test_receipt_chain_and_same_assembly_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pins = self.fixture(root)
            result = self.call(root, pins)
            self.assertEqual(result['execution_sha256'], admission.sha(root/'execution.json'))
            with self.assertRaises(ValueError):
                self.call(root, pins, assembly='different')

    def test_rehashed_driver_cannot_change_reviewed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pins = self.fixture(root)
            (root/'driver.py').write_text('changed')
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['files']['driver.py']=admission.sha(root/'driver.py');write(root/'manifest.json',manifest)
            execution=json.loads((root/'execution.json').read_text())
            execution['manifest_sha256']=admission.sha(root/'manifest.json');write(root/'execution.json',execution)
            with self.assertRaises(ValueError):self.call(root,pins)

    def test_missing_cleanup_or_changed_actual_core_rejected(self):
        for mutation in ('cleanup', 'core'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);pins=self.fixture(root)
                if mutation=='cleanup':
                    execution=json.loads((root/'execution.json').read_text());execution.pop('after_cleanup_identities')
                    write(root/'execution.json',execution)
                else:(root/'runtime/out.core').write_text('changed')
                with self.assertRaises(ValueError):self.call(root,pins)

    def test_cleanup_failure_cannot_be_hidden_by_rehashing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pins=self.fixture(root)
            write(root/'runtime/runtime-evidence.json',dict(primary_error=None,cleanup_errors=[{'operation':'runner-stop'}]))
            result=json.loads((root/'results.json').read_text())
            result['output_sha256']['runtime-evidence.json']=admission.sha(root/'runtime/runtime-evidence.json')
            write(root/'results.json',result)
            execution=json.loads((root/'execution.json').read_text())
            execution['results_sha256']=admission.sha(root/'results.json');write(root/'execution.json',execution)
            with self.assertRaises(ValueError):self.call(root,pins)


if __name__=='__main__':unittest.main()
