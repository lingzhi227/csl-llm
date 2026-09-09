"""Guard/receipt tests only; no full-model candidate is available yet."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from sdk_probe import sha
from model_generation_probe import limits, first_admission_receipt


class GuardTests(unittest.TestCase):
    def test_budget_requires_explicit_bounded_integer(self):
        base=dict(timeout_seconds=1800,rss_limit_kib=24*1024*1024,simulator_threads=16)
        self.assertEqual(limits(base),1800)
        for key,value in [('timeout_seconds',True),('timeout_seconds',1799),('timeout_seconds',86401),
                          ('rss_limit_kib',24*1024*1024+1),('simulator_threads',16.0)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                limits(dict(base,**{key:value}))

    def test_copied_first_receipt_must_bind_same_assembly_and_actual_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);first=root/'first-integration';first.mkdir()
            def write(name,value):(first/name).write_text(json.dumps(value)+'\n')
            write('manifest.json',{'synthetic_fixture':True})
            write('results.json',dict(success=True,output_sha256={'out.core':'core-digest'}))
            write('execution.json',dict(success=True,sdk_sha256='sdk',
                manifest_sha256=sha(first/'manifest.json'),results_sha256=sha(first/'results.json')))
            proof=dict(sdk_sha256='sdk',assembly_execution_sha256='assembly',
                actual_output_sha256={'out.core':'core-digest'},
                **{name+'_sha256':sha(first/(name+'.json')) for name in ('manifest','execution','results')})
            write('admission.json',proof)
            config=dict(first_admission_sha256=sha(first/'admission.json'),assembly_execution_sha256='assembly')
            self.assertEqual(first_admission_receipt(root,config),proof)
            with self.assertRaises(ValueError):
                first_admission_receipt(root,dict(config,assembly_execution_sha256='other'))
            write('results.json',dict(success=True,output_sha256={'out.core':'tampered'}))
            with self.assertRaises(ValueError):first_admission_receipt(root,config)


if __name__=='__main__':unittest.main()
