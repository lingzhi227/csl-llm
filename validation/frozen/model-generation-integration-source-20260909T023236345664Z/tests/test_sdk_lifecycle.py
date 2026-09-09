"""Cleanup failures must not suppress SDK disposal or the original failure."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from sdk_lifecycle import stop_and_record


def fail(message):
    raise RuntimeError(message)


class LifecycleTests(unittest.TestCase):
    def test_all_cleanup_attempts_and_original_error_preserved(self):
        attempts, receipts = [], []
        original = ValueError('original')
        runner = SimpleNamespace(stop=lambda: attempts.append('stop'))
        references = [SimpleNamespace(close=lambda: fail('close0')),
                      SimpleNamespace(close=lambda: attempts.append('close1'))]
        stop_and_record(runner, stage=lambda _: fail('log'), record=receipts.append,
                        primary_error=original, references=references)
        self.assertEqual(attempts, ['stop', 'close1'])
        self.assertEqual(receipts[0]['primary_error'], {'type': 'ValueError', 'message': 'original'})
        self.assertEqual([e['operation'] for e in receipts[0]['cleanup_errors']],
                         ['stage-stop', 'reference-close-0'])
        self.assertEqual(len(original.__notes__), 2)

    def test_stop_and_receipt_failure_propagates_stop_with_receipt_note(self):
        with self.assertRaisesRegex(RuntimeError, '^stop$') as raised:
            stop_and_record(SimpleNamespace(stop=lambda: fail('stop')),
                stage=lambda _: None, record=lambda _: fail('disk'))
        self.assertIn('evidence-write: RuntimeError: disk', raised.exception.__notes__)

    def test_unwritable_receipt_does_not_mask_active_primary(self):
        original = ValueError('numerical failure')
        stop_and_record(SimpleNamespace(stop=lambda: None), stage=lambda _: None,
            record=lambda _: fail('disk'), primary_error=original)
        self.assertIn('evidence-write: RuntimeError: disk', original.__notes__)


if __name__ == '__main__':
    unittest.main()
