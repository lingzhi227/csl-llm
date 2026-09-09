"""Scheduling/failure injection only; these are not SDK numerical tests."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from model_request_sequence import run_requests
from model_step_validate import sequence_expectations
from model_core_validate import physical_cache


class Reference:
    def __init__(self, prompt, limit, winners, value):
        self.prompt, self.limit, self.winners = prompt, limit, winners
        self.steps = sequence_expectations(prompt, limit, winners)
        self.value = value

    def step(self, index):
        shape = (24, 2, self.steps[index]['consumed'], 64)
        return dict(cache_key=np.full(shape, self.value, np.float32),
                    cache_value=np.full(shape, -self.value, np.float32))


class StateIO:
    def __init__(self):
        self.invocation = 0
        self.prompt = None
        self.pending = self.failed = False
        self.finished = True
        self.events = []

    def reset_request(self, prompt, limit):
        assert not self.pending and not self.failed
        self.prompt = prompt
        self.finished = False
        self.events.append(('reset', prompt, limit, self.invocation))

    def step(self, **options):
        assert not self.pending and not self.finished and not self.failed
        self.pending = True
        self.invocation += 1
        self.events.append(('step', options, self.invocation))
        return {}

    def accept_step(self):
        assert self.pending
        self.pending = False
        self.finished = self.expected_finished
        self.events.append(('accept', self.invocation))


class RequestTests(unittest.TestCase):
    def validator(self, io):
        def check(actual, expected, plan, **kwargs):
            io.events.append(('validate', kwargs['invocation']))
            io.expected_finished = plan['finished']
            return {'normal_only': True}
        return check

    def test_prefill_decode_eos_then_shorter_reset_retains_physical_cache(self):
        refs = [Reference([10, 11], 3, [12, 151645], 2), Reference([20], 1, [21], 3)]
        io = StateIO()
        with patch('model_request_sequence.validate_step', side_effect=self.validator(io)):
            result = run_requests(io, refs, precision={},
                record=lambda r, s, a, report: io.events.append(('record', io.invocation)))
        self.assertEqual(result['invocation'], 4)
        self.assertFalse(result['runtime_accepted'])
        self.assertEqual([r['reason'] for r in result['requests']], [2, 1])
        self.assertEqual([e for e in io.events if e[0] == 'reset'],
                         [('reset', (10, 11), 3, 0), ('reset', (20,), 1, 3)])
        for i, event in enumerate(io.events):
            if event[0] == 'step':
                self.assertEqual([e[0] for e in io.events[i:i+4]], ['step', 'validate', 'record', 'accept'])
        cache, written = physical_cache(result['reference_history'], 'key')
        np.testing.assert_array_equal(cache[:, :, 0], 3)
        np.testing.assert_array_equal(cache[:, :, 1:3], 2)
        self.assertEqual(written, 3)
        self.assertFalse(cache[:, :, 3:].any())

    def test_validation_or_record_failure_prevents_accept_and_reset(self):
        for failure in ('validate', 'record'):
            with self.subTest(failure=failure):
                io = StateIO()
                refs = [Reference([1], 1, [2], 1), Reference([3], 0, [], 2)]
                def record(*args):
                    raise RuntimeError('record failed')
                check = self.validator(io) if failure == 'record' else RuntimeError('numerical mismatch')
                with patch('model_request_sequence.validate_step', side_effect=check):
                    with self.assertRaises(RuntimeError):
                        run_requests(io, refs, precision={}, record=record)
                self.assertTrue(io.failed)
                self.assertTrue(io.pending)
                self.assertEqual([e[0] for e in io.events].count('reset'), 1)
                self.assertNotIn('accept', [e[0] for e in io.events])

    def test_tampered_or_overflowing_schedule_rejected_before_io(self):
        ref = Reference([1], 1, [2], 1)
        ref.steps[0]['token'] = 9
        io = StateIO()
        with self.assertRaises(ValueError):
            run_requests(io, [ref], precision={}, record=lambda *args: None)
        self.assertEqual(io.events, [])
        long = Reference([1]*2048, 0, [], 1)
        with self.assertRaises(ValueError):
            run_requests(io, [long]*32, precision={}, record=lambda *args: None)
        self.assertEqual(io.events, [])


if __name__ == '__main__':
    unittest.main()
