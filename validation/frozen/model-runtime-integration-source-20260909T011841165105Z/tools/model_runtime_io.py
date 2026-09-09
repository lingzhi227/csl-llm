"""Diagnostic SDK I/O for the original 193x210 model placement.

This layer never loads activations, weights or generated tokens from the host.
The owning runner must verify compiled artifacts, own load/run/stop, preserve
partial readbacks, and validate normal results plus the stopped actual core.
Readback traffic here is diagnostic cost, not device inference performance.
"""
import numpy as np


def vocabulary_logits(rows, tail):
    """Reconstruct original vocabulary order and verify every allreduce copy."""
    if rows.dtype != np.float32 or tail.dtype != np.float32 or rows.shape != (49, 192, 128) or tail.shape != (1, 88, 128):
        raise ValueError('Read only the 9,496 matrix PEs; padding has result[1]')
    groups = np.concatenate((rows.reshape(-1, 128), tail.reshape(-1, 128))).reshape(1187, 8, 128)
    if not np.isfinite(groups).all():
        raise ValueError('Nonfinite vocabulary output')
    bits = groups.view(np.uint32)
    if not np.all(bits == bits[:, :1, :]):
        raise ValueError('Vocabulary allreduce results differ between participants')
    return groups[:, 0, :].reshape(151936).copy()


class ModelRuntimeIO:
    def __init__(self, runner, *, memcpy_data_type, memcpy_order, save, stage):
        self.runner, self.save, self.stage = runner, save, stage
        self.options = dict(streaming=False, data_type=memcpy_data_type.MEMCPY_32BIT,
                            order=memcpy_order.ROW_MAJOR, nonblock=False)
        self.type16 = memcpy_data_type.MEMCPY_16BIT
        self.ids = {name: runner.get_id(name) for name in ('progress', 'control', 'summary',
            'prompt', 'generated', 'status', 'output', 'input', 'result', 'winner', 'best', 'timing')}
        self.invocation = 0
        self.prompt = None
        self.limit = None
        self.pending = False
        self.read_complete = False
        self.finished = True
        self.last_actual = None
        self.failed = False

    def reset_request(self, prompt, limit):
        prompt = tuple(prompt)
        if (self.failed or self.pending or not prompt or type(limit) is not int or not 0 <= limit <= 256 or
                len(prompt)+limit > 2048 or any(type(t) is not int or not 0 <= t < 151936 for t in prompt)):
            raise ValueError('Invalid request or unfinished preceding work')
        # A partial request write cannot be recovered by reusing the prior
        # metadata. Only the owning runner may stop and destroy this instance.
        self.pending = True
        self.failed = True
        for name, values in [('prompt', prompt), ('control', (len(prompt), limit))]:
            self.stage('h2d-'+name)
            self.runner.memcpy_h2d(self.ids[name], np.asarray(values, np.uint32),
                                  192, 160, 1, 1, len(values), **self.options)
        self.stage('reset-model')
        self.runner.launch('reset_model', nonblock=False)
        self.prompt, self.limit = prompt, limit
        self.finished = False
        self.pending = False
        self.failed = False
        # Reset does not reset the epoch. Physical KV beyond seen/valid is stale.

    def _read(self, record, key, name, x, y, width, height, count, *, dtype=np.uint32, timing=False):
        self.stage('d2h-'+key)
        values = np.zeros(width*height*count, dtype)
        options = dict(self.options)
        if timing:
            options['data_type'] = self.type16
        self.runner.memcpy_d2h(values, self.ids[name], x, y, width, height, count, **options)
        record[key] = values.reshape(height, width, count)
        self.save(record)
        return values

    def step(self, *, head, generated_count):
        if (self.failed or self.prompt is None or self.pending or self.finished or self.invocation >= 65535 or type(head) is not bool or
                type(generated_count) is not int or not 0 <= generated_count <= self.limit):
            raise ValueError('Invalid diagnostic step or exhausted epoch')
        self.pending = True
        self.read_complete = False
        self.invocation += 1
        actual = {}
        read = lambda key, name, x, y, w, h, n, **kw: self._read(actual, key, name, x, y, w, h, n, **kw)
        self.stage('global-prepare')
        self.runner.launch('prepare', nonblock=False)
        actual['ready'] = read('ready', 'progress', 0, 0, 193, 210, 1).reshape(210, 193)
        np.testing.assert_array_equal(actual['ready'], np.full((210, 193), self.invocation*2-1, np.uint32))
        self.stage('compute-device-model')
        self.runner.launch('compute', nonblock=False)
        actual['summary'] = read('summary', 'summary', 192, 160, 1, 1, 7)
        actual['progress'] = read('progress', 'progress', 0, 0, 193, 210, 1).reshape(210, 193)
        np.testing.assert_array_equal(actual['progress'], np.full((210, 193), self.invocation*2, np.uint32))
        actual['control'] = read('control', 'control', 192, 160, 1, 1, 2)
        actual['prompt'] = read('prompt', 'prompt', 192, 160, 1, 1, len(self.prompt))
        actual['generated'] = (read('generated', 'generated', 192, 160, 1, 1, generated_count)
                               if generated_count else np.empty(0, np.uint32))
        actual['model_input'] = read('model_input', 'input', 192, 160, 1, 1, 896, dtype=np.float32)
        actual['model_timing'] = read('model_timing', 'timing', 192, 160, 1, 1, 6, timing=True)
        outputs, states, cache, clocks = [], [], [], []
        for layer in range(24):
            x, y = (layer % 3)*64, (layer//3)*20
            outputs.append(read(f'layer_{layer}_output', 'output', x+62, y, 1, 1, 896, dtype=np.float32))
            states.append(read(f'layer_{layer}_status', 'status', x+62, y, 1, 1, 2))
            clocks.append(read(f'layer_{layer}_timing', 'timing', x+62, y, 1, 1, 6, timing=True))
            cache.append([read(f'kv_{layer}_{head_id}', 'status', x+32+head_id*11, y+14, 11, 6, 2).reshape(66, 2)
                          for head_id in (0, 1)])
        actual.update(layer_outputs=np.stack(outputs), controller_status=np.stack(states),
                      kv_status=np.asarray(cache), controller_timing=np.stack(clocks))
        if head:
            actual['final_norm'] = actual['model_input'].copy()
            actual['winner'] = read('winner', 'winner', 192, 160, 1, 1, 1)
            actual['best'] = read('best', 'best', 192, 160, 1, 1, 1, dtype=np.float32)
            rows = read('vocabulary_rows', 'result', 0, 160, 192, 49, 128, dtype=np.float32).reshape(49, 192, 128)
            tail = read('vocabulary_tail', 'result', 0, 209, 88, 1, 128, dtype=np.float32).reshape(1, 88, 128)
            actual['logits'] = vocabulary_logits(rows, tail)
        self.save(actual)
        self.read_complete = True
        self.last_actual = actual
        # This is only a successful readback. The owner must validate it and
        # protocol drainage before choosing another step or resetting.
        return actual

    def accept_step(self):
        """Called only after the owner validates this step's numerical/state checks."""
        if self.failed or not self.pending or not self.read_complete:
            raise ValueError('No completely read pending step to accept')
        self.finished = bool(self.last_actual['summary'][2])
        self.pending = False
