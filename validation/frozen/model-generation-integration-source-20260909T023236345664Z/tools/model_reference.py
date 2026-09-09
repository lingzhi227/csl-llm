"""Read the pinned official full-model traces without running neural compute.

Each SDK step is checked against its original-input causal reference. The first
official call contains the entire prompt; later calls contain one new token.
This adapter never conditions the reference on an observed SDK activation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from model_step_validate import sequence_expectations

DECODER_STAGES = ('input_norm', 'q', 'k', 'v', 'context', 'o', 'attention_residual',
                  'post_norm', 'up', 'gate', 'activation', 'down', 'output')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class OfficialModelReference:
    def __init__(self, directory, *, report_sha256, model_manifest_sha256, prompt_id, limit):
        self.directory = Path(directory)
        if sha(self.directory/'report.json') != report_sha256:
            raise ValueError('Official report differs from frozen reference identity')
        report = json.loads((self.directory/'report.json').read_text())
        if (not report['success'] or report['dtype'] != 'float32' or
                report['revision'] != '7ae557604adf67be50417f59c2c2f167def9a775' or
                report['model_manifest_sha256'] != model_manifest_sha256 or
                not report['eager_attention'] or not report['tied_embedding'] or
                report['versions']['transformers'] != '4.51.3'):
            raise ValueError('Reference does not match the pinned model arithmetic')
        entries = [p for p in report['prompts'] if p['id'] == prompt_id]
        if len(entries) != 1 or not entries[0]['official_generate_exact']:
            raise ValueError('A unique official greedy trajectory is required')
        entry = entries[0]
        self.prompt = entry['prompt_tokens']
        self.limit = limit
        self.winners = entry['generated_tokens'][:limit]
        self.steps = sequence_expectations(self.prompt, limit, self.winners)
        self.files = report['files']
        self.traces = []
        for index in range(max(1, len(self.winners))):
            record = entry['steps'][index]
            relative = f"{prompt_id}/{record['trace']}"
            path = self.directory/relative
            if (path.resolve().parent != (self.directory/prompt_id).resolve() or
                    record['step'] != index or record['context_tokens'] != len(self.prompt)+index or
                    record['selected_token'] != entry['generated_tokens'][index] or
                    sha(path) != self.files[relative]):
                raise ValueError('Official trace identity or causal position mismatch')
            self.traces.append(path)
        self._loaded_index = None
        self._loaded = None

    def step(self, index):
        if type(index) is not int or not 0 <= index < len(self.steps):
            raise ValueError('Reference step outside the frozen trajectory')
        plan = self.steps[index]
        trace_index = max(0, index-len(self.prompt)+1)
        position = index if trace_index == 0 else 0
        if self._loaded_index != trace_index:
            if self._loaded is not None:
                self._loaded.close()
            self._loaded = np.load(self.traces[trace_index], allow_pickle=False)
            self._loaded_index = trace_index
        trace = self._loaded
        outputs = np.stack([trace[f'model.layers.{layer}'][0, position]
                            for layer in range(24)])
        result = dict(layer_outputs=outputs,
                      final_norm=trace['model.norm'][0, position].copy(),
                      embedding=trace['model.embed_tokens'][0, position].copy())
        if plan['head']:
            result['logits'] = trace['logits'].copy()
            if result['logits'].shape != (151936,) or int(np.argmax(result['logits'])) != plan['generated'][-1]:
                raise ValueError('Full logit trace differs from the original greedy trajectory')
        # The first trace has all prompt KV. Only the logical consumed prefix
        # is visible to the sequential SDK at an earlier prefill position.
        for kind in ('key', 'value'):
            result[f'cache_{kind}'] = np.stack([
                trace[f'cache.{layer}.{kind}'][0, :, :plan['consumed']]
                for layer in range(24)])
        independent = [f'independent.layers.{layer}.{name}' for layer in range(24) for name in DECODER_STAGES]
        present = [name in trace.files for name in independent]
        if any(present):
            if not all(present):
                raise ValueError('Partial independent full-model stage reference')
            for layer in range(24):
                for name in DECODER_STAGES:
                    result[f'layer_{layer}_{name}'] = trace[f'independent.layers.{layer}.{name}'][position].copy()
        expected = {'layer_outputs': (24, 896), 'embedding': (896,), 'final_norm': (896,),
                    'cache_key': (24, 2, plan['consumed'], 64),
                    'cache_value': (24, 2, plan['consumed'], 64)}
        for name, value in result.items():
            if value.dtype != np.float32 or not np.isfinite(value).all() or (
                    name in expected and value.shape != expected[name]):
                raise ValueError(f'Invalid full-model reference tensor: {name}')
        return result

    def close(self):
        if self._loaded is not None:
            self._loaded.close()
            self._loaded = None
            self._loaded_index = None
