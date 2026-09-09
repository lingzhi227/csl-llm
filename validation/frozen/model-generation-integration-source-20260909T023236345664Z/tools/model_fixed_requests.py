"""Bind the existing three-prompt corpus to persistent SDK request scheduling.

Read-only reference admission; no tokenization, model imports or neural math.
The repeated arithmetic request checks reset isolation after the other two
requests. This short corpus is distinct from the declared capacity gate.
"""
import json
from pathlib import Path

from model_reference import OfficialModelReference, sha

REPORT_SHA256 = 'ed7a56c88ffe6b6b629715b847fbc7415bbc82e6856c48ebbc03a6a080d16340'
MODEL_SHA256 = '9e4305503478f0f118ced0c5b87b148b73ae5823504f5636fb7c778e1252c75e'
PROMPTS_SHA256 = '3b17a15b7c3787f1b826b635d1017d7824d446c056bd2bc9cf34c7c31fe1fb30'
REQUEST_IDS = ('arithmetic', 'english', 'chinese', 'arithmetic')


def open_references(directory):
    """Open hash-bound references; the owning runner must close all of them."""
    directory = Path(directory)
    if sha(directory/'report.json') != REPORT_SHA256 or sha(directory/'prompts.json') != PROMPTS_SHA256:
        raise ValueError('Fixed official reference corpus identity mismatch')
    report = json.loads((directory/'report.json').read_text())
    prompts = json.loads((directory/'prompts.json').read_text())
    if (report['prompts_sha256'] != PROMPTS_SHA256 or
            report['generation_policy'] != dict(do_sample=False, repetition_penalty=1.0, raw_logits_argmax=True) or
            prompts['max_new_tokens'] != 16 or
            [p['id'] for p in prompts['prompts']] != list(REQUEST_IDS[:3]) or
            {p['id'] for p in report['prompts']} != set(REQUEST_IDS)):
        raise ValueError('Fixed request policy mismatch')
    references = []
    try:
        for prompt_id in REQUEST_IDS:
            references.append(OfficialModelReference(directory, report_sha256=REPORT_SHA256,
                model_manifest_sha256=MODEL_SHA256, prompt_id=prompt_id, limit=16))
    except BaseException:
        for reference in references:
            reference.close()
        raise
    return references
