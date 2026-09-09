"""Require actual first-model evidence before preparing persistent generation.

This check is only a prerequisite for a future candidate. Independent review
and current process/resource admission remain separate dispatch requirements.
It never runs the SDK or manufactures a missing completion receipt.
"""
import json
from pathlib import Path

from sdk_probe import sha, verify

# Exact dependencies copied by the reviewed022128 single-token prepare routine.
FIRST_FILES = {'tools/sdk_probe.py': '9ccab37dc8b6ecfb0d23d56bf0826da7b87ce5d3c22d0a4ba94cb4aea6501e1c',
 'tools/sdk_core.py': '10ba97cebcccf4749e5ee0c22be5518f800e3820429f7d2ae1425b7f66b9e88d',
 'tools/sdk_lifecycle.py': 'b9d469ff88a21d9468684d46acbd377df590e63a7e7ba4e2723c910ecfb39949',
 'tools/model_reference.py': '748cc583e5118c5c2d17a0b954fa6f4793e155cd3d05877b8e30dbff99645791',
 'tools/model_step_validate.py': '2d023b2b43f6eade8938a05b101f3a3349b85c70d130428ccb902325540ec022',
 'tools/model_runtime_io.py': 'dce79641a31672797e78407a3a304a34b0554609eabad12bde0710ae0977a09d',
 'tools/model_core_validate.py': '5e1965232595251e5d17c36683baa0bbf539a4d4d75f40747820abf9cd96c7a7',
 'tools/model_partition_assemble.py': '691c57c0ead50585ca579c47511e649c1e6616645f5537263b606a6ceb43c245',
 'tools/model_initializer_audit.py': '497abba2a3ce9927325ea8e56f9d2c97fb2ecd785f3fec3105aef70fb0511e29',
 'tools/model_symbols.py': '9e73d3b1eb4595fb47bc7d538aa30a610240c5069880de0a559b74e868f13e90',
 'tools/decoder_chain_symbols.py': 'd5a3485726d8b8e192e3786dff4b5a54327e8d14f581f28c28455dffce7f0d35',
 'src/csl_llm/__init__.py': 'e330d6b21720c96f282971a7ec9519f943e1c21b9bd944c611bbb4763ca390e2',
 'src/csl_llm/coordinate_identity.py': '33943a93a0c562b0647fef94cb43e9942c88fa1f7116537bdaeddda186ad7f83',
 'src/csl_llm/decoder_layout.py': '20fa5b2f051dcad58cb2c44f7d7f511b41fde243113c61a338ac2c661877ec13',
 'src/csl_llm/regions.py': '2b23cc1644fc6e28ebe461bb68dfe0b4c77ac03570c094411d78d477377728c2',
 'driver.py': '650b7e2fdb02a6c2055ad8247f4d59f62d1e38a20d431d637e5f58604a7c5197',
 'executor.py': '9ccab37dc8b6ecfb0d23d56bf0826da7b87ce5d3c22d0a4ba94cb4aea6501e1c'}


def first_single_token(root, *, assembly_execution_sha256, sdk_sha256):
    """Validate the full actual first-run bundle, not a copied success flag."""
    root = Path(root)
    manifest = verify(root)
    execution = json.loads((root/'execution.json').read_text())
    result = json.loads((root/'results.json').read_text())
    config = json.loads((root/'config.json').read_text())
    checks = (
        execution.get('success') is True,
        result.get('success') is True,
        result.get('normal', {}).get('passed') is True,
        result.get('core', {}).get('passed') is True,
        result.get('core', {}).get('coordinates') == 40530,
        config.get('prompt') == [151644], config.get('generation_limit') == 1,
        config.get('assembly_execution_sha256') == assembly_execution_sha256,
        result.get('assembly_execution_sha256') == assembly_execution_sha256,
        manifest.get('sdk_sha256') == sdk_sha256 == execution.get('sdk_sha256'),
        execution.get('manifest_sha256') == sha(root/'manifest.json'),
        execution.get('results_sha256') == sha(root/'results.json'),
        all(manifest['files'].get(name) == digest and sha(root/name) == digest
            for name, digest in FIRST_FILES.items()),
    )
    if not all(checks):
        raise ValueError('Missing or incompatible actual single-token model acceptance evidence')
    # These receipts must be from the successful guarded run. Current actual
    # PID absence is independently checked by the dispatcher, not inferred here.
    if (not execution.get('observed_process_identities') or
            set(execution.get('after_cleanup_identities', {})) != set(execution['observed_process_identities']) or any(
            value is not None for value in execution.get('after_cleanup_identities', {}).values())):
        raise ValueError('First-model cleanup receipt is incomplete or reports a remaining identity')
    outputs = result['output_sha256']
    required = {'normal-d2h.npz', 'normal-report.json', 'actual.npz', 'protocol.json',
                'runtime-evidence.json', 'out.core'}
    if set(outputs) != required:
        raise ValueError('Full first-model normal/core evidence is required')
    for name in sorted(required):
        if sha(root/'runtime'/name) != outputs[name]:
            raise ValueError(f'Changed actual first-model output: {name}')
    lifecycle = json.loads((root/'runtime/runtime-evidence.json').read_text())
    if lifecycle.get('primary_error') is not None or lifecycle.get('cleanup_errors') != []:
        raise ValueError('First-model runtime cleanup did not complete cleanly')
    return dict(manifest_sha256=sha(root/'manifest.json'), execution_sha256=sha(root/'execution.json'),
        results_sha256=sha(root/'results.json'), assembly_execution_sha256=assembly_execution_sha256,
        sdk_sha256=sdk_sha256, actual_output_sha256=outputs,
        scope='Actual first-token prerequisite only; no persistent-generation dispatch or acceptance.')
