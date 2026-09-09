# Full-model runtime validation

This design concerns CSL-LLM only. Current execution and acceptance are in
`../RESUME.json`; source review and host tests do not qualify model inference.
The complete original model assembly and an immutable execution entry point
are prerequisites, not implied by these diagnostic modules.

## Ownership and host behavior

The device owns tied embedding lookup, all 24 decoder layers, final RMS,
151,936-way logits, minimum-index greedy selection, EOS and the next token.
The host writes prompt IDs and two request-control words once per request.
It launches device steps and reads diagnostic outputs. It never replaces an
activation, KV entry, weight or generated token with a reference value.

`model_runtime_io.py` expects a fresh SDK instance immediately after device
initialization. Its epoch starts at zero and persists across request resets.
Every step joins all 40,530 ready markers before compute and all completion
markers before diagnostic result reads. The caller validates the readback
before accepting the step. A failed request write poisons that instance: stop
and destroy it; do not retry using old host request metadata.

The owning execution entry point must freeze source, SDK/compiler, complete
assembly, references and tolerances; own resource limits and PID cleanup;
save partial transfers; and preserve the actual core even if SDK stop raises.
`model_runtime_probe.py` now implements that owner for one raw token and one
head step. Its source has independent review and a frozen source snapshot. No executable
run bundle has been prepared or executed; that requires the complete original
assembly. An explicit reviewed
wall-time budget must be supplied when preparing that first measurement.

## Numerical and state evidence

`model_reference.py` verifies the pinned official FP32 trace hashes and exposes
causal per-step outputs without running neural arithmetic. The initial official
trace contains the whole prompt; earlier sequential steps may observe only
their consumed prefix. Subsequent traces contain one new token. Frozen greedy
IDs are validation expectations, never data sent back to the device.

`model_step_validate.py` checks all layer outputs, controller and KV metadata,
device sequence state and fixed numerical gates. EOS and finish reasons are
derived independently from reference tokens and request limits. Timestamp
transfers must contain six valid 16-bit words per controller; reported layer
intervals include upstream waiting and must not be added together.

For a head step, two diagnostic `result` transfers cover exactly the 9,496
vocabulary matrix PEs: `(0,160,192,49)` and `(0,209,88,1)`, each with 128 words
per PE. Row-major data becomes `[1187,8,128]`. All eight allreduce copies must
agree bitwise before selecting one copy in original vocabulary order. Padding
PEs have only `result[1]` and are excluded. These reads transfer 4,861,952 bytes;
their simulator cost is not device inference performance. Prefill steps without
a head must not validate stale vocabulary buffers as current logits.

`model_symbols.py` has an explicit diagnostic mode. It checks actual ELF symbol
sizes, alignment, names and ownership for arithmetic, KV and protocol data.
The two-layer diagnostic endpoint at `(128,0)` is explicitly disabled for the
full model, where that coordinate belongs to layer 2. An audit of a partial
weight partition cannot supply unverified addresses for the eventual complete
assembly; audit that final artifact again.

`model_core_validate.py` reads actual stopped-core bytes using that verified
contract. It compares original immutable state, all identities and completion,
normal/core outputs and timestamps, all 312 decoder stages when bound to the
accepted single-token equation reference, all vocabulary allreduce copies, logical
KV visibility, physical KV and drained packet/boundary state. Its initializer
iterator must come from the independently accepted **complete original**
assembly; coordinate coverage alone does not prove original model weights.

Reset leaves physical SRAM outside the new visible prefix unchanged. The core
validator uses the independent request history to reconstruct these retained
slots, checks every written token separately under the existing numerical
gate, and requires never-written capacity and padding to remain zero. A short
history does not establish 2,048-token capacity or 256-token generation.

## Persistent request scheduling

`model_request_sequence.py` adds source-only scheduling over a fresh,
caller-owned `ModelRuntimeIO`. It independently reconstructs every frozen
request schedule before any transfer and rejects combined epoch overflow.
Each step must pass normal numerical/state validation and durable evidence
recording before acceptance allows the next launch. Any exception poisons
the instance for disposal by its owner; it cannot silently reset and continue.

The scheduler sends only original prompts and request limits through the I/O
layer. Reference winners determine expected diagnostics, never a device input.
At request boundaries it retains independent final KV prefixes, including
older prefixes that a shorter reset leaves physically present. It returns the
final normal readback and reference history for the owner's stopped-core
validation, explicitly with `runtime_accepted=False`.

This helper has host scheduling/failure tests only. It does not yet provide a
guarded fixed-three-prompt execution bundle, source/reference freeze, actual
SDK result, or independent core/protocol acceptance. The first single-token
runtime's numerical scope remains unchanged.

`model_fixed_requests.py` admits the original hash-bound FP32 corpus in the
order arithmetic, English, Chinese, then arithmetic again, with limit 16.
These requests contain 39, 45, 43 and 39 device steps. Admission verifies
existing references only; it performs no neural computation.

`model_generation_runtime.py` supplies the persistent runtime body, with
separate partial readbacks and reports for every step. The final arithmetic
request must reproduce the first request's layer outputs, model input,
generated tokens and head results bit-for-bit. Its final receipt still says
`runtime_accepted=False`: an owning guarded driver must bind complete original
assembly, first single-token acceptance, source and budget, then independently
qualify the actual stopped core and immutable weights. That guarded driver is
not yet implemented or frozen.

Both runtime bodies use `sdk_lifecycle.py` to attempt stop logging, SDK stop,
each reference close and receipt writing independently. Cleanup failures do
not mask the primary error. Host failure tests and SDK-image Python lifecycle
tests establish this control behavior, not SDK neural inference. The earlier
011841 source snapshot remains immutable and is superseded for future runtime
dispatch because its stop logging could prevent the SDK stop attempt.

## Remaining integration

Complete all original weight partitions and their whole-ELF contracts, then
assemble and qualify the complete artifact's initializers and diagnostic ABI.
Freeze the reviewed guarded runner with the accepted single-token original
reference, admit its resource budget and execute the actual model. Independently review numerical, core, resource and lifecycle evidence.
Expand from measured short integration to the declared context/generation
acceptance without changing tolerances or substituting host neural compute.
