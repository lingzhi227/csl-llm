# Evidence boundaries

`reviews/` contains selected original independent review JSON files, copied without rewriting their numerical results or hashes. Some references point to original execution artifacts that are intentionally not distributed. Those records support the reported historical scope but are not self-contained rerunnable bundles.

The publication includes source preparation/execution tools so new evidence can be produced using an independently obtained checkpoint and SDK. It excludes original NPZ tensors, model weights, ELF/core files, SDK images and private session coordination. A new run has a new manifest and must pass its own checks.

S0 acceptance may reference additional internal review documents absent from this selection. Review hashes identify original snapshots, not automatically the latest published source. `release/SOURCE-SNAPSHOT.json` records the selected source bytes before documented host-path edits; `release/MANIFEST.json` authenticates this release payload.
