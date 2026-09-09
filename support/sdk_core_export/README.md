# Two-PE explicit core diagnostic source

This isolated directory contains the accepted export driver (`driver.py`), two-PE CSL (`layout.csl`, `pe.csl`, `packets.csl`), core reader, lifecycle helper and SDK executor. It is inspection material with recorded source identity; it is not a self-contained runnable historical bundle.

Read [the accepted cases, failure outcomes and reproduction prerequisites](../../docs/SDK-CORE-EXPORT.md) before using it. `driver.py` intentionally requires the exact historical compile manifest. The private compiled bundle and actual cores are not distributed. A fresh build needs separate provenance and qualification.

Only `sdk_probe.py` has portable SDK path substitutions (`CSL_LLM_SDK_IMAGE`, `CSL_LLM_CS_PYTHON`); `source-provenance.json` is explicitly derived metadata. Other source bytes match the frozen export snapshot. These publication edits have no fresh SDK acceptance. The local manifest records this published copy, not the historical compiled identity.
