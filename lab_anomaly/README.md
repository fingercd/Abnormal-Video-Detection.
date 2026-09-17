# `lab_anomaly`: retained VideoMAE v2 + MIL prototype

This directory is a retained local prototype, not the VADBench benchmark entry point. It independently encodes fixed video clips with `OpenGVLab/VideoMAEv2-Base`, then pools them with an MIL head into a binary video prediction: `normal` or `anomaly`.

It does not implement streaming state or cross-clip KV reuse, the official UCF-Crime split, frame-level evaluation, or VADBench manifests, provenance, and artifact schemas. Start benchmark work from the repository [README](../README.md), `src/vadbench/`, and the UCF-Crime protocol instead.

The runnable prototype path is:

```text
index_build.py -> precompute_clips.py -> train_end2end.py -> rtsp_service.py
```

Run modules from the repository root. `index_build.py` has a CLI; `precompute_clips.py` and `train_end2end.py` use their fixed in-code configuration plus optional fixed YAML locations. The supplied training YAML uses 12 frames while the precompute default is 16; align the precompute parameters before running, or manifest validation will reject the training run. See [README-CN.md](README-CN.md) for exact commands and limitations.
