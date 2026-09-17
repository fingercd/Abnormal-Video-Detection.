# ICASSP 2027: Normal–Anomaly Properties in Video Encoders

The active research workflow compares internal token distributions, spatiotemporal relations,
attention and readout behaviour before selecting a token-reduction method. The project profile
selects `videomaev2`, `timesformer`, `vjepa2` and `videomae`; the full VADBench catalog remains available.

- [Current project status and validation](projects/icassp2027/progress.md)
- [Active profile](projects/icassp2027/profile.yaml) and [development protocol](projects/icassp2027/protocol.yaml)
- [Observation plan](docs/icassp2027/01_NORMAL_ANOMALY_PROBES.md), [refactoring scope](docs/icassp2027/02_ENCODER_SCOPE_AND_REFACTOR.md), [execution runbook](docs/icassp2027/03_EXECUTION_RUNBOOK.md)
- [Paper workspace](paper/icassp2027/README.md) and [experiment rules](docs/icassp2027/06_EXPERIMENT_RULES_AND_PITFALLS.md)

Use the already working interpreter for the chosen model. Configuration inspection is lightweight:

```bash
python -m vadbench.paper status --project projects/icassp2027/profile.yaml
python -m vadbench.paper probe --project projects/icassp2027/profile.yaml --suite configs/papers/icassp2027/suites/probe-pilot.yaml --dry-run
```

`status` reports configuration and path availability, not model readiness. `--dry-run` loads no
weights and writes no run artifacts. Real observer verification and input/cohort instructions
are documented in [the project guide](projects/icassp2027/README.md). No final reducer or
normal–anomaly finding is claimed by this foundation refactor.

## VADBench framework reference (historical snapshot, September 2026)

VADBench is a research framework for comparing fixed-clip video representations and stateful long-video/VLM paths under a common UCF-Crime data, timeline, feature, and evaluation contract.

The reference paths are VideoMAE V2 Base (independent clips, no cross-clip cache) and HERMES with LLaVA-OneVision-Qwen2-0.5B (language-model decoder KV). HERMES projected visual tokens precede decoder context; cache-conditioned accuracy experiments require the contextual feature stage.

As inspected on 2026-09-11, server and local source agree at `0badc34` on `qzt/refactor-vadbench-simplification`. There are 25 candidates and 21 registered integrations. Existing evidence records 14 successful real-weight routes, 2 license blocks, 5 missing manual assets, and 4 candidate-only routes. The subsequent implementation revalidated 14 CPU routes and two real-video extraction paths. GPU forward verification remains incomplete because the available cuDNN libraries require a newer host GLIBC and all GPUs were occupied at the final check. The canonical server UCF-Crime directory is empty and the default manifests are absent, so no complete UCF-Crime accuracy result is claimed.

The framework implements manifest import/enrichment/audit, feature extraction, frozen-head training, checkpoint prediction, frame ROC-AUC/AP evaluation, smoke testing, and performance benchmarking. The extraction path/ID conflict and stale-success promotion have been fixed. Effective model configuration and weight identity are resolved together; strict evaluation shares coverage validation with prediction and requires a matching v2 dataset audit in official mode. See the implementation record for validation and remaining data prerequisites.

- [Chinese README and installation](README-CN.md)
- [Documentation index](docs/README.md)
- [Command workflows](docs/operations/workflows.md)
- [Server operations](docs/operations/server.md)
- [Current status and evidence](docs/progress/current-status.md)
- [Architecture](docs/architecture/current-system.md)
- [Baseline review](docs/reviews/2026-09-11-architecture-review.md) and [implementation record](docs/progress/2026-09-11-implementation.md)
- [UCF-Crime protocol](docs/research/ucf-crime-protocol.md)
- [Legacy code scope](lab_anomaly/README.md)

Use a repository checkout with Python 3.10–3.12. Heavy encoders require their own pinned assets and runtime dependencies in addition to the framework installation. Large videos, weights, features, environments and external repositories remain Git-ignored. Repository code is MIT-licensed; upstream code, datasets and weights retain their own terms.
