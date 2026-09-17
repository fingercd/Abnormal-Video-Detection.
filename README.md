# VADBench

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
