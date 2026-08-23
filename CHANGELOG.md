# Changelog

## 0.3.0 - 2026-08-24

- Added SRT/WebVTT timeline rendering and speaker annotations.
- Added four-slot multi-speaker voice mapping with a fallback voice.
- Added literal, word and regular-expression pronunciation overrides.
- Added per-segment retries, atomic checkpoints and deterministic resume.
- Added recursive TXT/SRT/VTT batch queues with isolated failures and ZIP export.
- Added four ComfyUI V3 nodes and a desktop batch-production workbench.
- Hardened checkpoint writes against short Windows antivirus/indexer file locks.

## 0.2.0 - 2026-08-23

- Added a tokenizer-aware long-text V3 node with deterministic sentence splitting.
- Added `<pause=800ms>` / `<pause=1.2s>` explicit pauses, bounded to 60 seconds.
- Added deterministic per-segment seeds, 48 kHz audio assembly and detailed plan JSON.
- Reused one loaded runtime and one prepared voice prompt across all segments.
- Added API/UI example workflows and real SOAR/MF 2-Steps CUDA smoke coverage.

## 0.1.2 - 2026-08-23

- Replaced five optional environment-variable configuration reads with their existing
  safe defaults so the Registry scanner no longer flags the package for environment
  manipulation.
- Preserved normal backend selection, logging, profiling and ComfyUI model discovery
  behavior while keeping configuration explicit through function parameters.

## 0.1.1 - 2026-08-23

- Anchored root model/output ignore rules so the vendored `dots_tts/models` Python
  package is included in GitHub and Registry archives.
- Added a publish-time regression check for critical vendored inference modules.

## 0.1.0 - 2026-08-23

- Initial ComfyUI V3 release with six stable T8 node IDs.
- Added SOAR and MF 2-Steps model discovery and pinned download manifests.
- Added text-only, continuation-clone and x-vector-clone inference.
- Added standard 48 kHz ComfyUI AUDIO output, environment diagnostics and explicit
  model release.
- Added API/UI example workflows and Registry/Manager metadata.
