# Changelog

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
