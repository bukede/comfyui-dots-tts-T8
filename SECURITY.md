# Security

## Reporting

Please report vulnerabilities privately through GitHub Security Advisories for
`T8mars/comfyui-dots-tts-T8`. Do not include access tokens, private voice samples or
other sensitive data in a public issue.

## Runtime boundaries

- Node import performs no network access, package installation or model loading.
- Model repositories and revisions are fixed in `manifests/dots_tts_models.json`.
- The bundled downloader validates declared sizes and SHA-256 values before reporting
  a model ready.
- The node never evaluates workflow input as code and never spawns a package installer.
- Reference audio is written only to a temporary WAV for the duration of inference and
  is removed afterwards.

Model weights use PyTorch-compatible serialization from the official dots-studio
repositories. Only use the pinned official sources or files you trust.
