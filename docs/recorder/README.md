# Recorder and replay

[The recorder README](../../recorder/README.md) is the reference for CLI, HTTP API, version-1 format, browser viewer and `Recording`/`ReplaySource`. It captures Station motor state, JPEG frames, commands and lifecycle events, with both arrival and Station timestamps. The service is a read-only Station consumer; it never writes motor goals.

`RECORDER_DIR` selects a writable directory; recordings are ignored by git and can grow roughly 1.4 MB/s with two cameras. Live capture requires a separate NormaCore checkout and Station; offline replay reads a recording directory without hardware. Raw recordings are not shipped here. The proposed drop-in replay wiring in the historical README is a proposal, not a claim that every legacy server already supports it.

Ports: recorder/viewer :8900; twin session viewer :8891; executor console :8910; learning dashboard :8920. Configure paths/camera roles for your site rather than copying the original camera hashes.
