# Release journal

## 2026-11-22 10:11 CET — inspection and baseline

- Read the common brief first, all four r5 reports (including Codex takeover reports), topic docs and journals, the external integration board, LOG.md, and subsystem READMEs. No AGENTS.md found under the worktree parent.
- The updated brief overrides earlier move instructions: no code moves, commits, pushes or live robot access. Preserve real launchers and coordinate by avoiding r6-owned implementation files entirely.
- Started the full common suite plus Generaloid and comms before edits, with Station/real motion disabled and temporary DATA_ROOT. Added reportlab, required by the merged kitting tests. Output: `/private/tmp/r5-release-baseline.log`.
- Initial shipping-tree scan: 577 files, no file above 5,000,000 bytes. Full reachable-history patch and blob scan: no oversized blob; one Google-key-shaped value in models/tests/conftest.py at commit ed7ab675, explicitly FAKE_KEY containing “fake”, used solely for mocked redaction checks. No actual credential identified. Broad secret/token/.env matches require classification, not automatic removal.
- Found 40 developer-machine path occurrences across 31 files. Will document every affected file; retain provenance and historical evidence, and supply explicit portable environment settings. Avoid changing r6 launchers, calibration, connectors or dashboard code.
- Packaging will support uv from a source checkout; legacy bare imports and asset lookup make a standalone wheel premature. The quickstart must fetch pinned upstream assets rather than reuse the local NormaCore checkout. No git clone will be run.
