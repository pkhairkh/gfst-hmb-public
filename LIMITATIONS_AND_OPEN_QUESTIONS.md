# Limitations And Open Questions

- The strongest evidence in this release is systems and diagnostics evidence, not complete scientific closure.
- Exact-token recovery remains the clearest bottleneck in the recorded sequence-length-256 evaluations.
- The public bundle does not include upstream teacher assets or raw checkpoints, so full run replication is intentionally incomplete.
- Some source files, tests, scripts, and configs were withheld because they embedded internal path assumptions that could not be sanitized without code edits.
- Comparisons to external baselines are not comprehensive in this release package.
- The public evidence should not be framed as a frontier-validated memory result.

Open questions that remain visible from the artifacts:

- How much of the remaining gap comes from readout limitations versus write/support formation?
- How stable are the clean parity and handoff results across broader settings?
- What public configuration set should replace the internal-path configs for a fuller open-source release?
