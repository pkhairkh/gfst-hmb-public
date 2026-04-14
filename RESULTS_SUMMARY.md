# Results Summary

This summary is derived from recorded repository artifacts from the working repository, as summarized during release preparation and documented in the release audit. The raw report tree itself is not part of the staged public bundle.

## Training Snapshot

One recorded canonical run at sequence length `256` and active micro-batch size `6` reports:

- `val_ce = 12.2674` at step `99`
- `val_ce = 11.7247` at step `199`
- `val_ce = 10.9624` at step `299`
- `val_ce = 11.1946` at step `399`

The same run reports teacher gap:

- `3.2528` at step `99`
- `2.7102` at step `199`
- `1.9479` at step `299`
- `2.1801` at step `399`

This supports a restrained claim that the recorded run improved materially through step `299`, then remained imperfect rather than monotonically solved.

## Retrieval And Readout Snapshot

From the same evaluation stream at length `256`:

- `top1_block_recall = 1.0` at steps `99`, `199`, `299`, and `399`
- `top_r_block_recall = 1.0` at the same checkpoints
- `exact_token_recall` remained limited: `0.0391`, `0.1230`, `0.1172`, `0.1152`
- `selected_copy_matches_gold_rate_on_oracle_positions` remained limited: `0.1408` to `0.1620`

Interpretation:

- Coarse block retrieval looked strong in the recorded length-256 evaluation.
- Exact token recovery and final readout remained a bottleneck.

## Runtime And Probe Evidence

The repository contains lightweight evidence for several runtime checks:

- Parity probe summaries report zero or near-zero deltas in selected configurations.
- A block handoff probe summary reports clean commit and consume sites with zero output deltas in one recorded run.
- A batched canonical summary probe reports a clean result at sequence length `256`.
- Boundary-transition and global-path diff probes exist, including examples of clean paths and examples of residual mismatch, which supports the view that these checks were used as engineering diagnostics rather than as marketing artifacts.

## Public Conclusion

The public evidence supports a characterization of GFST-HMB as research-engineering work with meaningful runtime validation and promising retrieval behavior at the block level. It does not support claiming a fully validated exact-memory breakthrough or a complete end-to-end closure on exact token reconstruction.
