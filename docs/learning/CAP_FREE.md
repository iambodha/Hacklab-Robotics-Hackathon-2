# A free thread and an extracted cap are different observations

`learning/capfree.py` takes stroke records and the active configuration. Its reference is the largest mean j7 load
in the first two good strokes. The default signal requires two consecutive successful strokes at or below 70% of
that reference, only after 280° achieved rotation. Slips, aborted strokes and strokes achieving less than 50% of
command do not qualify and break the consecutive run. These four thresholds and the achieved fraction are named
config keys, not fixed judgments in the tool body.

Replay of the twelve recorded attempts:

- Attempts 1–8: no free signal. These include false positives from the old short lift.
- Attempt 9: no signal (already partially loosened at start, 83 → 64 mean load).
- Attempt 10: reference 104, threshold 72.8; stroke 11 mean 69.7, stroke 12 mean 71.3. Signal begins at 11,
  confirms at 12. Later loads reach 44.7. This is the fit evidence, not an independent validation dataset.
- Attempt 11: no motion, insufficient evidence.
- Attempt 12: no signal at medium speed (153.1 → 114). Medium-speed load dynamics differ; absence of the
  signature does not prove attachment. The selected default is slow.

`twist.stop_on_free=on` stops while keeping the grip. A stop is a reason to attempt guarded extraction, never proof
that the cap is in the air. The report includes actual stroke count, achieved rotation, reference and confirmation.
`twist.time_budget_s=110` includes approach, planning and twisting. Predictive checks avoid starting another cycle
that is unlikely to fit. An asynchronous deadline also bounds a stalled approach/IK wait, holds arm goals at measured
position, and uses at most two seconds of the ten-second reply reserve for stop acknowledgement. A hardware IO call
that ignores cancellation or a failed stop acknowledgement remains a transport fault, not a guarantee of stopping
physical motion. No hardware timing validation was performed in this run.

Normal budget stops retain the grip. A resumed call checks it is still holding near the supplied cap, then releases,
backs off and regrips before its next forward stroke. A deadline interrupt is marked in the episode: the next call
reapproaches instead of assuming a completed cycle. Strokes interrupted inside a motion may have rotated the cap
without a complete recorded stroke; inspect after such an interrupt and treat count as completed strokes only.

`check_cap()` keeps its old 15 mm default. `extract_cap()` takes its 60 mm default from the learned profile. Both use
one shared guarded lift: small 3 mm steps, fixed orientation, safety loads and tracking aborts, and configurable
lag/rise fractions. A cap held back stops the lift; the MCP wrapper opens the jaws before retreating. Failed opening
prevents retreat so an attached cap is not dragged upward. The 400-unit two-joint load-rise stop and fixed hard loads
remain safety limits. IK branch jumps are rejected.

The extraction verdict requires holding before and after, no stop/resistance, and enough measured rise. The j7
signal is attached to the lift’s evidence and rendered alongside the verdict; it is advisory, since attempt 12 opened
without it. Rise still measures the **gripper**, not independent cap tracking. Confirm the visible air gap in camera
images; moving the whole bottle or holding another object could fool this motor-only verdict. A larger guarded lift
improves visibility, but does not create a new cap sensor.
