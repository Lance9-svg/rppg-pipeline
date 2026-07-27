# Phase 1 — Contact-PPG Reference

## Scope

Phase 1 constructs the reference heart rate from UBFC contact PPG only. No rPPG
estimate or rPPG error was read while selecting the peak rule.

## Frozen Method

1. Read contact PPG, stored sensor HR, and timestamps.
2. Sort timestamps and average samples sharing a timestamp.
3. Resample using the median positive timestamp interval.
4. Linearly detrend and apply a zero-phase 0.7-4.0 Hz Butterworth filter.
5. Detect peaks using:

   `prominence = 0.5 * 1.4826 * median absolute deviation`

6. Refine discrete peak positions with three-point parabolic interpolation.
7. Calculate the primary reference as `60 / median(IBI)`.
8. Calculate a 4096-point Hann-periodogram estimate with log-power peak
   interpolation.
9. Assign:

   - concordant: difference <= 5 bpm;
   - uncertain: difference > 5 and <= 10 bpm;
   - discordant: difference > 10 bpm;
   - insufficient: fewer than five peaks or estimator failure.

The sensor-HR row is secondary QC only. Raw values are preserved. An explicitly
labelled QC copy corrects the observed high-rate `127 -> 1` wrap by adding 128
bpm to positive values below 42 bpm.

## Verification

- Synthetic rates: 42, 60, 75, 100, 120, 180, and 240 bpm.
- Disturbances: drift, amplitude modulation, controlled noise, missing samples,
  and duplicate timestamps.
- Constant corrupted signal: classified as insufficient.
- Full local test suite: 26 passed.
- All 42 subjects parsed.

## PPG-Only Pilot

| Category | Windows | Percentage |
|---|---:|---:|
| Concordant | 2,319 | 96.34% |
| Uncertain | 67 | 2.78% |
| Discordant | 21 | 0.87% |
| Insufficient | 0 | 0.00% |
| Total | 2,407 | 100.00% |

Median time-frequency disagreement was 0.736 bpm and the 95th percentile was
3.941 bpm.

Prominence sensitivity:

| MAD multiplier | Concordant | Uncertain | Discordant |
|---:|---:|---:|---:|
| 0.25 | 2,228 | 76 | 103 |
| 0.50 | 2,319 | 67 | 21 |
| 0.75 | 2,319 | 67 | 21 |
| 1.00 | 2,321 | 66 | 20 |

The 0.50 multiplier was frozen as the lowest stable setting. The 5/10 bpm
category thresholds were not relaxed.

## Quality-Gate Decision

Phase 1 passed. Its outputs are frozen before Phase 2 rPPG processing. Any
subsequent reference change must be documented and regenerated as a new result
version.
