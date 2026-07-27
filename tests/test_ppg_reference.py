from __future__ import annotations

import numpy as np

from rppg_pipeline.ppg_reference import (
    PPGReferenceConfig,
    build_subject_reference,
    classify_reference,
    estimate_frequency_domain_hr,
    estimate_time_domain_hr,
    prepare_uniform_ground_truth,
    preprocess_ppg,
)
from rppg_pipeline.ubfc import UBFCGroundTruth


def test_synthetic_rates_are_recovered_by_both_estimators() -> None:
    config = PPGReferenceConfig(window_sec=10.0)
    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)

    for expected_bpm in (42.0, 60.0, 75.0, 100.0, 120.0, 180.0, 240.0):
        ppg = np.sin(2 * np.pi * (expected_bpm / 60.0) * time_sec)
        filtered = preprocess_ppg(ppg, sample_rate_hz, config)
        time_result = estimate_time_domain_hr(filtered, sample_rate_hz, config)
        frequency_result = estimate_frequency_domain_hr(
            filtered,
            sample_rate_hz,
            config,
        )

        assert abs(time_result["time_hr_bpm"] - expected_bpm) <= 1.0
        assert abs(frequency_result["frequency_hr_bpm"] - expected_bpm) <= 1.0


def test_uniform_preparation_handles_missing_and_duplicate_timestamps() -> None:
    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 20.0, 1.0 / sample_rate_hz)
    pulse_hz = 75.0 / 60.0
    ppg = (
        np.sin(2 * np.pi * pulse_hz * time_sec)
        + 0.2 * np.sin(2 * np.pi * 0.1 * time_sec)
    )
    ppg *= 1.0 + 0.15 * np.sin(2 * np.pi * 0.05 * time_sec)
    ppg += np.random.default_rng(7).normal(0.0, 0.05, len(time_sec))
    sensor = np.full(len(time_sec), 75.0)

    keep = np.ones(len(time_sec), dtype=bool)
    keep[::17] = False
    time_missing = time_sec[keep]
    ppg_missing = ppg[keep]
    sensor_missing = sensor[keep]
    duplicate_indices = np.array([20, 100, 250])
    ground_truth = UBFCGroundTruth(
        ppg_trace=np.concatenate([ppg_missing, ppg_missing[duplicate_indices]]),
        sensor_hr=np.concatenate(
            [sensor_missing, sensor_missing[duplicate_indices]]
        ),
        timestamp_sec=np.concatenate(
            [time_missing, time_missing[duplicate_indices]]
        ),
    )

    uniform = prepare_uniform_ground_truth(ground_truth)
    result = build_subject_reference("subject1", ground_truth)

    assert uniform.duplicate_timestamp_count == len(duplicate_indices)
    assert abs(uniform.sample_rate_hz - sample_rate_hz) < 0.01
    assert len(result.windows) > 0
    assert (result.windows["reference_category"] == "concordant").mean() > 0.8


def test_constant_signal_is_flagged_insufficient() -> None:
    config = PPGReferenceConfig()
    sample_rate_hz = 30.0
    values = np.ones(300)
    filtered = preprocess_ppg(values, sample_rate_hz, config)
    time_result = estimate_time_domain_hr(filtered, sample_rate_hz, config)
    frequency_result = estimate_frequency_domain_hr(
        filtered,
        sample_rate_hz,
        config,
    )
    category, _ = classify_reference(
        float(time_result["time_hr_bpm"]),
        float(frequency_result["frequency_hr_bpm"]),
        int(time_result["n_peaks"]),
        config,
    )

    assert category == "insufficient"


def test_reference_categories_use_frozen_tolerances() -> None:
    config = PPGReferenceConfig()

    assert classify_reference(70.0, 74.9, 8, config)[0] == "concordant"
    assert classify_reference(70.0, 75.1, 8, config)[0] == "uncertain"
    assert classify_reference(70.0, 80.0, 8, config)[0] == "uncertain"
    assert classify_reference(70.0, 80.1, 8, config)[0] == "discordant"
    assert classify_reference(np.nan, 70.0, 0, config)[0] == "insufficient"


def test_frequency_interpolation_recovers_off_grid_rate() -> None:
    config = PPGReferenceConfig()
    sample_rate_hz = 30.0
    expected_bpm = 73.4
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    ppg = np.sin(2 * np.pi * (expected_bpm / 60.0) * time_sec)
    filtered = preprocess_ppg(ppg, sample_rate_hz, config)

    result = estimate_frequency_domain_hr(filtered, sample_rate_hz, config)

    assert abs(result["frequency_hr_bpm"] - expected_bpm) < 0.5
    assert abs(result["frequency_interpolation_delta"]) > 0


def test_sensor_hr_wrap_is_preserved_raw_and_corrected_for_qc() -> None:
    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    ppg = np.sin(2 * np.pi * (130.0 / 60.0) * time_sec)
    sensor_hr = np.full(len(time_sec), 2.0)
    ground_truth = UBFCGroundTruth(
        ppg_trace=ppg,
        sensor_hr=sensor_hr,
        timestamp_sec=time_sec,
    )

    uniform = prepare_uniform_ground_truth(ground_truth)
    result = build_subject_reference("subject1", ground_truth)

    assert np.allclose(uniform.sensor_hr_raw, 2.0)
    assert np.allclose(uniform.sensor_hr_qc, 130.0)
    assert uniform.sensor_wrap_corrected_sample_count == len(time_sec)
    assert np.allclose(result.windows["sensor_hr_raw_median_bpm"], 2.0)
    assert np.allclose(result.windows["sensor_hr_qc_median_bpm"], 130.0)
