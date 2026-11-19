"""Unit tests for smoothed hypsometry and continuous BPDC / dzsnow/dt helpers."""

import numpy as np

from esbmtk import Model
from esbmtk.carbonate_chemistry import compute_bpdc_and_dzsnow
from esbmtk.sealevel import hypsometry


def test_smoothed_hypsometry_positivity_and_conservation():
    """Verify PCHIP smoothing yields strictly positive a'(z) and conserves area."""
    m_smooth = Model(stop="1 yr", max_timestep="1 yr", register="local")
    hyp_smooth = hypsometry(name="hyp_s", model=m_smooth, register=m_smooth)

    m_raw = Model(stop="1 yr", max_timestep="1 yr", register="local")
    hyp_raw = hypsometry(
        name="hyp_r", model=m_raw, register=m_raw, smooth_step=0
    )

    area_s = hyp_smooth.get_lookup_table_area()
    area_dz_s = -hyp_smooth.get_lookup_table_area_dz()
    area_r = hyp_raw.get_lookup_table_area()

    z_deep = hyp_smooth.max_seafloor_depth
    assert z_deep == hyp_raw.max_seafloor_depth
    assert np.all(area_dz_s[: z_deep + 1] > 0.0)
    assert np.isclose(area_s[0], area_r[0], rtol=1e-12)

    for z1, z2 in [(200, 6000), (1000, 4800), (3700, z_deep)]:
        assert np.isclose(
            np.sum(area_dz_s[z1:z2]), area_s[z1] - area_s[z2], rtol=1e-12
        )


def test_compute_bpdc_and_dzsnow_edge_cases():
    """Verify integer-bin BPDC, interpolated a_prime, zero-flux/floor guards, and clamping."""
    AD = 3.36e14
    area_dz = np.full(6001, 1e10, dtype=float)
    csat = np.linspace(4.0e-5, 1.2e-4, 6001, dtype=float)

    # 1. zsnow <= zcc resets/relaxes toward zcc with zero BPDC
    bpdc, dzdt = compute_bpdc_and_dzsnow(
        4000, 3950.0, 6000, 8.0e-5, 8840.0, 529.0, AD, area_dz, csat
    )
    assert bpdc == 0.0
    assert dzdt == 50.0

    # 2. zcc < zsnow < zcc + 1 preserves 1 m deadband (0.0, 0.0)
    bpdc, dzdt = compute_bpdc_and_dzsnow(
        4000, 4000.5, 6000, 5.0e-5, 8840.0, 529.0, AD, area_dz, csat
    )
    assert bpdc == 0.0
    assert dzdt == 0.0

    # 3. Oversaturated (co3 >= csat everywhere) returns (0.0, 0.0) even if a'(z)==0
    area_dz_zero = area_dz.copy()
    area_dz_zero[5000] = 0.0
    bpdc, dzdt = compute_bpdc_and_dzsnow(
        4000, 5000.0, 6000, 1.0, 8840.0, 529.0, AD, area_dz_zero, csat
    )
    assert bpdc == 0.0
    assert dzdt == 0.0

    # 4. Linearly interpolated a_prime on sloped area_dz at zsnow = 5000.5
    area_dz_sloped = np.linspace(1e10, 2e10, 6001, dtype=float)
    bpdc, dzdt = compute_bpdc_and_dzsnow(
        4000, 5000.5, 6000, 5.0e-5, 8840.0, 529.0, AD, area_dz_sloped, csat
    )
    expected_a_prime = 0.5 * (area_dz_sloped[5000] + area_dz_sloped[5001])
    assert bpdc > 0.0
    assert np.isclose(dzdt, -bpdc / (expected_a_prime * 529.0), rtol=1e-12)

    # 5. Zero-area bin with positive BPDC hits 1e-10 * AD floor without ZeroDivisionError
    area_dz_hole = area_dz.copy()
    area_dz_hole[5000] = 0.0
    area_dz_hole[5001] = 0.0
    bpdc, dzdt = compute_bpdc_and_dzsnow(
        4000, 5000.5, 6000, 5.0e-5, 8840.0, 529.0, AD, area_dz_hole, csat
    )
    assert bpdc > 0.0
    assert np.isfinite(dzdt) and dzdt < 0.0

    # 6. zsnow >= zmax clamps safely without IndexError
    bpdc_c, dzdt_c = compute_bpdc_and_dzsnow(
        4000, 6500.0, 6000, 5.0e-5, 8840.0, 529.0, AD, area_dz, csat
    )
    assert bpdc_c > 0.0 and np.isfinite(dzdt_c)
