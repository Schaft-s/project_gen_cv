from gencv.noise import VPSchedule
from gencv.schedules import (
    expected_nfe,
    karras_timesteps,
    timesteps_for_nfe,
    uniform_timesteps,
)


def test_uniform_length_and_order():
    ts = uniform_timesteps(19)
    assert len(ts) == 19
    assert ts == sorted(ts, reverse=True)
    assert len(set(ts)) == 19
    assert ts[-1] >= 0 and ts[0] < 1000


def test_karras_snaps():
    vp = VPSchedule.from_config()
    ts = karras_timesteps(19, vp)
    assert len(ts) == 19
    assert ts == sorted(ts, reverse=True)


def test_nfe_table():
    vp = VPSchedule.from_config()
    assert expected_nfe("heun", 10) == 19
    assert expected_nfe("ddim", 19) == 19
    for nfe in (9, 19, 39):
        for sampler in ("ddim", "dpm_pp_2m", "heun"):
            for kind in ("uniform", "karras"):
                ts = timesteps_for_nfe(nfe, sampler, kind, vp)
                k = {"ddim": nfe, "dpm_pp_2m": nfe, "heun": {9: 5, 19: 10, 39: 20}[nfe]}[sampler]
                assert len(ts) == k
