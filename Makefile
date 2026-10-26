.PHONY: s1 s2 s3 s4 s5 s6 s7 s8-mse s8-fid s9 test

PYTHON ?= python

s1:
	$(PYTHON) scripts/s1_baseline.py --n 8 --steps 20 --seed 0

s2:
	$(PYTHON) -m pytest tests/test_ddim_match.py -q
	$(PYTHON) scripts/s2_match_ddim.py

s3:
	$(PYTHON) -m pytest tests/test_noise.py tests/test_denoiser.py -q

s4:
	$(PYTHON) -m pytest tests/test_heun_match.py -q
	$(PYTHON) scripts/s4_match_heun.py

s5:
	$(PYTHON) -m pytest tests/test_dpm_match.py -q
	$(PYTHON) scripts/s5_match_dpm.py

s6:
	$(PYTHON) -m pytest tests/test_schedules.py -q
	$(PYTHON) scripts/s6_check_nfe.py

s7:
	$(PYTHON) scripts/s7_make_banks.py
	$(PYTHON) scripts/s7_references.py
	$(PYTHON) scripts/s7_search.py

s8-mse:
	$(PYTHON) scripts/s8_mse_table.py
	$(PYTHON) scripts/s8_figures.py

s8-fid:
	$(PYTHON) scripts/s8_fid.py --n 2000 --nfe 19

s9:
	$(PYTHON) scripts/s9_demo.py

test:
	$(PYTHON) -m pytest tests -q
