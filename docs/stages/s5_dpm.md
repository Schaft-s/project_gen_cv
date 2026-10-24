# S5 — DPM-Solver++ 2M

Зависимость: S3.

## Цель

Многошаговый DPM-Solver++ порядка 2, `solver_type=midpoint`, MATCH с `DPMSolverMultistepScheduler`.

## Сделать

1. `src/gencv/samplers/dpm_solver_pp.py` по `docs/methods.md` §6.
2. История `x̂₀` очищается в начале каждой генерации.
3. Первый шаг и переход к нулю — 1-й порядок. Для `exp(-h)-1` использовать `-expm1(-h)`.
4. MATCH обязан повторить `lower_order_final=True` и `final_sigmas_type="zero"`: при коротких сетях Diffusers понижает порядок в конце.
5. Не путать с `solver_type="heun"` этого же класса.
6. `outputs/s5_match_dpm.json` + `tests/test_dpm_match.py`.
7. NFE = `K`.

## Запуск

```bash
pytest tests/test_dpm_match.py -q
python scripts/s5_match_dpm.py
```

## Готово когда

Сверка в допуске. Три сэмплера зовут одну и ту же обёртку U-Net. Общий интерфейс генерации: `(z, timesteps_or_sigmas, nfe_counter) -> x`.
