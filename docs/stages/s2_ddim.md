# S2 — Свой DDIM

Зависимость: S1.

## Цель

Реализовать DDIM `η=0` и численно совпасть с `DDIMScheduler` на штатной leading-сетке.

## Сделать

1. `src/gencv/samplers/ddim.py`: шаг с **явной** парой `(t, t_next)` и полный цикл.
2. Режим MATCH: `clip_sample` как у Diffusers. RESEARCH-флаг — без clip.
3. `tests/test_ddim_match.py`:
   - один шаг на замороженном `ε̂`;
   - короткая траектория, 4 шума × 20 шагов.
4. `scripts/s2_match_ddim.py` пишет `outputs/s2_match_ddim.json` (`max_abs_step`, `max_abs_traj`).
5. Не использовать `scheduler.timesteps` как способ задать нерегулярную сетку.

Допуски: шаг `< 1e-5`, траектория `< 1e-4` (fp32). Если MPS даёт больше — перепроверить те же тензоры на CPU и записать оба числа.

## Запуск

```bash
pytest tests/test_ddim_match.py -q
python scripts/s2_match_ddim.py
```

## Готово когда

- JSON сверки в пределах допуска.
- Есть свой DDIM, который принимает список меток/`(t, t_next)`.
- Агент может объяснить один переход по `docs/methods.md` §4.
