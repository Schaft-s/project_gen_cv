# S6 — Сетки и протокол NFE

Зависимость: S2, S4, S5.

## Цель

Две стандартные сетки, привязка Karras к дискретным `t`, единый RESEARCH-протокол.

## Сделать

1. `src/gencv/schedules.py`: `uniform_timesteps(K)`, `karras_timesteps(K, rho=7)`.
2. Проверки: строгая убывание `t`, уникальность, длина `K`, диапазон `0…999`.
3. Для каждого сэмплера функция `timesteps_for_nfe(nfe)` с таблицей из `docs/experiments.md`.
4. RESEARCH: `clip_sample=False`, DPM `thresholding=False`, финальный clamp только при сохранении PNG.
5. `scripts/s6_check_nfe.py`: прогнать 1 картинку на всех 3×2×3 ячейках, записать фактический NFE. Любое несовпадение — ошибка.
6. Тесты на построение сеток без модели.

## Запуск

```bash
pytest tests/test_schedules.py -q
python scripts/s6_check_nfe.py
```

## Готово когда

`outputs/s6_nfe.json` содержит 18 строк с ожидаемым и фактическим NFE, все совпали.
