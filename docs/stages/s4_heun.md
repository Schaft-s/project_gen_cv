# S4 — Heun по EDM

Зависимость: S3.

## Цель

Детерминированный Heun (`S_churn=0`) в y-координатах и MATCH с `HeunDiscreteScheduler`.

## Сделать

1. `src/gencv/samplers/heun.py` по `docs/methods.md` §5.
2. Старт: `y = z * sqrt(1 + s_start²)`, `z` — тот же x-шум, что у DDIM.
3. Последний переход к 0 — Эйлер, 1 NFE. Иначе 2 NFE на шаг.
4. MATCH: scheduler из конфига модели, снять его `sigmas`, учесть `scale_model_input` и удвоенный цикл Diffusers. Сверять итоговый **x** (перевести `y` обратно) или работать целиком в том пространстве, в котором крутится pipeline.
5. `outputs/s4_match_heun.json` + `tests/test_heun_match.py`.
6. Юнит: при `K` положительных уровнях NFE = `2K-1`.

## Запуск

```bash
pytest tests/test_heun_match.py -q
python scripts/s4_match_heun.py
```

## Готово когда

Сверка в допуске из `configs/default.yaml`. Модель не вызывается при `s=0`. NFE совпадает с формулой.
