# Брифы этапов

Запускать агента **на один файл**. Порядок фиксированный.

| Файл | Этап |
|---|---|
| [s1_baseline.md](s1_baseline.md) | Diffusers, первая генерация |
| [s2_ddim.md](s2_ddim.md) | Свой DDIM + MATCH |
| [s3_adapter.md](s3_adapter.md) | VP-таблица и `x ↔ y` |
| [s4_heun.md](s4_heun.md) | Heun/EDM + MATCH |
| [s5_dpm.md](s5_dpm.md) | DPM-Solver++ 2M + MATCH |
| [s6_schedules.md](s6_schedules.md) | Uniform / Karras / NFE |
| [s7_search.md](s7_search.md) | Опоры и поиск сетки |
| [s8_experiments.md](s8_experiments.md) | Сравнение, перенос, FID |
| [s9_report.md](s9_report.md) | Картинки и таблицы для защиты |

Промпт агенту:

```
Реализуй только этап SX по docs/stages/sX_....md.
Сначала прочитай docs/plan.md, docs/methods.md и этот бриф.
Критерий готовности — секция «Готово когда». Не переходи к следующему этапу.
```
