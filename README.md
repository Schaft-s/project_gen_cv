# Сэмплеры диффузии и подбор расписания

Итоговый проект курса: свои **DDIM**, **DPM-Solver++ 2M** и **Heun (EDM)** на `google/ddpm-cifar10-32`, сверка с Diffusers, сравнение сеток при одинаковом NFE и подбор уровней шума.

**Отчёт:** [`docs/report.md`](docs/report.md)

PixArt-α из задания не делался: это дополнительный пункт, обязательный объём на CIFAR-10 закрыт.

## Что получилось

При NFE 19 лучший сэмплер — DPM-Solver++ 2M на сетке Karras. Найденное правило чуть улучшает все три метода и переносится с DPM на DDIM и Heun, но лидера не меняет.

| | MSE к DDIM-1000 (Karras / found) | FID, 2000 картинок (Karras / found) |
|---|---|---|
| DDIM | 0.01015 / 0.00925 | 40.15 / 39.33 |
| DPM-Solver++ 2M | **0.000751 / 0.000653** | **29.10 / 28.79** |
| Heun | 0.00560 / 0.00543 | 31.39 / 30.86 |

Сверка с Diffusers: DDIM и Heun — max abs 0, DPM — 2·10⁻⁶.

## Структура

```
docs/report.md          отчёт
docs/task.md            формулировка задания
docs/methods.md         формулы
docs/results.md         числа и пути к файлам
src/gencv/samplers/     DDIM, Heun, DPM-Solver++ 2M
scripts/                прогоны S1–S9
tests/                  сверка и юнит-тесты
configs/default.yaml    замороженные гиперпараметры
data/schedules/         найденная сетка и контроли
```

Заметки для разработки этапов: `docs/plan.md`, `docs/stages/`, `AGENTS.md`.

## Установка (macOS, MPS)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
python -m pytest tests -q
```

Версии рабочего прогона: torch 2.14.1, diffusers 0.32.2, transformers 4.57.6. На MPS генератор для Diffusers — CPU.

## Воспроизведение

Демо (нужны `data/schedules/found.yaml` и веса модели):

```bash
python scripts/s9_demo.py
```

Полный пайплайн, как считались числа отчёта:

```bash
python scripts/s1_baseline.py --n 8 --steps 20
python scripts/s2_match_ddim.py
python scripts/s4_match_heun.py
python scripts/s5_match_dpm.py
python scripts/s6_check_nfe.py
python scripts/s7_make_banks.py
python scripts/s7_references.py          # ~13 мин
python scripts/s7_search.py
python scripts/s8_mse_table.py           # ~11 мин
python scripts/s8_figures.py
python scripts/s8_failures.py
python scripts/s8_fid.py --n 2000 --nfe 19   # ~22 мин
python scripts/s9_demo.py
```

Тяжёлые банки шумов, опоры `.pt` и папки FID-картинок в git не лежат: их заново строят `s7_make_banks.py` и `s8_fid.py` (тот же seed). Маленькие таблицы и JSON сверки — в `outputs/`.
