# Генерация из диффузионных моделей: сэмплеры и расписание

Свои **DDIM**, **DPM-Solver++ 2M** и **Heun (EDM)** на `google/ddpm-cifar10-32`, сверка с Diffusers, сравнение сеток при одинаковом NFE и подбор уровней шума.

- Задание: [`docs/task.md`](docs/task.md)
- План: [`docs/plan.md`](docs/plan.md)
- Формулы: [`docs/methods.md`](docs/methods.md)
- Результаты: [`docs/results.md`](docs/results.md)

## Установка (macOS, MPS)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
```

Версии рабочего прогона: torch 2.14.1, diffusers 0.32.2, transformers 4.57.6. На MPS генератор для Diffusers — CPU.

## Что уже прогнано локально

```bash
python -m pytest tests -q
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
python scripts/s9_demo.py
```

Демо без полного пайплайна: `python scripts/s9_demo.py` (нужны `data/schedules/found.yaml` и веса модели).

## Colab — только FID

Локально FID **не запускать**. В Colab:

1. Залить репозиторий, открыть `notebooks/colab_runner.ipynb`.
2. `STAGE = "s8_fid"`, `FID_N = 2000`.
3. Нужны уже лежащие в репо `data/schedules/found.yaml` и пересоздаваемый тем же seed банк `fid` (`scripts/s7_make_banks.py`).

Подробности: [`docs/colab.md`](docs/colab.md).

## Статус

- [x] S1 baseline Diffusers
- [x] S2 свой DDIM (max abs 0 vs Diffusers)
- [x] S3 адаптер `x ↔ y`
- [x] S4 Heun (max abs 0)
- [x] S5 DPM-Solver++ 2M (max abs 2e-6)
- [x] S6 сетки, NFE 9/19/39
- [x] S7 опоры и поиск расписания
- [x] S8 MSE, графики, визуальные сетки
- [ ] S8 FID — **Colab**
- [x] S9 demo и `docs/results.md`
