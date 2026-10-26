# Colab — запасной канал

Обязательный объём, включая FID на 2000 картинках, уже посчитан на Mac (MPS + Inception на CPU). Этот ноутбук нужен, только если надо пересчитать FID на CUDA или нет локального MPS.

`data/schedules/found.yaml` должен быть в клоне. Банк `fid.pt` в git не лежит — его пересоздаёт тот же seed.

## Шаги

1. Залить репозиторий, GPU (T4 достаточно).
2. `notebooks/colab_runner.ipynb`: `STAGE = "s8_fid"`, `FID_N = 2000`.
3. Ноутбук ставит зависимости **без** переустановки torch (`transformers<5` уже в `requirements.txt`).
4. Перед FID, если нет `data/noise_banks/fid.pt`:

```bash
python scripts/s7_make_banks.py
python scripts/s8_fid.py --n 2000 --nfe 19
```

5. Скачать `outputs/s8/fid.json` обратно в репозиторий.
