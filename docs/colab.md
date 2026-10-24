# Colab — только FID

Опоры 1000 NFE и MSE уже посчитаны на Mac. Здесь нужен **FID** шести конфигураций при NFE 19.

`data/schedules/found.yaml` должен быть в клоне. Банк `fid.pt` в git не лежит — его пересоздаёт тот же seed.

## Шаги

1. Залить репозиторий, GPU (T4 достаточно).
2. `notebooks/colab_runner.ipynb`: `STAGE = "s8_fid"`, `FID_N = 2000` (5000, если L4/A100 и есть время).
3. Ноутбук ставит зависимости **без** переустановки torch (`transformers<5` уже в `requirements.txt`).
4. Перед FID, если нет `data/noise_banks/fid.pt`:

```bash
python scripts/s7_make_banks.py
python scripts/s8_fid.py --n 2000 --nfe 19
```

5. Скачать `outputs/s8/fid.json` обратно в репозиторий.

На MPS этот скрипт не запускать: Inception в clean-fid там медленный и плохо воспроизводимый.
