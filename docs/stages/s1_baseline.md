# S1 — Baseline Diffusers

## Цель

Загрузить `google/ddpm-cifar10-32`, сгенерировать детерминированные CIFAR-картинки штатным DDIM (`η=0`) и зафиксировать окружение.

## Сделать

1. Дописать `scripts/s1_baseline.py` (черновик уже есть): pipeline Diffusers, `DDIMScheduler.from_config` исходного scheduler, `eta=0`, fp32.
На MPS `torch.Generator` для Diffusers держать на CPU.

2. Сохранить:
   - сетку PNG;
   - отдельные PNG;
   - `metadata.json`: версии `torch`/`diffusers`, device, seed, `num_inference_steps`, полный `scheduler.config`;
   - `throughput.json`: секунд на батч после прогрева (с `gencv.device.sync`).
3. Повторить запуск с тем же seed — байты PNG должны совпасть.
4. Распечатать `scheduler_config` модели и убедиться: `num_train_timesteps=1000`, linear β `1e-4…0.02`, `clip_sample=true`.

## Не делать

Свои сэмплеры, FID, смену модели.

## Запуск

Локально:

```bash
python scripts/s1_baseline.py --n 8 --steps 20 --seed 0
```

## Готово когда

- В `outputs/s1_baseline/` есть `grid.png`, `metadata.json`, `throughput.json`.
- Повторный запуск даёт те же картинки.
- На картинках видны объекты CIFAR, не шум.
- В README можно проставить галочку S1.
