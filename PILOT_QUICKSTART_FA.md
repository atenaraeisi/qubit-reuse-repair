> **Update (MATERIALIZER Ridge):** See `NEW_RIDGE_QUICKSTART_FA.md` for the latest integrated pipeline. Older checkpoint instructions below are retained for historical comparison.

# Pilot سریع با Ridge موجود — بدون آموزش مجدد

## وضعیت

- نرم‌افزار Phase 6 توسعه داده شده تا `TRAIN_ONLY` checkpointهای Ridge به Search وصل شوند.
- در `pilot_models/` دو فایل **واقعاً آموزش‌دیده و ذخیره‌شده** از `ml_ridge_multistep25_v1` قرار دارند؛ یکی SWAP و یکی Routed Depth، هر دو `one_step`.
- وزن‌ها روی ۲۵ feature اصلی آموزش داده شده‌اند؛ ۲۳ ویژگی ساختاری از منطق فعلی، و `logical_depth` / `norm_depth` از **materializer جدید** همراه با reset lowering تاریخی می‌آیند. بنابراین توزیع Featureها دقیقاً مشابه داده‌های Training نیست.
- آمار نرمال‌سازی در هر اجرا یک بار، از Original و همسایه‌های معتبرِ تک‌گام ساخته می‌شود و **قبل از Search قفل** می‌شود. این Reference Pool همان Catalog تاریخی Training نیست. نتایج اکتشافی‌اند.
- هیچ Routing یا Label فیزیکی در Search/Feature Selection دخیل نیست. Routing فقط بعد از فریز شدن Plan رخ می‌دهد. `route_guarded` بهترین مدار Route‌شده از Original/Repaired را تحت Seed یکسان انتخاب می‌کند و Best-of-2 Original کنترل اضافی است.
- فایل‌های `GidNET` در `pilot_cases` از `reuse_audit_fixed/fixed_full/original_plans.csv` هستند. `source=CaQR` در CSV قدیمی یعنی **منبع بنچمارک** و نه Generator. هرگز آن‌ها را به عنوان نتیجه CaQR گزارش نکنید.

## ۱. نصب و تست — Windows PowerShell

پوشه قبلی را Back Up کنید و این ZIP را Extract کنید؛ محیط `caqr-qiskit045-env` فعال باشد.

```powershell
cd 'C:\Users\ASUS\Documents\Quantum computing\qubit-reuse-repair'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
```

اگر `pip install -e` به‌خاطر دسترسی به setuptools خطا داد، برای اجرای تست و اسکریپت‌ها همین `PYTHONPATH` کافی است؛ نصب مجدد پکیج لازم نیست.

## ۲. Smoke Test با Planهای موجود GidNET (نه CaQR)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_gidnet_smoke_qasm.ps1
python scripts\fast_pilot_batch.py `
  --manifest pilot_cases\gidnet_smoke_manifest.json `
  --objectives swap_count,routed_depth `
  --arm one_step --policy random_beam `
  --topology heavy_hex7 `
  --seed 200 --control-second-seed 201 `
  --out-dir pilot_gidnet_smoke
```

خروجی:
- `pilot_gidnet_smoke/pilot_rows.csv` — نتایج همهٔ آزمون‌ها حتی موارد Failure
- `pilot_gidnet_smoke/pilot_summary.json` — جمع‌بندی توصیفیِ بسیار محدود
- فولدر هر مدار شامل `routing_report.json` و فایل‌های `*_routed.qasm` خواهد بود.

**نکته:** اگر در یک مدار فضای همسایگی Reuse خالی باشد، مدل اجرا نمی‌شود و مورد به‌عنوان Failure/No alternative در گزارش ثبت می‌شود؛ نتایج ساختگی تولید نمی‌کنیم.

## ۳. آزمایش روی خروجی واقعی CaQR و DQCC

برای هر مورد باید این دو فایل وجود داشته باشد:
1. `original.qasm` — مدار منطقی **قبل از reuse** (Static QASM 2.0، با gateهای واحدی و measurement نهایی)
2. `plan.json` — ordered reuse chains خروجی واقعی Generator، مثال `[[0,4],[1,3],[2]]`

Plan باید با شناسه‌های کیوبیت منطقی مدار QASM اصلی نوشته شود. اگر Generator ایندکس‌ها را تغییر داده، باید mapping معکوس درست شود. ورودی Native این دو Generator هنوز خودکار parse نمی‌شود.

```powershell
python scripts\fast_ridge_pilot.py `
  --qasm 'C:\path\to\original.qasm' `
  --plan 'C:\path\to\actual_caqr_plan.json' `
  --generator CaQR `
  --checkpoint pilot_models\swap_count_one_step_TRAIN_ONLY.json `
  --arm one_step --objective swap_count `
  --policy random_beam --topology heavy_hex7 `
  --seed 200 --control-second-seed 201 `
  --out-dir pilot_real_caqr_swap
```

برای DQCC: همان دستور با `--generator DQCC` و مسیر Plan مربوط به DQCC. برای بهینه‌سازی Depth، `--objective routed_depth` و Checkpoint `routed_depth_one_step_TRAIN_ONLY.json`.

برای چند مدار، ابتدا مسیرهای واقعی را در `pilot_cases/real_generators_manifest_example.json` قرار دهید، سپس `fast_pilot_batch.py --manifest ...` را اجرا کنید.

**Overlaps:** اگر اسم QASM در Training Checkpoint وجود داشته باشد، اجرای اصلی Fail Closed می‌شود. برای Smoke توسعه‌ای تنها می‌توان `--allow-training-circuit` داد، اما نتیجه دیگر Independent نیست.

## تفسیر نتایج

- همهٔ درصدهای Pilot مربوط به **پروتکل کامپایل جدید و همان Seedهای Route** هستند.
- وزن‌ها و پارامترهای مدل قبلی Freeze شده‌اند ولی Feature Distribution با Training تفاوت دارد؛ پس درصدهای Pilot را Proof of Generalization تلقی نکنید.
- Guard هیچ Harm اندازه‌گیری‌شده‌ای در تابع هدف همان دو Routing نهایی نمی‌گذارد، اما قدرت Predictor را اثبات نمی‌کند. بخش `candidate_route` در `routing_report.json` برای بررسی Harm خود Predictor است.
- برای Abstract امشب فقط اگر روی **Plan واقعی CaQR/DQCC** نتیجه داشته باشیم، آن را با تعداد مدارها، تنظیمات و قید Preliminary ذکر می‌کنیم.
