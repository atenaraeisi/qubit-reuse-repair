# نسخه کامل Repair با Ridge جدید CaQR/DQCC

## کدام نسخه؟
این بسته از نسخه کامل `CaQR_162_campaign_English_Notebook_Full_Project.zip` ساخته شده است؛ **نه** `qubit-reuse-repair-phase6.zip` که فاقد اسکریپت‌های Pilot و کمپین ۱۶۲ مداری است.

دو مدل جدید در `pilot_models/` ذخیره شده‌اند:
- `swap_count_one_step_MATERIALIZER.json`
- `routed_depth_one_step_MATERIALIZER.json`

مدل‌های قدیمی `*_TRAIN_ONLY.json` برای مقایسه نگه داشته شده‌اند. کلاس جدید `MaterializerDeltaRidge` ابتدا Missingهای ویژگی خام هر Plan را با میانه Training جایگزین می‌کند، سپس اختلاف **Candidate - Original** را می‌سازد، با mean/std همان Training استاندارد می‌کند و در نهایت با ۲۵ وزن و intercept درصد Gain پیش‌بینی می‌کند. هیچ نرمال‌سازی per-circuit قدیمی روی ورودی این مدل اعمال نمی‌شود.

**نکتهٔ روش‌شناسی:** فایل آموزش دوستت کد Feature Extractor و داده‌های `pairs.jsonl` را شامل نمی‌شود. پیاده‌سازی ویژگی‌های جدید در این بسته از مسیر `StructuralFeatureExtractor` + `StaticPlanMaterializer` + `normalize_reset_for_depth` استفاده می‌کند. برابر بودن دقیق آن با Feature Extractor زمان Training به صورت مستقل تأیید نشده؛ تا وقتی نمونه‌های مرجع استخراج ویژگی‌ها یا کد آموزش دریافت نشود نباید این برابری قطعی فرض شود.

## نصب و تست

1. از پوشه فعلی پروژه بکاپ بگیر.
2. محتوای `qubit-reuse-repair/` داخل ZIP را **روی همان پوشه قبلی پروژه** قرار بده، نه داخل یک پوشه تو در تو. نتایج قبلی را حذف نکن.
3. در PowerShell و محیط مجازی Qiskit اجرا کن:

```powershell
cd "C:\Users\ASUS\Documents\Quantum computing\qubit-reuse-repair"
python -m pytest -q
```

روی محیط تولیدکنندهٔ بسته، ۹۲ تست پاس و ۱۰ تست اختیاری وابسته به Qiskit رد شده‌اند (Qiskit در آن محیط نصب نبود). روی سیستم شما تست‌های Qiskit باید هم اجرا شوند.

## بدون Notebook: اجرای مجدد روی همان ۱۶۰ مدار

Screening قدیمی را دوباره انجام نمی‌دهیم؛ فقط رکوردهای بدون خروجی Routing از آن کپی می‌شوند. Materializer نسبت به نسخه قبلی Patch مربوط به Reset دارد، پس Routing باید در پوشهٔ جدید محاسبه شود.

```powershell
python run_caqr_new_ridge.py prepare
python run_caqr_new_ridge.py evaluate
python run_caqr_new_ridge.py summary
```

یا بعد از تست محیط:

```powershell
python run_caqr_new_ridge.py all
```

خروجی جدید:

`caqr_162_results_materializer_ridge_v2/`

این نسخه `--batch-size 0` را به معنی همه Jobهای باقی‌مانده به کار می‌برد. برای اجرای تدریجی می‌توان گفت:

```powershell
python run_caqr_new_ridge.py evaluate --batch-size 15
```

**فقط نتایج Routing نسخه جدید را با یکدیگر ترکیب کن.** نتایج قبلیِ `caqr_162_results` باید جدا باقی بمانند. `evaluation_protocol.json` هش‌های کد، مدل و تنظیمات را قفل می‌کند و از ترکیب ناخواسته جلوگیری می‌کند. `wim_266` و `squar5_261` تا پایان انتخاب روش کنار گذاشته می‌شوند.

## DQCC و ورودی‌های دستی

برنامه `scripts/fast_pilot_batch.py` حالا به صورت پیش‌فرض `--model-set materializer` دارد. ورودی باید `manifest.json` شامل Generator واقعی، QASM اصلی و Plan اصلی در قالب chainهای مرتب باشد؛ صرف نوشتن DQCC در `generator`، آن Plan را DQCC نمی‌کند.

برای تک‌آزمایش:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python scripts/fast_ridge_pilot.py --qasm path/to/circuit.qasm --plan path/to/plan.json --generator DQCC --checkpoint pilot_models/swap_count_one_step_MATERIALIZER.json --objective swap_count --out-dir results/dqcc_swap --seed 200 --control-second-seed 1200
```

برای مدل قدیمی می‌توان `--model-set legacy` به اسکریپت Batch یا CaQR campaign داد (با **پوشهٔ خروجی متفاوت**).

## محدودیت‌های تأییدنشده
- مدل جدید روی ۱۴ مدار و ۱۹۹ زوج آموزش دیده؛ کارایی‌اش در Beam چندمرحله‌ای جداگانه نیاز به سنجش دارد.
- تغییر Materializer ممکن است ویژگی‌ها را برای بعضی Planها تغییر داده باشد. برابری با Pipeline آموزش به‌خصوص برای `norm_depth` باید با داده‌های Trainer بررسی شود.
- این بسته آموزش را دوباره انجام نمی‌دهد؛ وزن‌ها و scalerها را دقیقاً از Checkpointهای ارسالی می‌خواند.
- بهبود بالاتر از Best-of-2 و عملکرد روی Generator دوم هنوز با این بسته ثابت نشده‌اند؛ باید با Routing واقعی سنجیده شوند.
