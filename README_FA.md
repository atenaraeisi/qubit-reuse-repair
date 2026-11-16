> **Update (MATERIALIZER Ridge):** See `NEW_RIDGE_QUICKSTART_FA.md` for the latest integrated pipeline. Older checkpoint instructions below are retained for historical comparison.

# کمپین ۱۶۲ مداری CaQR — نسخهٔ Frozen

## انتخاب مجموعه

مخزن اصلی `ruadapt/CaQR` در مسیر `benchmarks/` و مخزن پژوهش قبلی در مسیر `benchmarks_compact/` هر دو ۱۶۸ QASM هم‌نام دارند. شش فایل آموزشی/تستی (`test.qasm` تا `test4.qasm` و `alu-v0_27_example.qasm`) حذف شده‌اند. **۱۶۲ مدار انتخاب شده‌اند**. نام‌ها قبل از دیدن نتایج این کمپین در فایل `campaign/benchmark_roster_162.csv` ثابت هستند.

مطابق `ml_split_v1/split_ledger.csv` قدیمی: ۳۸ فایل `known_train`، ۱۲ فایل `known_validation`، ۲ فایل `final_test_reserved` و **۱۱۰ فایل فاقد برچسب قبلی** (`unassigned`). برچسب `unassigned` به معنی «مستقل از آموزش» نیست؛ برای اثبات استقلال باید فایل، خانواده و منشأ همهٔ داده‌ها کنترل شوند.

دو مدار `wim_266.qasm` و `squar5_261.qasm` در این مرحله **اجرا نمی‌شوند**. تنها پس از Freeze شدن روش، پروتکل و انتخاب مدل می‌شود با پرچم `--release-final-test` از این قفل خارج شد؛ این کار نباید برای انتخاب روش و بازتنظیم آن‌ها انجام شود.

## روش اجرایی

۱. CaQR اصلی را از `CaQR-main/CaQR-main/main.py` اجرا می‌کنیم؛ فایل `_reuse_chain.txt` به `ReusePlan` تبدیل می‌شود و کیوبیت‌های فعال با زنجیرهٔ singleton تکمیل می‌شوند.
۲. Checker مستقل و سپس تولید همسایهٔ محدود انجام می‌شود. وضعیت‌های `ready`، `no_reuse`، `no_observed_neighbor`، `invalid_plan`، `caqr_failed` و ... ثبت می‌شوند. هیچ مداری به‌خاطر Gain بد حذف نمی‌شود.
۳. فقط موارد `ready` به دو Pipeline هدف `swap_count` و `routed_depth` می‌روند؛ از مدل‌های `one_step TRAIN_ONLY`، Search ثابت و Qiskit SABRE استفاده می‌شود.
۴. برای هر Seed، خروجی Original و Proposed اندازه‌گیری می‌شود؛ Final Guard صرفاً در همین Routing موردی، بهترین گزینه را حفظ می‌کند. Original Best-of-2 کنترل جداگانه است.
۵. گزارش per-circuit/per-objective وضعیت **هر ۱۶۲ مدار** را دارد؛ اعداد حذف‌شده/ناموفق به صفر تبدیل نمی‌شوند و آمار فقط از اجراهای تکمیل‌شده استخراج می‌شود.

> این مدل Ridge تاریخی با Materializer و Reference Pool متفاوتی آموزش دیده؛ نتایج صرفاً Exploratory Transfer هستند. مقایسهٔ اعداد این Pipeline با native CaQR بدون پروتکل کامپایل یکسان صحیح نیست. این خروجی، اثبات برتری روی DQCC نیست.

## نصب فایل‌ها

محتویات ZIP را **در ریشهٔ پروژهٔ `qubit-reuse-repair`** Extract کنید؛ پوشه‌های `campaign/` و `scripts/` و Notebook کنار فایل `pyproject.toml` قرار بگیرند. بستهٔ قبلی پروژه و `pilot_models/` باید همان‌جا موجود باشند. فقط سه فایل/پوشهٔ جدید اضافه می‌شود و کد Repair موجود جایگزین نمی‌شود.

کرنل Jupyter: `caqr-qiskit045-env`. Notebook `CaQR_162_benchmark_campaign.ipynb` مسیرهای قبلی سیستم شما را پیش‌فرض قرار داده.

## Command Line — Windows PowerShell

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$caqr = 'C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main'
python scripts\caqr_campaign.py roster --caqr-root "$caqr"
python scripts\caqr_campaign.py screen --caqr-root "$caqr" --batch-size 20
python scripts\caqr_campaign.py screen --caqr-root "$caqr" --batch-size 20
python scripts\caqr_campaign.py evaluate --batch-size 20 --seeds 200,211,223
python scripts\caqr_campaign.py summary
```

اجرای `screen` و `evaluate` را می‌توان تکرار کرد تا تمام کارهای باقی‌مانده تکمیل شوند. `--batch-size 0` یعنی **تمام کارهای باقی‌مانده** را همین بار اجرا کن. پیش‌فرض `evaluate` سه Seed مختلف است؛ برای اولین Smoke Test، `--batch-size 4` مناسب‌تر است. Seedها و پروتکل پس از اولین `evaluate` قفل می‌شوند؛ برای تغییر پروتکل، `--out-dir` جدید انتخاب کنید. از `--force` فقط برای تکرار آگاهانهٔ یک اجرا استفاده کنید.

## خروجی‌ها

- `caqr_162_results/frozen_roster.json`: هویت ثابت ۱۶۲ مدار، SHA-256 و تقسیم قبلی
- `caqr_162_results/screening_162.csv`: نتیجهٔ غربالگری هر ۱۶۲ مدار
- `caqr_162_results/per_circuit_seed.csv`: تا ۹۷۲ ردیف (۱۶۲ مدار × ۲ معیار × ۳ Seed)
- `caqr_162_results/per_circuit_objective.csv`: دقیقاً **۳۲۴ ردیف**، هر مدار × هر معیار
- `caqr_162_results/summary_162.json`: تعداد کامل/ناقص، میانگین Macro فقط روی ردیف‌های کامل، تفکیک Split
- `caqr_162_results/evaluation/cases/<circuit>/<objective>/seed_<seed>/routed/routing_report.json`: گزارش خام، مدار Route‌شده و کنترل Best-of-2

اگر `no_reuse` یا `no_observed_neighbor` باشد، Gain فیزیکی **اندازه‌گیری نشده** و در CSV خالی می‌ماند؛ هیچ عدد ساختگی صفر جایگزین نمی‌شود. در تحلیل نهایی می‌توان یک معیار پوشش All-eligible با Gain ساختاری صفر تعریف و جداگانه گزارش کرد.

## اولویت علمی

برای Abstract، تعداد کل ازپیش‌انتخاب‌شده، تعداد موفق در CaQR، تعداد دارای Plan معتبر و قابل Repair و تعداد Routing کامل را دقیق گزارش کنید. درصد Gain پس از Final Guard به‌تنهایی نشان‌دهندهٔ قدرت Predictor نیست؛ Gain قبل از Guard و برتری در مقابل Original Best-of-2 را هم بگویید. متوسط‌ها را با گروه‌بندی مدار و بعد Seed محاسبه کنید، و از ترکیب `known_train` با Validation برای ادعای generalization خودداری کنید.
