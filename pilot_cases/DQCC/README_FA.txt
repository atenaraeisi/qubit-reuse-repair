برای DQCC باید خروجی واقعی همان Generator را به یک ordered-chain Plan تبدیل کنیم.
این پوشه نمونهٔ ساختگی ندارد. فایل Plan باید فقط شامل شناسه های logical qubit
مدار اصلی باشد و هر logical active qubit دقیقاً یکبار در chainها حضور داشته باشد.
نام فایل qasm و plan را در DQCC manifest وارد کن. گزینه generator="DQCC"
به خودی خود ثابت نمی کند که Plan واقعاً با DQCC تولید شده: provenance و نسخه
کد DQCC را همراه گزارش ذخیره کن.
