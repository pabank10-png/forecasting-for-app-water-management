# โจทย์ 1: Inflow อ่างรายเดือน (1–3 และ 12 เดือน)

แต่ละโมเดลอยู่คนละ Folder มี `README.md` (ทำอะไร ใช้ข้อมูลอะไร สถานะ) และ `model.py`

| Folder | โมเดล | กลุ่ม | สถานะ |
| --- | --- | --- | --- |
| [01_climatology](01_climatology/) | ค่าเฉลี่ยเดือนเดียวกัน (Climatology) | Baseline | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [02_seasonal_naive](02_seasonal_naive/) | Seasonal naive | Baseline | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [03_sarimax](03_sarimax/) | SARIMAX | Statistics (Time Series) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [04_ets](04_ets/) | ETS (Holt-Winters) | Statistics (Time Series) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [05_lightgbm_quantile](05_lightgbm_quantile/) | LightGBM quantile | Machine Learning (Gradient-Boosted Trees) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [06_xgboost_quantile](06_xgboost_quantile/) | XGBoost quantile | Machine Learning (Gradient-Boosted Trees) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [07_nbeats](07_nbeats/) | N-BEATS / NBEATSx | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [08_nhits](08_nhits/) | N-HiTS | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [09_lstm](09_lstm/) | LSTM / Encoder–decoder LSTM | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [10_chronos2](10_chronos2/) | Chronos-2 (Amazon) | Deep Learning (Foundation Model) | เขียนโค้ดแล้ว ยังไม่ได้ทดสอบ (ต้องติดตั้ง torch + แพ็กเกจของโมเดล และโหลด weights) |
| [11_timesfm](11_timesfm/) | TimesFM 2.5 (Google) | Deep Learning (Foundation Model) | เขียนโค้ดแล้ว ยังไม่ได้ทดสอบ (ต้องติดตั้ง torch + แพ็กเกจของโมเดล และโหลด weights) |
| [12_moirai](12_moirai/) | Moirai 2.0 (Salesforce) | Deep Learning (Foundation Model) | เขียนโค้ดแล้ว ยังไม่ได้ทดสอบ (ต้องติดตั้ง torch + แพ็กเกจของโมเดล และโหลด weights) |
| [13_chronos2_finetune](13_chronos2_finetune/) | Chronos-2 fine-tune | Deep Learning (Foundation Model) | เขียนโค้ดแล้ว ยังไม่ได้ทดสอบ (ต้องติดตั้ง torch + แพ็กเกจของโมเดล และโหลด weights) |
| [90_conformal](90_conformal/) | Conformal prediction | Post-processing | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [91_ensemble](91_ensemble/) | Ensemble | Post-processing | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [95_water_balance](95_water_balance/) | Water Balance (runSc) | Water Balance | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
