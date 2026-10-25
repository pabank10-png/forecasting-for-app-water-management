# โจทย์ 1B: Inflow อ่างรายวัน 1–7 วัน (bridge)

แต่ละโมเดลอยู่คนละ Folder มี `README.md` (ทำอะไร ใช้ข้อมูลอะไร สถานะ) และ `model.py`

| Folder | โมเดล | กลุ่ม | สถานะ |
| --- | --- | --- | --- |
| [01_persistence](01_persistence/) | Persistence | Baseline | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [02_day_climatology](02_day_climatology/) | ค่าเฉลี่ยวันเดียวกันในอดีต | Baseline | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [03_gr4j](03_gr4j/) | GR4J | Hydrology (Conceptual) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [04_hbv](04_hbv/) | HBV | Hydrology (Conceptual) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [05_lightgbm_daily](05_lightgbm_daily/) | LightGBM quantile รายวัน | Machine Learning (Gradient-Boosted Trees) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [06_lstm_seq2seq](06_lstm_seq2seq/) | LSTM / Seq2Seq รายวัน | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [07_chronos2_exog](07_chronos2_exog/) | Chronos-2 + ฝนพยากรณ์ | Deep Learning (Foundation Model) | เขียนโค้ดแล้ว ยังไม่ได้ทดสอบ (ต้องติดตั้ง torch + แพ็กเกจของโมเดล และโหลด weights) |
| [90_rain_source_ensemble](90_rain_source_ensemble/) | รันทุกโมเดลด้วยฝนแต่ละค่าย | Post-processing | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
| [95_water_balance_daily](95_water_balance_daily/) | Water Balance รายวัน | Water Balance | ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว |
