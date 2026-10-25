# โจทย์ 2: ระดับน้ำลำน้ำ 1–48 ชั่วโมง

แต่ละโมเดลอยู่คนละ Folder มี `README.md` (ทำอะไร ใช้ข้อมูลอะไร สถานะ) และ `model.py`

| Folder | โมเดล | กลุ่ม | สถานะ |
| --- | --- | --- | --- |
| [01_persistence](01_persistence/) | Persistence / Persistence + แนวโน้ม | Baseline | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [02_travel_time](02_travel_time/) | Travel-time / lag correlation | Hydrology | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [03_muskingum](03_muskingum/) | Muskingum routing | Hydrology | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [04_arx](04_arx/) | ARX / ARIMAX | Statistics (Time Series) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [05_lightgbm](05_lightgbm/) | LightGBM / XGBoost (multi-horizon, quantile) | Machine Learning (Gradient-Boosted Trees) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [06_lstm_seq2seq](06_lstm_seq2seq/) | LSTM / Encoder–decoder LSTM (Seq2Seq) | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [07_tft](07_tft/) | Temporal Fusion Transformer (TFT) | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [08_nhits](08_nhits/) | N-HiTS | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [09_hybrid_lightgbm_lstm](09_hybrid_lightgbm_lstm/) | Hybrid LightGBM + LSTM | Deep Learning (Neural Network) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [10_foundation_zero_shot](10_foundation_zero_shot/) | Chronos-2, TimesFM 2.5 (zero-shot) | Deep Learning (Foundation Model) | เขียนโค้ดแล้ว ยังไม่ได้ทดสอบ (ต้องติดตั้ง torch + แพ็กเกจของโมเดล และโหลด weights) |
| [11_hec_hms_ras](11_hec_hms_ras/) | HEC-HMS + HEC-RAS 1D | Hydrology (Physical) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) ส่วน HEC-RAS เต็มต้องใช้โปรแกรม HEC |
| [12_rod_psu_2d](12_rod_psu_2d/) | ผลจำลอง 2 มิติ ROD ม.อ. | Hydrology (Physical) | โค้ด Bathtub + ScenarioLookup ทดสอบกับข้อมูลจำลองแล้ว ใช้ผล ม.อ. ต้องขออนุญาตก่อน |
| [14_radar_nowcast](14_radar_nowcast/) | Radar QPE + pySTEPS nowcast | Radar Nowcasting | ทดสอบกับภาพจำลองแล้ว (รุ่น pySTEPS ต้องติดตั้ง pysteps) |
| [90_error_correction](90_error_correction/) | Error correction / Kalman | Post-processing (Data Assimilation) | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [91_rain_ensemble](91_rain_ensemble/) | Ensemble ฝนพยากรณ์หลายโมเดล | Post-processing | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
| [92_conformal](92_conformal/) | Conformal | Post-processing | ใช้งานได้ ทดสอบกับข้อมูลจำลองแล้ว (รอข้อมูลรายชั่วโมงจริง) |
