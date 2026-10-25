# forecasting-for-app-water-management

ส่วนคำนวณพยากรณ์ของ Water Intelligence (inflow อ่างเก็บน้ำ และระดับน้ำลำน้ำ) รันด้วย GitHub Actions แล้วส่งผลขึ้น Supabase ให้เว็บอ่าน

**repo นี้เก็บแค่ code** ข้อมูลอยู่ที่ `../pilot/forecasting/1-reservoir-inflow/data/csv` บนเครื่อง (ตั้ง `FORECAST_DATA_DIR` เพื่อชี้ที่อื่นได้) และจะอยู่ใน Supabase เมื่อรันบน GitHub Actions

| Folder | โจทย์ |
| --- | --- |
| [common](common/) | โค้ดส่วนกลาง: อ่านข้อมูล, สร้าง feature, LightGBM/XGBoost, Neural network, Foundation Model, Hydrology, Backtest, Conformal, Ensemble |
| [task1_reservoir_monthly](task1_reservoir_monthly/) | โจทย์ 1: Inflow อ่างรายเดือน 1–3 และ 12 เดือน + Water Balance |
| [task1b_reservoir_daily](task1b_reservoir_daily/) | โจทย์ 1B: Inflow อ่างรายวัน 1–7 วัน (bridge) |
| [task2_river_hourly](task2_river_hourly/) | โจทย์ 2: ระดับน้ำลำน้ำ 1–48 ชั่วโมง |

แต่ละโมเดลอยู่คนละ Folder มี `README.md` และ `model.py` รายชื่อโมเดลและหลักการอยู่ใน `pilot/forecasting/MODELS.md`

## วิธีเรียกใช้โมเดล

ทุกโมเดลใช้แบบเดียวกัน

```python
from common.data import monthly_frame
from common.registry import load

df = monthly_frame("THREE")                      # แถวสุดท้าย = เดือนที่จะพยากรณ์ (y ว่าง)
m = load("task1_reservoir_monthly/05_lightgbm_quantile").Model().fit(df)
fc = m.predict(df, 3)                            # ตาราง step, p10, p50, p90
```

- `predict(df, horizon, future)`: `future` คือค่าที่รู้ล่วงหน้าของแต่ละ step เช่นฝนพยากรณ์ (คอลัมน์ `rain`)
- Backtest ย้อนหลัง: `common/backtest.py` (`run`, `summary`)
- Foundation Model (Chronos-2, TimesFM, Moirai) เขียนโค้ดแล้วแต่ยังไม่ได้ทดสอบ ต้องติดตั้งแพ็กเกจและโหลด weights จาก HuggingFace

## ทดสอบ

```
pip install -r requirements.txt
python3 tests/test_basics.py    # สูตรตรงกับตัวอย่างใน README
python3 tests/test_models.py    # ทุกโมเดล fit/predict ได้ (2–4 นาที)
```
