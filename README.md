# forecasting-for-app-water-management

ส่วนคำนวณพยากรณ์ของ Water Intelligence (inflow อ่างเก็บน้ำ และระดับน้ำลำน้ำ) รันด้วย GitHub Actions แล้วส่งผลขึ้น Supabase ให้เว็บอ่าน

**repo นี้เก็บแค่ code** ข้อมูลอยู่ที่ `../pilot/forecasting/1-reservoir-inflow/data/csv` บนเครื่อง (ตั้ง `FORECAST_DATA_DIR` เพื่อชี้ที่อื่นได้) และจะอยู่ใน Supabase เมื่อรันบน GitHub Actions

| Folder | โจทย์ |
| --- | --- |
| [common](common/) | อ่านข้อมูล, ตัววัดผล (MAE, P10–P90 coverage) |
| [task1_reservoir_monthly](task1_reservoir_monthly/) | โจทย์ 1: Inflow อ่างรายเดือน 1–3 และ 12 เดือน + Water Balance |
| [task1b_reservoir_daily](task1b_reservoir_daily/) | โจทย์ 1B: Inflow อ่างรายวัน 1–7 วัน (bridge) |
| [task2_river_hourly](task2_river_hourly/) | โจทย์ 2: ระดับน้ำลำน้ำ 1–48 ชั่วโมง |

แต่ละโมเดลอยู่คนละ Folder มี `README.md` และ `model.py` รายชื่อโมเดลและหลักการอยู่ใน `pilot/forecasting/MODELS.md`

ทดสอบ: `python3 tests/test_basics.py`
