# ตั้งค่าให้รันพยากรณ์จริงบน GitHub Actions

ทำครั้งเดียว ทุกค่าเป็นความลับ ใส่เองในหน้าตั้งค่า ห้าม commit ลง repo นี้ (Public)

## 1. Supabase: สร้างตาราง

เปิด Supabase › SQL Editor แล้ววางไฟล์ `supabase/migrations/20261027000100_forecast_runs.sql` จาก repo เว็บ (flood-intelligence) กด Run
ได้ตาราง `forecast_runs`, `forecast_run_steps`, `forecast_values`, `forecast_plans`

## 2. GitHub repo นี้: Settings › Secrets and variables › Actions › New repository secret

| ชื่อ | ค่า |
| --- | --- |
| `SUPABASE_URL` | ค่าเดียวกับใน repo เว็บ |
| `SUPABASE_BACKEND_KEY` | ค่าเดียวกับใน repo เว็บ (sb_secret_…) |
| `DATA_REPO_TOKEN` | Fine-grained token: github.com › Settings › Developer settings › Fine-grained tokens › Generate · Repository access: เฉพาะ `flood-intelligence` · Permissions: Contents = Read-only |

`DATA_REPO_TOKEN` ใช้อ่านข้อมูลจาก repo private และเป็นกุญแจเข้ารหัสข้อมูลที่ส่งต่อระหว่างงาน (artifact ของ repo public ใครก็โหลดได้ จึงเข้ารหัสไว้)
ถ้ายังไม่ใส่ cron จะข้ามไปเฉย ๆ ไม่ error

## 3. Cloudflare Pages (เว็บ): Settings › Variables and Secrets › Production › Add (Type: Secret)

| ชื่อ | ค่า |
| --- | --- |
| `GITHUB_DISPATCH_TOKEN` | Fine-grained token อีกอัน · Repository access: เฉพาะ `forecasting-for-app-water-management` · Permissions: Actions = Read and write |

ปุ่ม Run ใน Forecast HQ ใช้ token นี้สั่ง workflow และใช้ `ADMIN_PASSWORD`, `SUPABASE_URL`, `SUPABASE_BACKEND_KEY` ที่มีอยู่แล้ว

## รอบที่รัน

| รอบ | เวลา (ไทย) | ทำอะไร | ใช้เวลาโดยประมาณ |
| --- | --- | --- | --- |
| รายวัน | ทุกวัน 02:00 | Nowcast สิ้นเดือน + 1–3 เดือน (โจทย์ 1), Inflow 1–7 วันด้วยฝนพยากรณ์ ECMWF/GFS/ICON/JMA (โจทย์ 1B) ใช้คะแนนและแชมป์จากรอบรายเดือนล่าสุด | 10–20 นาที |
| รายเดือน | วันที่ 2 เวลา 03:00 | 12 เดือน + Backtest ทุกโมเดล แล้วเลือก Gate / น้ำหนัก / แชมป์ใหม่ | 1–3 ชั่วโมง (โมเดลรันขนานกัน) |
| กดเอง | ปุ่ม Run ใน Forecast HQ หรือ Actions › Forecast run › Run workflow | เลือกรายวันหรือรายเดือน | ตามชนิด |

**รอบแรกให้กดรายเดือน** เพราะรอบรายวันต้องใช้คะแนน Backtest จากรอบรายเดือน

## ข้อมูลและโมเดลอยู่ที่ไหน

- ข้อมูล: ดึงจาก repo private ทุกครั้งที่รัน แล้วอัปเดต RID รายวันและฝนสถานี ThaiWater ล่าสุด (รอบรายเดือนอัปเดตความชื้นดินและ ENSO/IOD ด้วย) ไม่เก็บใน repo นี้
- torch รุ่น CPU และแพ็กเกจ: cache ของ pip บน GitHub
- weights ของ Chronos-2, TimesFM, Moirai: โหลดจาก Hugging Face ครั้งแรก แล้วเก็บใน cache ของ GitHub Actions (ฟรี สูงสุด 10 GB ต่อ repo; cache ที่ไม่ได้ใช้ 7 วันจะถูกลบแล้วโหลดใหม่เอง)
- ผล: Supabase ตาราง `forecast_runs.summary` (ทุกอย่างที่ HQ แสดง) และ `forecast_values` (P10/P50/P90 ทุกโมเดล)

## License ของ weights

Chronos-2 และ TimesFM 2.5: Apache-2.0 · Moirai 2.0: **CC BY-NC 4.0 (ห้ามใช้เชิงพาณิชย์)** ถ้าระบบนี้ใช้ในงานขององค์กรที่เป็นธุรกิจ ให้ตัด `12_moirai` ออกจาก `pipeline/plan.py`
