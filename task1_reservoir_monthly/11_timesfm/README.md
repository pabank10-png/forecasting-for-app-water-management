# TimesFM 2.5 (Google)

- กลุ่ม: Deep Learning (Foundation Model)
- ทำอะไร: foundation model อีกค่าย ให้ quantile
- ข้อมูลที่ใช้: inflow + ฝน + ความชื้นดิน + ENSO (ONI) + IOD (DMI)
- Lag: inflow ย้อนหลังสูงสุด 1,024 เดือน | ฝน lag 1, ฝนรวม 3 เดือน lag 1, ความชื้นดิน lag 1, ONI lag 1, DMI lag 1 (regression)
- สถานะ: ใช้งานได้ ทดสอบกับข้อมูลจริงแล้ว

## หลักการ
Foundation model ของ Google ตัดอนุกรมเวลาเป็นท่อนๆ (patch) เช่นท่อนละ 32 ค่า แล้วให้ Transformer แบบ decoder-only ทำนายท่อนถัดไป เหมือนเดาคำถัดไปในประโยค ใช้แบบ zero-shot

## สูตร
patch = [y(t−31) … y(t)] → embedding → Transformer → patch อนาคต (h ค่า) + quantile heads

## ตัวอย่างการคำนวณ (ตัวเลขสมมติ)
inflow 64 เดือนล่าสุดถูกตัดเป็น 2 patch (patch ละ 32 ค่า) หลัง normalize (ลบ mean หาร std: mean 35, std 30)
โมเดลทาย ต.ค. ในสเกลปกติ = 2.3 → แปลงกลับ 35 + 2.3×30 = **104 MCM**

## ใช้ในโปรเจกต์นี้อย่างไร
`timesfm` ใช้เป็นอีกเสียงหนึ่งใน Ensemble ไม่ใช้ covariates ในรุ่นแรก
