# XGBoost quantile

- กลุ่ม: Machine Learning (Gradient-Boosted Trees)
- ทำอะไร: ตัวเทียบของ LightGBM
- ข้อมูลที่ใช้: ชีต LAG
- สถานะ: ยังไม่เริ่ม

## หลักการ
เหมือน LightGBM คือ Gradient boosting ของ Decision tree แต่ XGBoost สร้างต้นไม้แบบทีละชั้นเท่ากันทั้งต้น (level-wise) และมี Regularization ลงโทษใบไม้ที่มีค่ามาก ทำให้ผลต่างจาก LightGBM เล็กน้อย จึงเก็บไว้ช่วยใน Ensemble

## สูตร
ค่าที่ใบไม้ (leaf weight) = −ΣG / (ΣH + λ)
G = gradient ของ loss, H = hessian, λ = ค่าลงโทษ

## ตัวอย่างการคำนวณ (ตัวเลขสมมติ)
ใบหนึ่งมีข้อมูล 3 เดือน error (y − ŷ) = +30, +20, +10 ใช้ squared loss → G = −30, −20, −10 (ผลรวม −60), H = 1 ต่อแถว (ผลรวม 3)
- λ = 0: weight = 60 / 3 = 20
- λ = 3: weight = 60 / 6 = 10 (ถูกหดลงครึ่งหนึ่ง ป้องกัน Overfitting เมื่อข้อมูลน้อย)

## ใช้ในโปรเจกต์นี้อย่างไร
`xgboost` ตั้ง `objective="reg:quantileerror", quantile_alpha=[0.1,0.5,0.9]` ใช้ข้อมูลชุดเดียวกับ LightGBM
