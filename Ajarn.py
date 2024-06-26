import cv2
import mediapipe as mp
import numpy as np
import math
import time
from sklearn.metrics.pairwise import cosine_similarity

# สร้างวัตถุตรวจจับ Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# เปิดกล้อง
cap = cv2.VideoCapture('Trainer/Push.mp4')

v1 = [18.0, -83.0, 0]
v2 = [3.0, -10.0, 6.0]
v3 = [18.0, -83.0, 1.2]

# ตัวแปรสำหรับเก็บค่า cosine similarity
cosine_similarities_v1_v4 = []
cosine_similarities_v2_v5 = []

# ตั้งค่าตัวแปรสำหรับนับเวลา
start_time_down = None  # เก็บเวลาเริ่มต้นของการลงวิดพื้น
start_time_up = None    # เก็บเวลาเริ่มต้นของการขึ้นวิดพื้น
total_time_down = 0.0   # เก็บเวลาทั้งหมดขณะลงวิดพื้น
total_time_up = 0.0     # เก็บเวลาทั้งหมดขณะขึ้นวิดพื้น
num_push_ups = 0        # เก็บจำนวนครั้งของการวิดพื้นขึ้น
fix_form_time = 0.0     # เก็บเวลาที่ fix form

while True:
    # อ่านภาพจากกล้อง
    success, img = cap.read()
    img = cv2.resize(img, (650, 480))
    
    if not success:
        break

    # แปลงภาพเป็นสี BGR เป็น RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ตรวจจับโพสและแสดงผล
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # สร้างรายการเพื่อเก็บค่ามุมและเวลา
        v4 = np.zeros(len(v1))  # สร้างอาร์เรย์ขนาดเท่ากับจำนวน landmark ในแต่ละเฟรม และกำหนดค่าเริ่มต้นเป็น 0
        v5 = np.zeros(len(v2))
        v6 = np.zeros(len(v3))
        # ตรวจสอบหาว่ามีการลงหรือขึ้นของการวิดพื้น
        if start_time_down is None:  # ถ้ายังไม่เริ่มนับเวลาขณะลงวิดพื้น
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y:
                start_time_down = time.time()  # เริ่มนับเวลาเมื่อวิดพื้นเริ่มลง
        else:  # ถ้าเริ่มนับเวลาแล้วขณะลงวิดพื้น
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y:
                total_time_down += time.time() - start_time_down  # เพิ่มเวลาที่ผ่านไปเมื่อวิดพื้นขึ้นขึ้น
                start_time_down = None  # เริ่มนับเวลาใหม่เมื่อวิดพื้นลงอีกครั้ง
                v5[2] = total_time_down
        
        if start_time_up is None:  # ถ้ายังไม่เริ่มนับเวลาขณะขึ้นวิดพื้น
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y:
                start_time_up = time.time()  # เริ่มนับเวลาเมื่อวิดพื้นเริ่มขึ้น
        else:  # ถ้าเริ่มนับเวลาแล้วขณะขึ้นวิดพื้น
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y:
                total_time_up += time.time() - start_time_up  # เพิ่มเวลาที่ผ่านไปเมื่อวิดพื้นลงลง
                start_time_up = None  # เริ่มนับเวลาใหม่เมื่อวิดพื้นขึ้นอีกครั้ง
                num_push_ups += 1  # เพิ่มจำนวนครั้งของการวิดพื้นขึ้น
        total_time = total_time_down + total_time_up

        # คำนวณและเพิ่มค่ามุมและเวลาลงในรายการ v2
        # วาดเส้นตรงในแนวแกน x จากข้อเท้าไปยังจุดที่ y เท่ากับ 0
        right_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img.shape[1])
        right_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img.shape[0])
        right_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img.shape[1])
        right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img.shape[0])
        right_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * img.shape[1])
        right_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * img.shape[0])
        angle_body = math.degrees(math.atan2(right_shoulder_y - right_ankle_y, right_shoulder_x - right_ankle_x)) # คำนวณมุมระหว่างสองเส้น
        angle_body = abs(angle_body)
        if angle_body > 18:
            print('Fix form',angle_body)
            v4[0] = angle_body  # Body_angle_fixform
        
        if angle_body < 3:
            print('Down',angle_body)
            v5[0] = angle_body



        angle_elbow = math.degrees(math.atan2(right_shoulder_y - right_elbow_y, right_shoulder_x - right_elbow_x)) # คำนวณมุมระหว่างสองเส้น
        angle_text = f"Angle Elbow: {angle_elbow:.2f} degrees"
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 3)
        cv2.putText(img, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # แสดงผลมุมบนวิดีโอ
        if angle_elbow < -80:
            print('Fix form',angle_elbow)
            v4[1] = angle_elbow  # Body_angle_fixform

        if angle_elbow < -10:
            print('Fix form',angle_elbow)
            v5[1] = angle_elbow  

        # เพิ่มโค้ด v3 ซึ่งเก็บเวลาที่ fix form ในตำแหน่งที่ 3 ของ v2
        v4[2] = fix_form_time

        
        # คำนวณ cosine similarity ระหว่าง v1 และ v2
        v1_np = np.array(v1).reshape(1, -1)  # แปลง v1 เป็น numpy array และ reshape เพื่อให้มีมิติเดียว
        v4_np = np.array(v4).reshape(1, -1)  # แปลง v2 เป็น numpy array และ reshape เพื่อให้มีมิติเดียว
        similarity_v1_v4 = cosine_similarity(v1_np, v4_np)  # คำนวณ cosine similarity
        # แสดงผลรายการ v1 และ v2 พร้อมกับค่า cosine similarity
        print("v1:", v1)
        print("v4:", v4)
        print("Cosine Similarity-v1-v4:", similarity_v1_v4)

        v2_np = np.array(v2).reshape(1, -1)  # แปลง v2 เป็น numpy array และ reshape เพื่อให้มีมิติเดียว
        v5_np = np.array(v5).reshape(1, -1)  # แปลง v5 เป็น numpy array และ reshape เพื่อให้มีมิติเดียว
        similarity_v2_v5 = cosine_similarity(v2_np, v5_np)  # คำนวณ cosine similarity
        print("v2:", v2)
        print("v5:", v5)
        print("Cosine Similarity-v2-v5:", similarity_v2_v5)

        
        # แสดงผลรายการ v1 และ v2 พร้อมกับค่า cosine similarity

        # ... (โค้ดเพิ่มเติมสำหรับประมวลผลและแสดงผล)

    # แสดงผลภาพ
    cv2.imshow('Pose Detection', img)

    # กด 'q' เพื่อออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการใช้งานกล้องและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
