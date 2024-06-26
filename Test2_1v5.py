import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# สร้างวัตถุตรวจจับ Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# เปิดกล้อง
cap = cv2.VideoCapture('Trainer/Push2_1.mp4')

# กำหนดค่า v1 ถึง v10
v1 = np.array([8, -93.91, 0, 41.04])

# สร้างรายการเพื่อเก็บค่ามุม
v_current = [0, 0, 0, 0]  # สร้างรายการสำหรับเก็บค่ามุม 4 ค่า
push_up_phase = None  # เก็บสถานะของการวิดพื้น (up หรือ down)
is_form_correct = False  # เก็บสถานะว่าอยู่ในท่าถูกต้องหรือไม่
push_up_count = 0  # เก็บจำนวนครั้งของการวิดพื้น

# สร้างกราฟ
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 11)  # กำหนดขอบเขตแกน x ตั้งแต่ 0 ถึง 11
ax.set_ylim(0.1, 1.1)  # กำหนดขอบเขตแกน y ตั้งแต่ 0.1 ถึง 1.1
ax.set_xlabel('Push-up Count')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Push-up Form Evaluation')

# เริ่มต้นกราฟ
line, = ax.plot([], [], lw=2,color='blue', marker='o',markerfacecolor='red', markeredgecolor='red')

# เก็บข้อมูลสำหรับแสดงผลในกราฟ
push_up_counts = []
cosine_similarities = []

# กำหนดค่า v_target ตามจำนวนครั้งของการวิดพื้น
max_count = 10  # กำหนดจำนวนครั้งสูงสุดที่ต้องการแสดงผล
display_count = 0  # เริ่มต้นจำนวนครั้งที่แสดงผลเป็น 0

while True:
    # อ่านภาพจากกล้อง
    success, img = cap.read()
    
    if not success:
        break

    # ตรวจสอบขนาดของภาพก่อนรีไซซ์
    if img.shape[0] > 0 and img.shape[1] > 0:
        img = cv2.resize(img, (650, 480))
    else:
        continue  # ข้ามการประมวลผลสำหรับเฟรมนี้

    # แปลงภาพเป็นสี BGR เป็น RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ตรวจจับโพสและแสดงผล
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # ตรวจสอบท่าวิดพื้น
        right_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img.shape[1])
        right_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img.shape[0])
        right_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * img.shape[1])
        right_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * img.shape[0])
        right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img.shape[0])
        right_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img.shape[1])
        right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img.shape[0])
        right_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * img.shape[1])
        right_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * img.shape[0])
        #เข่า
        cv2.line(img, (right_ankle_x, right_ankle_y), (right_knee_x -150, right_knee_y), (200, 100, 255), 3)
        cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x + 350, right_ankle_y), (0, 255, 0), 3)
        cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x, right_ankle_y + -200), (0, 0, 255), 3)
        # ข้อศอก
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 3)
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x + 350, right_elbow_y), (0, 255, 0), 3)
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x, right_elbow_y + -200), (0, 0, 255), 3)
        # คำนวณมุมลำตัว
        angle_knee = math.degrees(math.atan2(right_knee_y - right_ankle_y, right_knee_x - right_ankle_x))
        angle_knee = abs(angle_knee)
        angle_text = f"Angle Body: {angle_knee:.2f} degrees"
        # แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
        # คำนวณมุมข้อศอก
        angle_elbow = math.degrees(math.atan2(right_shoulder_y - right_elbow_y, right_shoulder_x - right_elbow_x))
        angle_text = f"Angle Elbow: {angle_elbow:.2f} degrees"
        # แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # แสดงผลมุมบนวิดีโอ
        
        # ตรวจสอบท่าถูกต้อง
        if angle_knee > 8:
            v_current[0] = angle_knee  # Body_angle_fixform
            is_form_correct = True
        else:
            is_form_correct = False
            
        if angle_knee < 0:
            v_current[2] = angle_knee
            
        if angle_elbow < -90:
            v_current[1] = angle_elbow  # Body_angle_fixform
            is_form_correct = True
        else:
            is_form_correct = False
            
        if angle_elbow > 40:
            v_current[3] = angle_elbow
        
        # ตรวจสอบสถานะของการวิดพื้น
        if push_up_phase is None:
            if is_form_correct:
                push_up_phase = "down"
        elif push_up_phase == "down":
            if not is_form_correct:
                push_up_phase = "up"
        elif push_up_phase == "up":
            if is_form_correct:
                push_up_phase = None
                push_up_count += 1
                cv2.putText(img, f'Push-ups: {push_up_count}', (430, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # แสดงผลจำนวนครั้งของการวิดพื้น
                
                # กำหนดค่า v_target ตามจำนวนครั้งของการวิดพื้น
                if push_up_count == 1:
                    v_target = v1
                elif push_up_count == 2:
                    v_target = v1
                elif push_up_count == 3:
                    v_target = v1
                elif push_up_count == 4:
                    v_target = v1
                elif push_up_count == 5:
                    v_target = v1
                elif push_up_count == 6:
                    v_target = v1
                elif push_up_count == 7:
                    v_target = v1
                elif push_up_count == 8:
                    v_target = v1
                elif push_up_count == 9:
                    v_target = v1
                elif push_up_count == 10:
                    v_target = v1
                
                # คำนวณ norm ของเวกเตอร์ v_target และ v_current
                norm_v1 = np.linalg.norm(v1)
                norm_v_current = np.linalg.norm(v_current)

                # แปลงเวกเตอร์เป็น normalized vector
                normalized_v1 = v1 / norm_v1
                normalized_v_current = v_current / norm_v_current

                # คำนวณ 1 - norm ของผลต่างของเวกเตอร์ที่ normalized
                result = 1 - np.linalg.norm(normalized_v1 - normalized_v_current)

                print(f"Push-up Count: {push_up_count}")
                print(f"v_target for Count {push_up_count}: {v_target}")
                print(f"v_current for Count {push_up_count}: {v_current}")
                print(f"Cosine Similarity for Count {push_up_count}: {result}")

                # เก็บข้อมูลสำหรับแสดงผลในกราฟ
                push_up_counts.append(push_up_count)
                cosine_similarities.append(result)

                # แสดงผลกราฟเส้น
                line.set_data(push_up_counts, cosine_similarities)
                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.draw()
                fig.canvas.flush_events()

                display_count += 1  # เพิ่มจำนวนครั้งที่แสดงผล

                # ตรวจสอบว่าแสดงผลครบจำนวนครั้งสูงสุดหรือไม่
                if display_count >= max_count:
                    break  # ออกจากวงวน
                
                # รีเซ็ตค่าในรายการ v_current
                v_current = [0, 0, 0, 0]


    # แสดงผลภาพ
    cv2.imshow('Pose Detection', img)

    # กด 'q' เพื่อออกจาก loop
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# ปิดการใช้งานกล้องและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
plt.show()  # แสดงกราฟ