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
v1 = np.array([-20, -18, -85, -9, 3.6, 10.85])

# สร้างรายการเพื่อเก็บค่ามุม
vt = [0, 0, 0, 0, 0, 0]  # สร้างรายการสำหรับเก็บค่ามุม 4 ค่า
push_up_phase = None  # เก็บสถานะของการวิดพื้น (up หรือ down)
is_form_correct = False  # เก็บสถานะว่าอยู่ในท่าถูกต้องหรือไม่
push_up_count = 0  # เก็บจำนวนครั้งของการวิดพื้น

# สร้างกราฟ
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 11)  # กำหนดขอบเขตแกน x ตั้งแต่ 0 ถึง 11
ax.set_ylim(0, 100)  # กำหนดขอบเขตแกน y ตั้งแต่ 0.1 ถึง 1.1
ax.tick_params(axis='both', which='major', labelsize=14)
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
ax.set_xlabel('Push-up Count',fontsize = 20)
ax.set_ylabel('% Accuracy',fontsize = 20)
ax.set_title('Push-up Correctness Verification System Using image Processing',fontsize = 20)

# เริ่มต้นกราฟ
line, = ax.plot([], [], lw=5,color='blue', marker='x', markersize = 15, markeredgewidth=3 ,markerfacecolor='red', markeredgecolor='red')

# เก็บข้อมูลสำหรับแสดงผลในกราฟ
push_up_counts = []
cosine_similarities = []

push_up_count = 0

# กำหนดค่า v_target ตามจำนวนครั้งของการวิดพื้น
max_count = 10  # กำหนดจำนวนครั้งสูงสุดที่ต้องการแสดงผล
display_count = 0  # เริ่มต้นจำนวนครั้งที่แสดงผลเป็น 0

# เพิ่มโค้ดจับเวลา
start_time = None
timer_running = False
elapsed_time = 0

hip_up_time = None
hip_up_after_005s = None

ankle_up_time = None
ankle_up_time_up_after_005s = None

elbow_up_time = None
elbow_up_after_005s = None

hip_down_time = None
hip_down_after_005s = None

ankle_down_time = None
ankle_down_after_005s = None

elbow_down_time = None
elbow_down_after_005s = None

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
    
    # ตรวจสอบปุ่มกด
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
       if not timer_running:
           start_time = time.time()
           timer_running = True
       else:
           timer_running = False
    elif key == ord('r'):
       elapsed_time = 0
       start_time = None
       timer_running = False

    # จับเวลาหากกำลังจับเวลาอยู่
    if timer_running:
       elapsed_time = int(time.time() - start_time)
       if elapsed_time >= 30:  # หากเวลาครบ 60 วินาที
           timer_running = False
           # แสดงกราฟ
           plt.show()
           break

    # แสดงเวลา
    cv2.putText(img, f'Time: {elapsed_time} seconds', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if results.pose_landmarks:
        # ตรวจสอบท่าวิดพื้น
        right_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img.shape[1])
        right_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img.shape[0])
        right_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * img.shape[1])
        right_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * img.shape[0])
        right_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * img.shape[1])
        right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img.shape[0])
        right_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img.shape[1])
        right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img.shape[0])
        right_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * img.shape[1])
        right_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * img.shape[0])
        #สะโพก
        cv2.line(img, (right_hip_x, right_hip_y), (right_shoulder_x , right_shoulder_y), (0, 150, 255), 3) #สะโพก
        cv2.line(img, (right_hip_x, right_hip_y), (right_hip_x + 250 , right_hip_y), (0, 255, 0), 3)
        cv2.line(img, (right_hip_x, right_hip_y), (right_hip_x , right_hip_y -200), (0, 0, 255), 3)
        #ข้อเท้า
        cv2.line(img, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), (200, 100, 255), 3) #เข่า
        cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x + 350, right_ankle_y), (0, 255, 0), 3)
        cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x, right_ankle_y -200), (0, 0, 255), 3)
        #ข้อศอก
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 3) #ข้อศอก
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x + 350, right_elbow_y), (0, 255, 0), 3)
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x, right_elbow_y -200), (0, 0, 255), 3)
        #คำนวณมุมสะโพก
        hip_x = right_shoulder_x - right_hip_x  # ปรับให้สะโพกเป็นจุด (0,0)
        hip_y = right_shoulder_y - right_hip_y
        angle_hip = math.degrees(math.atan2(hip_y, hip_x))
        angle_text = f"Angle Hip: {angle_hip:.2f} degrees"
        #แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        #คำนวณมุมข้อเท้า
        ankle_x = right_knee_x - right_ankle_x # ปรับให้ข้อเท้าเป็นจุด (0,0)
        ankle_y = right_knee_y - right_ankle_y
        angle_ankle = math.degrees(math.atan2(ankle_y, ankle_x))
        angle_text = f"Angle Ankle: {angle_ankle:.2f} degrees"
        # แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 250, 150), 2, cv2.LINE_AA) 
        # คำนวณมุมข้อศอก
        elbow_x = right_shoulder_x - right_elbow_x  # ปรับให้ข้อศอกเป็นจุด (0,0)
        elbow_y = right_shoulder_y - right_elbow_y
        angle_elbow = math.degrees(math.atan2(elbow_y, elbow_x))
        angle_text = f"Angle Elbow: {angle_elbow:.2f} degrees"
        # แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA) # แสดงผลมุมบนวิดีโอ
        # แสดงผลจำนวนครั้งของการวิดพื้น
        cv2.putText(img, f'Push-ups: {push_up_count}', (430, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 2, cv2.LINE_AA) # แสดงผลจำนวนครั้งของการวิดพื้น
        
        # ตรวจสอบท่าถูกต้อง
        if angle_hip < -15:
            if hip_up_time is None:
                hip_up_time = time.time()
            elif time.time() - hip_up_time >= 0.05:
                hip_up_after_005s = angle_hip
                vt[0] = hip_up_after_005s 
                hip_up_time = None
        if angle_ankle < -15:
            if ankle_up_time is None:
                ankle_up_time = time.time()
            elif time.time() - ankle_up_time >= 0.05:
                ankle_up_after_005s = angle_ankle
                vt[1] = ankle_up_after_005s
                ankle_up_time = None
        if angle_elbow < -80:
            if elbow_up_time is None:
                elbow_up_time = time.time()
            elif time.time() - elbow_up_time >= 0.05:
                elbow_up_after_005s = angle_elbow
                vt[2] = elbow_up_after_005s
                elbow_up_time = None
            is_form_correct = True
        else:
            is_form_correct = False

        if angle_hip > -15:
            if hip_down_time is None:
                hip_down_time = time.time()
            elif time.time() - hip_down_time >= 0.05:
                hip_down_after_005s = angle_hip
                vt[3] = hip_down_after_005s
                hip_down_time = None
        if angle_ankle > 0:
            if ankle_down_time is None:
                ankle_down_time = time.time()
            elif time.time() - ankle_down_time >= 0.05:
                ankle_down_after_005s = angle_ankle
                vt[4] = ankle_down_after_005s
                ankle_down_time = None
            elif angle_ankle > 7:
                vt[4] = 0
        if angle_elbow > 3:
            if elbow_down_time is None:
                elbow_down_time = time.time()
            elif time.time() - elbow_down_time >= 0.05:
                elbow_down_after_005s = angle_elbow
                vt[5] = elbow_down_after_005s
                elbow_down_time = None

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
        
                
                vc = v1
                    
                # คำนวณ norm ของเวกเตอร์ vc และ vt
                norm_vc = np.linalg.norm(v1)
                norm_vt = np.linalg.norm(vt)

                # แปลงเวกเตอร์เป็น normalized vector
                normalized_vc = v1 / norm_vc
                normalized_vt = vt / norm_vt

                # คำนวณ 1 - norm ของผลต่างของเวกเตอร์ที่ normalized
                accuracy_percentage = (1 - np.linalg.norm(normalized_vc - normalized_vt)) * 100

                print(f"Push-up Count: {push_up_count}")
                print(f"vc for Count {push_up_count}: {vc}")
                print(f"vt for Count {push_up_count}: {vt}")
                print(f"% Accuracy for Count {push_up_count}: {accuracy_percentage}")

                # เก็บข้อมูลสำหรับแสดงผลในกราฟ
                push_up_counts.append(push_up_count)
                cosine_similarities.append(accuracy_percentage)

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
                        
                # รีเซ็ตค่าในรายการ vt
                vt = [0, 0, 0, 0, 0, 0]


    # แสดงผลภาพ
    cv2.imshow('Pose Detection', img)

    # กด 'q' เพื่อออกจาก loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ปิดการใช้งานกล้องและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
plt.show()  # แสดงกราฟ