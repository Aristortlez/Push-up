import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# สร้างวัตถุตรวจจับ Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# กำหนดค่า v1 ถึง v10
v1 = np.array([8, -93.91, 0, 41.04])

# สร้างรายการเพื่อเก็บค่ามุม
v_current = [0, 0, 0, 0]
push_up_phase = None
is_form_correct = False
push_up_count = 0

# สร้างรายการเพื่อเก็บข้อมูลสำหรับกราฟ
push_up_counts = []
cosine_similarities = []

# สร้างหน้าต่าง GUI
root = tk.Tk()
root.title("Push-up Form Evaluation")

# สร้างช่องสำหรับกรอกจำนวนครั้ง
count_frame = tk.Frame(root)
count_frame.pack(side=tk.TOP, pady=10)
count_label = tk.Label(count_frame, text="Enter number of push-ups:")
count_label.pack(side=tk.LEFT)
count_entry = tk.Entry(count_frame)
count_entry.pack(side=tk.LEFT)

# สร้างช่องสำหรับกรอกเวลา
time_frame = tk.Frame(root)
time_frame.pack(side=tk.TOP, pady=10)
time_label = tk.Label(time_frame, text="Enter duration (seconds):")
time_label.pack(side=tk.LEFT)
time_entry = tk.Entry(time_frame)
time_entry.pack(side=tk.LEFT)

# ฟังก์ชันสำหรับตรวจจับท่าทาง
def start_detection():
    global push_up_count, push_up_phase, is_form_correct, v_current, push_up_counts, cosine_similarities

    # ตรวจสอบเงื่อนไขจำนวนครั้งและเวลา
    max_count = int(count_entry.get() or 0)
    duration = int(time_entry.get() or 0)

    if max_count == 0 and duration == 0:
        print("Please enter either number of push-ups or duration")
        return

    start_time = time.time()
    end_time = start_time + duration if duration > 0 else float('inf')

    while time.time() < end_time or duration == 0:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (650, 480))
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
    
        if results.pose_landmarks:
            # ตรวจสอบท่าวิดพื้น
            right_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img.shape[1])
            right_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img.shape[0])
            right_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * img.shape[1])
            right_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * img.shape[0])
            right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img.shape[0])
            right_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img.shape[1])
            right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img.shape[0])
            right_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * img.shape[1])
            right_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * img.shape[0])

            # เข่า
            cv2.line(img, (right_ankle_x, right_ankle_y), (right_knee_x - 150, right_knee_y), (200, 100, 255), 3)
            cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x + 350, right_ankle_y), (0, 255, 0), 3)
            cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x, right_ankle_y - 200), (0, 0, 255), 3)
            # ข้อศอก
            cv2.line(img, (right_elbow_x, right_elbow_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 3)
            cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x + 350, right_elbow_y), (0, 255, 0), 3)
            cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x, right_elbow_y - 200), (0, 0, 255), 3)
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
                    v_current = [0, 0, 0, 0]
            elif push_up_phase == "down":
                if not is_form_correct:
                    push_up_phase = "up"
                    v_current = [0, 0, 0, 0]
            elif push_up_phase == "up":
                if is_form_correct:
                    push_up_phase = None
                    push_up_count += 1
                    cv2.putText(img, f'Push-ups: {push_up_count}', (430, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # แสดงผลจำนวนครั้งของการวิดพื้น
                
                    # เก็บค่า v_current เฉพาะตอนวิดพื้นลงสุดและขึ้นสุด
                    if angle_knee > 8:
                        v_current[0] = 1
                    elif angle_knee < 0:
                        v_current[1] = 1
                    elif angle_elbow < -90:
                        v_current[2] = 1
                    elif angle_elbow > 40:
                        v_current[3] = 1

                    # คำนวณ norm ของเวกเตอร์ v_target และ v_current
                    norm_v1 = np.linalg.norm(v1)
                    norm_v_current = np.linalg.norm(v_current)

                    # แปลงเวกเตอร์เป็น normalized vector
                    normalized_v1 = v1 / norm_v1
                    normalized_v_current = v_current / norm_v_current

                    # คำนวณ 1 - norm ของผลต่างของเวกเตอร์ที่ normalized
                    accuracy_percentage = (1 - np.linalg.norm(normalized_v1 - normalized_v_current)) * 100

                    print(f"Push-up Count: {push_up_count}")
                    print(f"v_target for Count {push_up_count}: {v1}")
                    print(f"v_current for Count {push_up_count}: {v_current}")
                    print(f"Cosine Similarity for Count {push_up_count}: {accuracy_percentage}")
                
                    push_up_counts.append(push_up_count)
                    cosine_similarities.append(accuracy_percentage)
                    line.set_data(push_up_counts, cosine_similarities)
                    ax.set_xlim(0, push_up_count + 1)
                    fig.canvas.draw()

                    v_current = [0, 0, 0, 0]

            cv2.imshow("Push-up Form Evaluation", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if max_count > 0 and push_up_count >= max_count:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# สร้างปุ่มเริ่มต้น
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(side=tk.TOP, pady=10)

# สร้างพื้นที่สำหรับกราฟ
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.set_xlabel('Push-up Count')
ax.set_ylabel('Cosine Similarity (%)')
ax.set_title('Cosine Similarity for Push-up Forms')
line, = ax.plot([], [], 'r-')

# สร้าง Canvas สำหรับกราฟ
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.mainloop()
