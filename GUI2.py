import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# สร้างวัตถุตรวจจับ Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# กำหนดค่า v1 ถึง v10
v1 = np.array([-21.05, -22.66, -87.91, -6.23, 7.58, 20.21])

# สร้างรายการเพื่อเก็บค่ามุม
v_current = [0, 0, 0, 0, 0, 0]
push_up_phase = None
is_form_correct = False
push_up_count = 0

# สร้างกราฟ
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 15)
ax.set_ylim(0, 100)
ax.set_xlabel('Push-up Count')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Push-up Form Evaluation')
line, = ax.plot([], [], lw=2, color='blue', marker='o', markerfacecolor='red', markeredgecolor='red')

# เก็บข้อมูลสำหรับแสดงผลในกราฟ
push_up_counts = []
cosine_similarities = []

# สร้างหน้าต่าง GUI
root = tk.Tk()
root.title("Push-up Counter")

# สร้างพื้นที่แสดงผลกล้องและกราฟ
video_frame = tk.Frame(root)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

canvas = tk.Canvas(video_frame, width=650, height=480)
canvas.pack()

graph_frame = tk.Frame(root)
graph_frame.pack(side=tk.LEFT, padx=10, pady=10)

graph_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
graph_canvas.draw()
graph_canvas.get_tk_widget().pack()

# สร้างพื้นที่สำหรับกรอกข้อมูล
input_frame = tk.Frame(root)
input_frame.pack(side=tk.TOP, padx=10, pady=10)

label_count = tk.Label(input_frame, text="Number of Push-ups:")
label_count.pack(side=tk.LEFT)

entry_count = tk.Entry(input_frame)
entry_count.pack(side=tk.LEFT)

label_time = tk.Label(input_frame, text="Time (seconds):")
label_time.pack(side=tk.LEFT)

entry_time = tk.Entry(input_frame)
entry_time.pack(side=tk.LEFT)

start_button = tk.Button(input_frame, text="Start")
start_button.pack(side=tk.LEFT)

reset_button = tk.Button(input_frame, text="Reset")
reset_button.pack(side=tk.LEFT)

# กำหนดฟังก์ชันสำหรับปุ่ม Start
def start_countdown():
    target_count = int(entry_count.get() or 0)
    target_time = int(entry_time.get() or 0)

    if target_count > 0 or target_time > 0:
        countdown_label = tk.Label(root, text="Starting in 5 seconds...")
        countdown_label.pack()

        root.after(5000, countdown_label.destroy)
        root.after(5000, start_detection, target_count, target_time)

start_button.config(command=start_countdown)

# กำหนดฟังก์ชันสำหรับปุ่ม Reset
def reset_values():
    global push_up_count, push_up_phase, is_form_correct, v_current, push_up_counts, cosine_similarities

    push_up_count = 0
    push_up_phase = None
    is_form_correct = False
    v_current = [0, 0, 0, 0, 0, 0]
    push_up_counts = []
    cosine_similarities = []

    line.set_data([], [])
    graph_canvas.draw()

reset_button.config(command=reset_values)

# กำหนดฟังก์ชันสำหรับการตรวจจับท่าวิดพื้น
def start_detection(target_count, target_time):
    global push_up_count, push_up_phase, is_form_correct, v_current, push_up_counts, cosine_similarities

    cap = cv2.VideoCapture(0)  # เปิดกล้อง

    start_time = time.time()
    elapsed_time = 0

    while True:
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
            #เข่า
            cv2.line(img, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), (200, 100, 255), 3) #เข่า
            cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x + 350, right_ankle_y), (0, 255, 0), 3)
            cv2.line(img, (right_ankle_x, right_ankle_y), (right_ankle_x, right_ankle_y -200), (0, 0, 255), 3)
            # ข้อศอก
            cv2.line(img, (right_elbow_x, right_elbow_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 3) #ข้อศอก
            cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x + 350, right_elbow_y), (0, 255, 0), 3)
            cv2.line(img, (right_elbow_x, right_elbow_y), (right_elbow_x, right_elbow_y -200), (0, 0, 255), 3)
            # คำนวณมุมสะโพก
            angle_hip = math.degrees(math.atan2(right_shoulder_y - right_hip_y, right_shoulder_x - right_hip_x))
            angle_text = f"Angle Hip: {angle_hip:.2f} degrees"
            # แสดงผลมุมบนวิดีโอ
            cv2.putText(img, angle_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            # คำนวณมุมเข่า
            angle_knee = math.degrees(math.atan2(right_knee_y - right_ankle_y, right_knee_x - right_ankle_x))
            angle_text = f"Angle Knee: {angle_knee:.2f} degrees"
            # แสดงผลมุมบนวิดีโอ
            cv2.putText(img, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 250, 150), 2, cv2.LINE_AA) 
            # คำนวณมุมข้อศอก
            angle_elbow = math.degrees(math.atan2(right_shoulder_y - right_elbow_y, right_shoulder_x - right_elbow_x))
            angle_text = f"Angle Elbow: {angle_elbow:.2f} degrees"
            # แสดงผลมุมบนวิดีโอ
            cv2.putText(img, angle_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA) # แสดงผลมุมบนวิดีโอ
            # แสดงผลจำนวนครั้งของการวิดพื้น
            cv2.putText(img, f'Push-ups: {push_up_count}', (430, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 2, cv2.LINE_AA) # แสดงผลจำนวนครั้งของการวิดพื้น
            
            # ตรวจสอบท่าถูกต้อง
            if angle_hip < -21.05 :
                v_current[0] = angle_hip  # Body_angle_fixform
                is_form_correct = True
            else:
                is_form_correct = False

            if angle_hip > -6.23:
                v_current[3] = angle_hip

            if angle_knee < -22.66:
                v_current[1] = angle_knee  # Body_angle_fixform
                is_form_correct = True
            else:
                is_form_correct = False

            if angle_knee > 7.58:
                v_current[4] = angle_knee

            if angle_elbow < -85:
                v_current[2] = angle_elbow  # Body_angle_fixform
                is_form_correct = True
            else:
                is_form_correct = False

            if angle_elbow > 15:
                v_current[5] = angle_elbow
            # (โค้ดสำหรับตรวจสอบท่าวิดพื้นจากไฟล์ต้นฉบับ)

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
                    # (โค้ดสำหรับกำหนดค่า v_target จากไฟล์ต้นฉบับ)

                    # คำนวณ norm ของเวกเตอร์ v_target และ v_current
                    norm_v1 = np.linalg.norm(v1)
                    norm_v_current = np.linalg.norm(v_current)

                    # แปลงเวกเตอร์เป็น normalized vector
                    normalized_v1 = v1 / norm_v1
                    normalized_v_current = v_current / norm_v_current

                    # คำนวณ 1 - norm ของผลต่างของเวกเตอร์ที่ normalized
                    accuracy_percentage = (1 - np.linalg.norm(normalized_v1 - normalized_v_current)) * 100

                    print(f"Push-up Count: {push_up_count}")
                    print(f"v_target for Count {push_up_count}: {v_target}")
                    print(f"v_current for Count {push_up_count}: {v_current}")
                    print(f"Cosine Similarity for Count {push_up_count}: {accuracy_percentage}")

                    # เก็บข้อมูลสำหรับแสดงผลในกราฟ
                    push_up_counts.append(push_up_count)
                    cosine_similarities.append(accuracy_percentage)

                    # แสดงผลกราฟเส้น
                    line.set_data(push_up_counts, cosine_similarities)
                    ax.relim()
                    ax.autoscale_view(True, True, True)
                    graph_canvas.draw()

                    # ตรวจสอบว่าครบจำนวนครั้งหรือเวลาที่กำหนดหรือไม่
                    elapsed_time = int(time.time() - start_time)
                    if push_up_count >= target_count or elapsed_time >= target_time:
                        break

                    # รีเซ็ตค่าในรายการ v_current
                    v_current = [0, 0, 0, 0, 0, 0]

                    # แสดงผลภาพจากกล้อง
                    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    root.update()

    cap.release()

def main():
    target_count = int(entry_count.get() or 0)
    target_time = int(entry_time.get() or 0)

    start_detection(target_count, target_time)

    # วงวนหลัก Tkinter เพื่อรักษาสถานะของหน้าต่าง GUI
    root.mainloop()

start_button.config(command=main)