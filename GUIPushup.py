import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# สร้างวัตถุตรวจจับ Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# กำหนดค่า v1 ถึง v10
v1 = np.array([-20, -18, -85, -9, 3.6, 10.85])

# สร้างรายการเพื่อเก็บค่ามุม
vt = [0, 0, 0, 0, 0, 0]  # สร้างรายการสำหรับเก็บค่ามุม 4 ค่า
push_up_phase = None  # เก็บสถานะของการวิดพื้น (up หรือ down)
is_form_correct = False  # เก็บสถานะว่าอยู่ในท่าถูกต้องหรือไม่
push_up_count = 0  # เก็บจำนวนครั้งของการวิดพื้น

# เก็บข้อมูลสำหรับแสดงผลในกราฟ
push_up_counts = []
percent_accuracy = []

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

# สร้าง GUI
root = tk.Tk()
root.title("Push-up Correctness Verification System")
root.geometry("1200x800")

# สร้างเฟรมสำหรับวิดีโอ
video_frame = ttk.Frame(root, width=650, height=480)
video_frame.grid(row=0, column=0, padx=10, pady=10)

# สร้างเฟรมสำหรับกราฟ
graph_frame = ttk.Frame(root, width=500, height=400)
graph_frame.grid(row=0, column=1, padx=10, pady=10)

# สร้างกราฟ
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('Push-up Count', fontsize=10)
ax.set_ylabel('% Accuracy', fontsize=10)
ax.set_title('Push-up Correctness Verification System', fontsize=12)

# เริ่มต้นกราฟ
line, = ax.plot([], [], lw=2, color='blue', marker='x', markersize=5, markeredgewidth=2, markerfacecolor='red', markeredgecolor='red')

# สร้าง FigureCanvasTkAgg
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# สร้างเฟรมสำหรับการควบคุม
control_frame = ttk.Frame(root)
control_frame.grid(row=1, column=0, columnspan=2, pady=10)

# สร้างช่องกรอกจำนวนครั้ง
count_label = ttk.Label(control_frame, text="Target Push-ups:")
count_label.grid(row=0, column=0, padx=5, pady=5)
count_entry = ttk.Entry(control_frame)
count_entry.grid(row=0, column=1, padx=5, pady=5)

# สร้างช่องกรอกเวลา
time_label = ttk.Label(control_frame, text="Target Time (seconds):")
time_label.grid(row=0, column=2, padx=5, pady=5)
time_entry = ttk.Entry(control_frame)
time_entry.grid(row=0, column=3, padx=5, pady=5)

# สร้างตัวแปรสำหรับเก็บค่าเป้าหมาย
target_count = tk.IntVar()
target_time = tk.DoubleVar()

# สร้างฟังก์ชันสำหรับเริ่มต้นการทำงาน
def start():
    global cap, push_up_count, push_up_counts, percent_accuracy, start_time
    global hip_up_time, hip_up_after_005s
    global ankle_up_time, ankle_up_time_up_after_005s
    global elbow_up_time, elbow_up_after_005s
    global hip_down_time, hip_down_after_005s
    global ankle_down_time, ankle_down_after_005s
    global elbow_down_time, elbow_down_after_005s
    global is_webcam_ready
    
    # รีเซ็ตค่าต่างๆ
    push_up_count = 0
    push_up_counts = []
    percent_accuracy = []
    start_time = time.time()
    is_webcam_ready = False

    # อ่านค่าเป้าหมาย
    target_count.set(int(count_entry.get()) if count_entry.get() else 0)
    target_time.set(float(time_entry.get()) if time_entry.get() else 0)

    # ปรับแกน x ของกราฟตามค่าเป้าหมาย
    ax.set_xlim(0, target_count.get() + 1)
    ax.figure.canvas.draw()
    
    # เปิดกล้อง
    cap = cv2.VideoCapture(0)
    
    update_frame()

# สร้างฟังก์ชันสำหรับรีเซ็ตการทำงาน
def reset():
    global cap, push_up_count, push_up_counts, percent_accuracy
    global hip_up_time, hip_up_after_005s
    global ankle_up_time, ankle_up_time_up_after_005s
    global elbow_up_time, elbow_up_after_005s
    global hip_down_time, hip_down_after_005s
    global ankle_down_time, ankle_down_after_005s
    global elbow_down_time, elbow_down_after_005s
    
    if cap is not None:
        cap.release()

    hip_up_time = ankle_up_time = elbow_up_time = None
    hip_down_time = ankle_down_time = elbow_down_time = None
    hip_up_after_005s = ankle_up_time_up_after_005s = elbow_up_after_005s = None
    hip_down_after_005s = ankle_down_after_005s = elbow_down_after_005s = None
    
    push_up_count = 0
    push_up_counts = []
    percent_accuracy = []
    
    # รีเซ็ตกราฟ
    line.set_data([], [])
    ax.set_xlim(0, 11)  # รีเซ็ตแกน x กลับเป็นค่าเริ่มต้น
    ax.relim()
    ax.autoscale_view(True, True, True)
    canvas.draw()
    
    # รีเซ็ตค่าในช่องกรอก
    count_entry.delete(0, tk.END)
    time_entry.delete(0, tk.END)
    
    # อัพเดทหน้าจอ
    video_label.config(image='')

# สร้างปุ่ม Start และ Reset
start_button = ttk.Button(control_frame, text="Start", command=start)
start_button.grid(row=1, column=1, padx=5, pady=5)

reset_button = ttk.Button(control_frame, text="Reset", command=reset)
reset_button.grid(row=1, column=2, padx=5, pady=5)

# สร้าง Label สำหรับแสดงวิดีโอ
video_label = ttk.Label(video_frame)
video_label.pack()

def update_frame():
    global push_up_count, push_up_phase, is_form_correct, vt
    global hip_up_time, hip_up_after_005s
    global ankle_up_time, ankle_up_time_up_after_005s
    global elbow_up_time, elbow_up_after_005s
    global hip_down_time, hip_down_after_005s
    global ankle_down_time, ankle_down_after_005s
    global elbow_down_time, elbow_down_after_005s
    global start_time, push_up_counts, percent_accuracy
    global is_webcam_ready

    success, img = cap.read()
    
    if not success:
        root.after(10, update_frame)
        return

    if img.shape[0] > 0 and img.shape[1] > 0:
        img = cv2.resize(img, (650, 480))
    else:
        root.after(10, update_frame)
        return
    
    # เริ่มนับเวลาเมื่อเว็บแคมพร้อม
    if not is_webcam_ready:
        is_webcam_ready = True
        start_time = time.time()

    # คำนวณเวลาที่ผ่านไป
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    time_text = f"Time: {minutes:02d}:{seconds:02d}"
    
    # แปลงภาพเป็น RGB สำหรับ MediaPipe
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # โค้ดสำหรับการตรวจจับท่าทางและคำนวณมุมต่างๆ (เหมือนเดิม)
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
        angle_hip = math.degrees(math.atan2(right_shoulder_y - right_hip_y, right_shoulder_x - right_hip_x))
        angle_text = f"Angle Hip: {angle_hip:.2f} degrees"
        #แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        #คำนวณมุมข้อเท้า
        angle_ankle = math.degrees(math.atan2(right_knee_y - right_ankle_y, right_knee_x - right_ankle_x))
        angle_text = f"Angle Ankle: {angle_ankle:.2f} degrees"
        # แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 250, 150), 2, cv2.LINE_AA) 
        # คำนวณมุมข้อศอก
        angle_elbow = math.degrees(math.atan2(right_shoulder_y - right_elbow_y, right_shoulder_x - right_elbow_x))
        angle_text = f"Angle Elbow: {angle_elbow:.2f} degrees"
        # แสดงผลมุมบนวิดีโอ
        cv2.putText(img, angle_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA) # แสดงผลมุมบนวิดีโอ
        # แสดงผลจำนวนครั้งของการวิดพื้น
        cv2.putText(img, f'Push-ups: {push_up_count}', (430, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 2, cv2.LINE_AA) # แสดงผลจำนวนครั้งของการวิดพื้น
        cv2.putText(img, time_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # แสดงผลจำนวนเวลาของการวิดพื้น
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
                percent_accuracy.append(accuracy_percentage)

                # ในส่วนที่อัพเดทกราฟ (หลังจากคำนวณ accuracy_percentage)
                line.set_data(push_up_counts, percent_accuracy)
                ax.relim()
                ax.autoscale_view(True, True, True)

                # ตรวจสอบว่าต้องขยายแกน x หรือไม่
                if push_up_count > ax.get_xlim()[1] - 1:
                    ax.set_xlim(0, push_up_count + 1)
                
                canvas.draw()
                                    
                # รีเซ็ตค่าในรายการ vt
                vt = [0, 0, 0, 0, 0, 0]


    # แปลงภาพเป็น ImageTk
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # ตรวจสอบเงื่อนไขการสิ้นสุด
    if (target_count.get() > 0 and push_up_count >= target_count.get()) or \
        (target_time.get() > 0 and time.time() - start_time >= target_time.get()):
        # อัพเดทกราฟครั้งสุดท้าย
        line.set_data(push_up_counts, percent_accuracy)
        ax.relim()
        ax.autoscale_view(True, True, True)
        canvas.draw()
        cap.release()
        return

    # เรียกฟังก์ชันนี้อีกครั้งหลังจาก 10 มิลลิวินาที
    root.after(10, update_frame)

# เริ่มการทำงานของ GUI
root.mainloop()