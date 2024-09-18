from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import csv

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

user_data = {}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def gen_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    start_time = datetime.now()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed_time = datetime.now() - start_time
            remaining_time = max(0, 30 - int(elapsed_time.total_seconds()))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)
                
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)
                    
            except:
                pass

            cv2.rectangle(image, (0, 0), (320, 73), (245, 117, 16), -1)
            
            cv2.putText(image, 'REPS', (15, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(counter), 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (65, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f'TIME LEFT: {remaining_time}s', 
                        (400, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if remaining_time == 0:
                break

    if user_data['mode'] == 'competitive':
        save_counter_data(user_data['name'], user_data['grade'], user_data['student_id'], user_data['department'], counter)
    elif user_data['mode'] == 'individual':
        if counter >= 15:
            grade = '상'
        elif counter >= 10:
            grade = '중'
        else:
            grade = '하'
        user_data['grade_result'] = grade

    cap.release()

def save_counter_data(name, grade, student_id, department, counter):
    with open('counter_data.csv', mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([name, grade, student_id, department, counter])

def get_sorted_data():
    data = []
    with open('counter_data.csv', mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            name, grade, student_id, department, counter = row[0], row[1], row[2], row[3], int(row[4])
            data.append((name, grade, student_id, department, counter))
    
    sorted_data = sorted(data, key=lambda x: x[4], reverse=True)
    return sorted_data

@app.route('/', methods=['GET', 'POST'])
def index():
    global user_data
    if request.method == 'POST':
        user_data['name'] = request.form['name']
        user_data['grade'] = request.form['grade']
        user_data['student_id'] = request.form['student_id']
        user_data['department'] = request.form['department']
        user_data['mode'] = request.form['mode']  
        return render_template('index.html', video_feed=True)
    return render_template('index.html', video_feed=False)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/leaderboard')
def leaderboard():
    sorted_data = get_sorted_data()
    return render_template('leaderboard.html', sorted_data=sorted_data, enumerate=enumerate)

@app.route('/finished')
def finished():
    if user_data['mode'] == 'individual':
        return render_template('result.html', grade_result=user_data['grade_result'])
    else:
        return redirect(url_for('leaderboard'))

if __name__ == '__main__':
    app.run(debug=True)
