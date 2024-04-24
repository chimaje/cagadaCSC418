import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


video_path = "imag/SST_foyer_video.mp4"
video = None
paused = False


def start_processing():
    global video, paused
    video = cv2.VideoCapture(video_path)
    paused = False
    process_video()


def pause_processing():
    global paused
    paused = True


def resume_processing():
    global paused
    paused = False
    process_video()


def stop_processing():
    global video
    video.release()
    cv2.destroyAllWindows()


def process_video():
    global video, paused
    ret, frame = video.read()
    if ret:
        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layers_outputs = net.forward(output_layers_names)
            
        boxes = []
        confidences = []
        class_ids = []

        for output in layers_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        
        if not paused:
            label_img.imgtk = imgtk
            label_img.configure(image=imgtk)
        
        
        if not paused:
            label_img.after(10, process_video)  
    else:
        stop_processing()


root = tk.Tk()
root.title("Object Detection Application")


label_img = tk.Label(root)
label_img.pack()

# Load YOLO
net = cv2.dnn.readNet('cfg/yolov3.weights', 'cfg/yolov3.cfg')
classes = []
with open('cfg/coco.names', 'r') as f:
    classes = f.read().splitlines()

start_button = tk.Button(root, text="Start", command=start_processing)
pause_button = tk.Button(root, text="Pause", command=pause_processing)
resume_button = tk.Button(root, text="Resume", command=resume_processing)
stop_button = tk.Button(root, text="Stop", command=stop_processing)


start_button.pack(side=tk.LEFT, padx=5, pady=5)
pause_button.pack(side=tk.LEFT, padx=5, pady=5)
resume_button.pack(side=tk.LEFT, padx=5, pady=5)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)


root.mainloop()
