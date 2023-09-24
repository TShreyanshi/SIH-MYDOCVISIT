import cv2
from simple_facerec_improved import ImprovedFacerec
import datetime
import os
import openpyxl

excel_file = "recognized_faces.xlsx"
if not os.path.exists(excel_file):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Timestamp"])
else:
    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active

sfr = ImprovedFacerec()
sfr.load_encoding_images("images/")

for camera_index in range(4):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break

if not cap.isOpened():
    print("Error: Unable to open any camera.")
    exit()

recognized = False

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        if name != "Unknown" and not recognized:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            text_size = cv2.getTextSize(current_datetime, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, current_datetime, (x1, y2 + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

            ws.append([name, current_datetime])
            wb.save(excel_file)
            
            recognized = True


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27 or recognized:
        break

cap.release()
cv2.destroyAllWindows()
