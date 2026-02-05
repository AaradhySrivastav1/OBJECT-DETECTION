import cv2
import numpy as np
import imutils

# ---------- RATIOS (tumhare physical sheet ke) ----------
x_ratio = 0.005
y_ratio = 0.57
w_ratio = 0.99
h_ratio = 0.22

cap = cv2.VideoCapture(0)

print("SPACE = capture | ESC = exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=1000)
    display = frame.copy()

    # -------- LIVE GRAYSCALE + THRESHOLD ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    th = cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,7)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # -------- LIVE CONTOUR ----------
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts)>0:
        c = max(cnts,key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        cv2.drawContours(display,[box],-1,(0,255,0),2)

    # -------- SHOW LIVE WINDOWS ----------
    cv2.imshow("Camera + Green Box", display)
    cv2.imshow("Live Threshold", th)

    key = cv2.waitKey(1)

    # ---------- CAPTURE ----------
    if key == 32:   # SPACE
        print("Captured")

        sheet = frame.copy()

        H,W = sheet.shape[:2]

        x=int(W*x_ratio)
        y=int(H*y_ratio)
        cw=int(W*w_ratio)
        ch=int(H*h_ratio)

        crop = sheet[y:y+ch,x:x+cw]

        cv2.imshow("Final MCQ Block", crop)
        cv2.waitKey(0)
        break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
