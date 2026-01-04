import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("Read success:", ret)
if ret:
    print("Frame shape:", frame.shape)
cap.release()
