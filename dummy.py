import cv2
cap=cv2.VideoCapture(0)
while True:
    r,frame=cap.read()
    print(frame)

    cv2.imshow("GRAY",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # After the loop release the cap object
cap.release()
    # Destroy all the windows
cv2.destroyAllWindows()