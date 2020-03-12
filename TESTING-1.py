import numpy as np
import cv2

import time
from keras.models import load_model

cap = cv2.VideoCapture ('a.mp4')

model = load_model('HAR.h5')


def Label_decode(argument): 
	switcher = { 
		0: "READING", 
		1: "SITTING",
                2: "SLEEPING", 
		3: "STANDING",
                4: "TILTING", 
		5: "WALKING",
                
	} 
	return switcher.get(argument, "nothing") 




while(cap.isOpened()):

    ret, frame = cap.read()
    img = cv2.resize(frame, (100,100))
    roi_X = np.expand_dims(img, axis=0)
    predictions = model.predict(roi_X)
    result = Label_decode(np.argmax(predictions[0]))
    print(result) 


    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame,result, (55,55), font, 1.0, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





# When everything done, release the capture
cap.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
