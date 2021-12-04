import numpy as np
import cv2 as cv
from numpy.linalg import norm

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def brightness(img):
    
    img = cv.resize(img, (500,500), interpolation=cv.INTER_AREA) #resize the image
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   #convert image to grayscale
    
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4)                        
    #detect face
    
    print(f'No of faces found in image: {len(faces_rect)}')

    for (x,y,z,w) in faces_rect:
        cv.rectangle(img, (x,y), (x+z,y+w), (255,0,0), 1)
    
    #cv.imshow('Before',img)
    img = img[y:y+w, x:x+z] 
    cv.imshow('After',img)                                                                                  
    
    #based-on face detected crop the image
    if len(img.shape) == 3:
     # Colored RGB or BGR (*Do Not* use HSV images with this function)
     # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
    # Grayscale
        return np.average(img) #average brightness of image



#--------------------Testing---------------------------------------------------

#Case:1

#print(f"The brightness accuracy of image is: {(brightness(cv.imread('test3.jpg',cv.IMREAD_COLOR))/200)*100}")


#Case:2

# capture = cv.VideoCapture(0)

# while True:
#     isTrue, frame = capture.read()
    
#     #cv.imshow('Video', frame)
#     print((brightness(frame)/200)*100)
#     if cv.waitKey(1) & 0xFF==ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()

#-----------------------------------------------------------------------------
cv.waitKey(0)