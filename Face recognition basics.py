import cv2
import face_recognition
from face_recognition.api import face_distance

image= face_recognition.load_image_file("Images_attendance/rohit.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_test = face_recognition.load_image_file("Images_attendance/shahrukh.jpg")
image_test=cv2.cvtColor(image_test,cv2.COLOR_BGR2RGB)

image_location=face_recognition.face_locations(image)
image_test_location=face_recognition.face_locations(image_test)
my_face_encoding = face_recognition.face_encodings(image)[0]
unknown_face_encoding = face_recognition.face_encodings(image_test)[0]
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
image_distance=face_recognition.face_distance([my_face_encoding], unknown_face_encoding)
print(image_distance)
cv2.rectangle(image,(image_location[0][3],image_location[0][0]),(image_location[0][1],image_location[0][2]),(255,0,0),2)
cv2.rectangle(image_test,(image_test_location[0][3],image_test_location[0][0]),(image_test_location[0][1],image_test_location[0][2]),(255,0,0),1)
cv2.putText(image_test,f"{results} {round(image_distance[0],2)}",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,0,255],2)
cv2.imshow('image',image)
cv2.imshow('image_test',image_test)
cv2.waitKey(0)
cv2.destroyAllWindows()