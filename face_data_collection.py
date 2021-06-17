import cv2           #read & write  images
import numpy as np

cap = cv2.VideoCapture(0)       #capturing image from default camera that’s why 0
#face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")  # creating classifier object which works on facial data

skip = 0
face_data = [] #array
dataset_path = './data/' # data folder
file_name = input("Enter the name of the person : ")
while True:
	ret,frame = cap.read()        #returns Boolean value and frame that has been captured
 
	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #// converting to gray frame
	

	faces = face_cascade.detectMultiScale(frame,1.3,5)    #// giving face to classifier using Multiscale  1.3(30%)= scale factor(how much image size is reduced at each img scale,5=no.of neighbours each rectangle should have(3-6 good values)//
	if len(faces)==0:
		continue
		
	faces = sorted(faces,key=lambda f:f[2]*f[3]) #// sorting on area using lambda function so that largest one will be picked
	for face in faces[-1:]:
		x,y,w,h = face #// values of these at face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) #// drawing rectangle around frame with color
#extract region of interest 
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]  #// slicing of face section
		face_section = cv2.resize(face_section,(100,100))  

		skip += 1
		if skip%10==0:  #//storing every 10th frame
			face_data.append(face_section)
			print(len(face_data))   #//no of faces captured

	cv2.imshow("Frame",frame)  #// display the frame
	cv2.imshow("Face Section",face_section)
#wait for user input q then stop the loop
	key_pressed = cv2.waitKey(1) & 0xFF  #// wait for 1ms till window is destroyed
	if key_pressed == ord('q'):    #//ord gives ASCII values of q =113
         break

# convert to np array		
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()             
cv2.destroyAllWindows()
