import cv2
import numpy as np 
import os 
import time
import datetime
#######KNN code###########
def distance(v1, v2):
	print(len(v1))
	print(len(v2))
	return np.sqrt(((v1-v2)**2).sum())
	

def knn(train, test, k=5): #// takes training,testing  data and value of k
	dist = []
	
	for i in range(train.shape[0]):
		ix = train[i, :-1] #//vector
		iy = train[i, -1] #//label
		d = distance(test, ix) #// computing distance from test point x
		dist.append([d, iy])
	dk = sorted(dist, key=lambda x: x[0])[:k] #// sorting based on distance and get top k
	labels = np.array(dk)[:, -1]  #// only labels are retrieved
	
	output = np.unique(labels, return_counts=True) #// frequencies of each label
	index = np.argmax(output[1])#//finds max frequency and corres labels
	return output[0][index]

########################################################
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = [] # // loads x values
labels = [] #// y values

class_id = 0 #// labels for given file will be incremented acc
names = {} #// mapping btw id and name 

# data preparation
for fx in os.listdir(dataset_path): #// gives all files in data path
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		print("Loaded "+fx)
		data_item = np.load(dataset_path+fx)  #// file name with its path
		face_data.append(data_item)

	#creating labels  for class
	target = class_id*np.ones((data_item.shape[0],))
	class_id += 1
	labels.append(target)
#concatenating list of targets and data item
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

time_taken_aggregate = 0
run_count = 0
##testing part  go to first code if not copied
while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue

	for face in faces:
		x,y,w,h = face

		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		start_time = time.time_ns()
		out = knn(trainset,face_section.flatten())
		time_taken = time.time_ns() - start_time
		run_count += 1
		time_taken_aggregate += time_taken

		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF # // wait for 1ms till window is destroyed
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
print(f'Avg. time taken: {time_taken_aggregate / run_count / 1000} us.')
