import numpy as np 
import os 
from PIL import Image 
import os 

def  read_utk_face(folder,ageGroup=None): 
	imgs = []
	labels = []
	for f in os.listdir(folder):
		if os.path.isfile(folder + '/' + f) and len(str(f).split('_')) >= 4:
			img = Image.open(folder + '/' + f).convert('L')
			img.load()
			properties = str(f).split('_')
			#img = img.resize((128,128),Image.ANTIALIAS)
			age = properties[0]
			gender = properties[1]
			race = properties[2]
			date_time = properties[3].split('.')[0]
			data = np.asarray(img,dtype="int32")
			if ageGroup==None or ageGroup==min(int(age)/10,10):	
				imgs.append(data)
				labels.append(int(age))

	labels = map(lambda x: min(int(x / 10),10),labels)  
	return (np.asarray(imgs),np.asarray(labels))


