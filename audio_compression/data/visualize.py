from PIL import Image
import numpy as np

data = np.load("data.npy")
data_size = 5
for i in range(5):
	img = data[i]
	cnt = 0
	'''
	for t in range(255, 1, -1):
		for f in range(255, 1, -1):
			img[t][f] -= img[t][f-1]
	'''
	for t in range(255, 1, -1):
		for f in range(255, 1, -1):
			img[t][f] = abs(img[t][f] - img[t-1][f])
			if img[t][f] > 150/255.: cnt+=1
	print("Data %d : %d" % (i, cnt))

	img = (img - img.min())*(img.max()-img.min())
	img = (img * 255.).astype(np.uint8)
	img = Image.fromarray(img, 'L')
	img.save('temp'+str(i)+'.png')
