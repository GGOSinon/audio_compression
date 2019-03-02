from model_rnn import Model
import numpy as np

max_step = 1000000
DISPLAY_STEP = 1
SAVE_STEP = 10
data = np.load("./data/data.npy")
print(data.shape)
model = Model(data)

avg_loss = 0
for step in range(1, max_step+1):
	loss = model.train()
	avg_loss += loss/DISPLAY_STEP
	if step % DISPLAY_STEP == 0:
		test_loss = model.test()
		print("Step %d - loss : %.5f, test_loss : %.5f" % (step, avg_loss, test_loss))
		avg_loss = 0
	if step % SAVE_STEP == 0:
		model.save("./model-rnn/"+str(step//SAVE_STEP)+".ckpt")

