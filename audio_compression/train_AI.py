from model import Model
import numpy as np
max_step = 1000000
DISPLAY_STEP = 100
data = np.load("./data/data.npy")
print(data.shape)
model = Model(data)

avg_C_loss, avg_G_loss = 0, 0
for step in range(1, max_step+1):
	C_loss, G_loss = model.train()
	avg_C_loss += C_loss/DISPLAY_STEP
	avg_G_loss += G_loss/DISPLAY_STEP
	if step % DISPLAY_STEP == 0:
		test_C_loss, test_G_loss = model.test()
		print("Step %d - C_loss : %.5f, G_loss : %.5f test_C_loss : %.5f test_G_loss : %.5f" % (step, avg_C_loss, avg_G_loss, test_C_loss, test_G_loss))
		avg_loss = 0


