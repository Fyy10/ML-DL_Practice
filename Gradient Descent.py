'Basic application of gradient descent'

__author__ = 'Jeff Fu'

import numpy as np
import matplotlib.pyplot as plt

x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# ydata = b + w * xdata

x = np.arange(-200, -100, 1)	#bias
y = np.arange(-5, 5, 0.1)		#weight
z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
	for j in range(len(y)):
		b = x[i]
		w = y[j]
		z[j][i] = 0
		for n in range(len(x_data)):
			z[j][i] = z[j][i] + (y_data[n] - b - w*x_data[n])**2
		z[j][i] = z[j][i]/len(x_data)

b = -120
w = -4

lr = 1
iteration = 100000

b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for i in range(iteration):
	b_grad = 0
	w_grad = 0
	for n in range(len(x_data)):
		b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
		w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
	b_grad = b_grad / len(x_data)
	w_grad = w_grad / len(x_data)

	lr_b = lr_b + b_grad ** 2
	lr_w = lr_w + w_grad ** 2

	b = b - lr/np.sqrt(lr_b) * b_grad
	w = w - lr/np.sqrt(lr_w) * w_grad

	b_history.append(b)
	w_history.append(w)

plt.contourf(x, y, z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
print b, w
