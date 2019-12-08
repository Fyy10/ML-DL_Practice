import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 10, 100)
y1 = x**2 + 1
y2 = x
plt.figure('LearnPLT')
plt.plot(x, y1, c='r')
plt.plot(x, y2, c='g')
plt.show()

plt.figure('Continue learning PLT')
new_ticks = np.linspace(0, 100, 11)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([1, 2, 3, 4, 5], [r'$\alpha num_1$', r'$\beta num_2$', r'$\gamma num_3$', r'$\zeta num_4$', r'$\phi num_5$'])
plt.plot(y1, y2, c='b')
plt.xlim(0, 101)
plt.ylim(-1, 10)
plt.xlabel('this is x')
plt.ylabel('this is y')
plt.show()

plt.figure()
# get current axis
ax = plt.gca()
print(ax)
ax.spines['top'].set_color('red')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.show()

plt.figure(num=2)
l1, = plt.plot(x, y1, label='line 1')   # use comma here coz plot returns a 'list' with only one element
l2, = plt.plot(x, y2, label='line 2')
print(l1, l2)
# plt.legend(loc='upper left')
plt.legend(handles=[l1, l2], labels=['l1', 'l2'], loc='best')
plt.show()


def fun(input_x):
    return 2 * input_x + 1


# some basic plotting
x = np.linspace(-3, 3, 100)
y = fun
plt.figure('Annotations')
plt.plot(x, y(x), label=r'$y = 2 x + 1$')
plt.legend(loc='best')
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
x0 = 1
y0 = y(x0)
plt.plot([x0, x0], [0, y0], 'k--', linewidth=2.5)
plt.scatter([x0], [y0], s=50, c='b')
# make annotations
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
# xy: position of point to annotate
# xycoords='data': choose the position based on the data
# xytext: the position relative to the point

# put text
plt.text(-4, 3, r'$This\ is\ a\ special\ point$', fontdict={'size': 16, 'color': 'r'})
plt.show()

# Scatter
plt.figure('Scatter')
n = 1024    # data size
x = np.random.rand(n)
y = np.random.rand(n)
T = np.arctan2(x, y)    # for color value
plt.scatter(x, y, c=T, alpha=0.5)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(())  # ignore x ticks
plt.yticks(())  # ignore y ticks
plt.show()

# Bar
plt.figure('Bar')
n = 12
x = np.arange(n)
y1 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
y2 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(x, +y1, facecolor='#9999ff', edgecolor='white')
plt.bar(x, -y2, facecolor='#ff9999', edgecolor='white')

# ha: horizontal alignment
# va: vertical alignment
for xa, ya in zip(x, y1):
    plt.text(xa, ya + 0.05, '%.2f' % ya, ha='center', va='bottom')

for xa, ya in zip(x, y2):
    plt.text(xa, -ya - 0.05, '%.2f' % ya, ha='center', va='top')

plt.xlim(-1, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()


# Contour
def f(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)


plt.figure('Contour')
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
# contour filling
plt.contourf(X, Y, f(X, Y), 10, alpha=0.75, cmap=plt.cm.hot)
# contour
C = plt.contour(X, Y, f(X, Y), 10, colors='black')
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
plt.show()

# Image
x = np.random.rand(3, 3)
print(x)
plt.imshow(x, cmap='bone', interpolation='nearest', origin='lower')
plt.colorbar(shrink=0.92)
plt.xticks(())
plt.yticks(())
plt.show()

# plot 3D
fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)
# rstride: row
# cstride: column
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax.contourf(x, y, z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
ax.set_zlim(-2, 2)
plt.show()
