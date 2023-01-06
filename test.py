import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig = plt.figure(figsize=(15,8))
axFFT = plt.axes([0.05, 0.2, 0.75, 0.75])

x, y = [],[]
sc = axFFT.scatter(x,y)
plt.xlim(0,10)
plt.ylim(0,10)

def on_close(event):
    print("closed")
    return

fig.canvas.mpl_connect('close_event', on_close)
plt.draw()
for i in range(1000):
    
    sc.set_offsets(np.c_[x,y])
    fig.canvas.draw_idle()
    plt.pause(0.1)

plt.waitforbuttonpress()