import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Verwendung von NumPy, um ein 2D-Array rechteckiger Ausschnitt von [-2,2] bis [-2,2] zu erstellen,
# welcher mit 800/800 Pixel abgetstet wird.
y,x = np.mgrid[-2:2:0.005, -2:2:0.005]
#Definieren der Koodinaten für jedes pixel
zArray = x+1j*y

#Erstellung der Tensoren
C = tf.constant(zArray.astype("complex64"))
Z = tf.Variable(tf.zeros_like(C))
color = tf.Variable(tf.zeros_like(C, "int32"))
lastMask = tf.Variable(tf.zeros_like(tf.cast(C, "bool")))

def mandelbrotmenge():
    for i in range(100):
        ind = tf.abs(Z) < 2
        Zn = Z**2 + C
        Z.assign(Zn)
        newDivergent = tf.not_equal(ind, lastMask)
        color.assign(tf.where(newDivergent,i,color))
        lastMask.assign(ind)
    return color

plt.imshow(tf.abs(mandelbrotmenge()))
plt.show()