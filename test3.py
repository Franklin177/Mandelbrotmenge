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
Z = tf.Variable(C)
n = tf.Variable(tf.zeros_like(C, "float32"))

@tf.function
def mandelbrotmenge():
    for i in range(100):
        ind = tf.abs(Z) < 2
        Zn = Z**2 + C
        Z.assign(Zn)
    n.assign(tf.cast(ind, "float32"))
    return n

plt.imshow(mandelbrotmenge())
plt.show()
