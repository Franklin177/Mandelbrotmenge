import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Verwendung von NumPy, um ein 2D-Array rechteckiger Ausschnitt von [-2,2] bis [-2,2] zu erstellen,
# welcher mit 800/800 Pixel abgesttet wird.
y, x = np.mgrid[-2:2:0.005, -2:2:0.005]
#Definieren der Koodinaten für jedes pixel
zArray = x+1j*y

#Erstellung von Tensoren.
C = tf.constant(zArray)
Z = tf.Variable(tf.zeros_like(C))
color = tf.Variable(tf.zeros_like(C, "int32"))
lastMask = tf.Variable(tf.zeros_like(tf.cast(C, "bool")))

def mandelbrotmenge():

      #Berechnug der Maske aus Booleschen Werte von allen divergierten und konvergierten Pixel
      ind = tf.abs(Z) < 2

      #Erstellung der Maske allen konvergierten Pixel
      maskConvergent = tf.where(ind)

      #Selectieren der konvergierten 'z' Pixels der Iteration
      zConvergent = tf.gather_nd(Z, maskConvergent)

      #Selectieren der zugehörigen 'c' Pixels der Iteration
      cConvergent = tf.gather_nd(C, maskConvergent)

      #Berechnung der Mandelbrot-Menge
      Zn = zConvergent**2 + cConvergent

      #Kopieren der neuen Maske 'Zn' in 'Z' und Aktualisierung der Tensor 'Z'
      Z.assign(tf.tensor_scatter_nd_update(Z, maskConvergent, Zn))

      #Maske aus Boolean Werte der neuen divergierten Pixels
      newDivergent = tf.not_equal(ind, lastMask)

      #Zahl der Iteration reinschreiben
      color.assign(tf.where(newDivergent, i, color))

      #Maske aus Booleschen Werte der divergierten und konvergierten Pixels speichern
      lastMask.assign(ind)

for i in range(100):
    mandelbrotmenge()


plt.imshow(tf.abs(color))
plt.show()