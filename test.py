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
i = tf.Variable(0)
color = tf.Variable(tf.zeros_like(C, "int32"))
lastMask = tf.Variable(tf.zeros_like(tf.cast(C, "bool")))

def mandelbrotmenge():

      #Maske aus Booleschen Werte der Divergierte und Convergierten Pixels berechnen
      ind = tf.abs(Z) < 2

      #Convergierte Maske
      maskConvergent = tf.where(ind)

      #selectieren der convergierten 'z' Pixels der Iteration
      zConvergent = tf.gather_nd(Z, maskConvergent)

      # selectieren der zugehörigen 'c' Pixels der Iteration
      cConvergent = tf.gather_nd(C, maskConvergent)

      #Berechnung der Mandelbrot-Menge
      Zn = zConvergent**2 + cConvergent

      #Pixels aktualisieren und kopieren der neuen Maske in 'z'
      Z.assign(tf.tensor_scatter_nd_update(Z, maskConvergent, Zn))

      #Maske aus Boolean Werte der neu divergierten Pixels
      newDivergent = tf.not_equal(ind, lastMask)

      #Zahl der Iteration reinschreiben
      color.assign(tf.where(newDivergent, i, color))

      #Maske aus Booleschen Werte der Divergierte und Convergierten Pixels speichern
      lastMask.assign(ind)


for n in range(100):
    mandelbrotmenge()
    i.assign(i + 1)

plt.imshow(tf.abs(color))
plt.show()

