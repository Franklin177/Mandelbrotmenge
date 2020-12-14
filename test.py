import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Verwendung von NumPy, um ein 2D-Array rechteckiger Ausschnitt von [-2,2] bis [-2,2] zu erstellen,
# welcher mit 800/800 Pixel abgetstet wird.
x, y = np.mgrid[-2:2:0.005, -2:2:0.005]

#Definieren der Koodinaten für jedes pixel
z = x+1j*y


c = tf.constant(z,"complex64")
z = tf.Variable(c)
i = tf.Variable(0)

lastMask = tf.Variable(tf.zeros_like(tf.cast(c,"bool")))
color = tf.Variable(tf.zeros_like(c, "int32"))


@tf.function
def graphe(z,i,color):

      #i.assign(i + 1)

      # Convergierten Pixels berechnen
      ind = tf.abs(z) < 2

      # Convergierte Maske
      maskConvergent = tf.where(ind)

      #selectieren der convergierten 'z' Pixels der Iteration
      zConvergent = tf.gather_nd(z, maskConvergent)

      # selectieren der convergierten 'z' Pixels der Iteration
      cConvergent = tf.gather_nd(c, maskConvergent)

      #Berechnung der Mandelbrot-Menge
      Z = zConvergent**2 + cConvergent

      #Pixels aktualisieren und kopieren der neuen Maske in 'z'
      z.assign(tf.tensor_scatter_nd_update(z, maskConvergent, Z))

      #die neuen divergierten Pixels
      newDivergent = tf.equal(~ind, lastMask)

      #Zahl der Iteration reinschreiben
      color.assign(tf.where(newDivergent, i, color))

      #Convergierten Pixels speichern
      lastMask.assign(ind)


for n in range(100):
    i.assign(i + 1)
    graphe(z, i, color)

plt.imshow(tf.abs(color))
plt.show()

