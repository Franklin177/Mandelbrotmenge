#Importieren der Bibliotheken
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Verwendung von NumPy, um ein 2D-Array rechteckiger Ausschnitt von [-2,2] bis [-2,2] zu erstellen,
# welcher mit 800/800 Pixel abgetstet wird.
y,x = np.mgrid[-2:2:0.005, -2:2:0.005]

#Definieren der Koodinaten für jedes pixel
zArray = x+1j*y

#Erstellung der Tensoren zur Berechnung
C = tf.constant(zArray.astype("complex64"))
Z = tf.Variable(tf.zeros_like(C))

#Erstellung der Tensoren zur Färbung
color = tf.Variable(tf.zeros_like(C, "int32"))
lastMask = tf.Variable(tf.zeros_like(tf.cast(C, "bool")))

def mandelbrotmenge():
    #Setzen der Iterationsschritte auf '100'
    for i in range(100):
        #Berechnug der Maske aus Booleschen Werte von allen divergierten und konvergierten Pixel
        ind = tf.abs(Z) < 2
        
        #Berechnung der Mandelbrot-Menge:(Z_(n+1)=Z_n^2+C ).
        Zn = Z**2 + C
        
        #Aktualisierung der Tensor 'Z'
        Z.assign(Zn)
        
        #Maske aus Boolean Werte der neuen divergierten Pixels
        newDivergent = tf.not_equal(ind, lastMask)
        
        #Zahl der Iteration in 'color' reinschreiben
        color.assign(tf.where(newDivergent,i,color))
        
        #Maske aus Booleschen Werte der divergierten und konvergierten Pixels speichern
        lastMask.assign(ind)
    return color

#Visualisierung der Mandelbrot-menge
plt.imshow(tf.abs(mandelbrotmenge()))
plt.show()
