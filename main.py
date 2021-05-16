import numpy as np
import csv
import matplotlib.pyplot as plt
from math import e, floor, ceil
from datetime import datetime
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from time import time
p: int = 4 # grado del polinomio de nuestro modelo de regresión no lineal

REGION = 0; REGION_ORIG = 1; DATE = 2; CONFIRMED = 3; DEATHS = 4

filename = 'covid.csv'
y_training_ds = []
x_training_ds = []
x_testing_ds = []
y_testing_ds = []

n: int = -1
m: int = -1
q: int = -1

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # n = len(list(csv_reader))
    n = 9934
    training_rows = floor(n*0.7)
    m = training_rows
    testing_rows = ceil(n*0.3)
    q = testing_rows
    print('total rows: ', n)
    print('training rows: ', training_rows)
    print('testing rows: ', testing_rows)
    idx: int = 0
    offset: int = datetime.strptime("01/01/2020", "%m/%d/%Y").timestamp()

    for row in csv_reader:
        if idx != 0:
          marca_de_tiempo: int = datetime.strptime(str(row[DATE]), "%m/%d/%Y").timestamp()
          nueva_marca_de_tiempo: float = (marca_de_tiempo - offset)/(3600 * 24)
          # print(nueva_marca_de_tiempo)
          new_x = [1, np.log(float(nueva_marca_de_tiempo)), np.log(float(row[REGION])), np.log(float(row[REGION_ORIG]))]
          new_y = np.log(float(row[DEATHS]))
          if idx <= training_rows:
            #print('training row')
            x_training_ds.append(new_x)
            y_training_ds.append(new_y)
          else:
            #print('testing row')
            x_testing_ds.append(new_x)
            y_testing_ds.append(new_y)
        idx += 1

print('total rows after insertion: ', n)
print('training rows after insertion: ', len(x_training_ds))
print('testing rows after insertion: ', len(x_testing_ds))

#rs = RandomState(MT19937(SeedSequence(123456789))); rs.rand(4)
#w = rs.rand(p)
w = np.random.rand(p)
print("TERMINOS: ",w)
exit(0)
landa = 1.5
alfa = 0.0025
gamma = 0.9
epsilon = pow(10, -8)
v = [0] * p
print(w)
# hipótesis
def h(w, x, j):
  # w: parametros
  # x: vector de caracteristicas
  # return: valor que predice el modelo 'h' para el j-esimo data point 
  return np.sum([w[i]*(x[j][i]**i) for i in range(p)]) 
  
# true = y_ds
# pred = h(x_training_ds)
def mse(true, h):
  return np.sum([(true[i] - h(w, x_training_ds, i))**2 for i in range(m)])/(2*m)

# true = y_ds
# pred = h(x_training_ds)
def mae(true, h):
  return np.sum([np.abs(true[i] - h(w, x_training_ds, i)) for i in range(m)])/m


def derivada_l1(true, h, l, w, j, x):
  term1 = np.sum([((true[i]-h(w, x, i)) / (np.abs(true[i]-h(w, x, i))))*(-x[i][j]**j) for i in range(m)])
  return term1

def derivada_l2(true, h, l, w, j, x):
  term1 = np.sum([ (true[i] - h(w, x, i))*(-x[i][j]**j) for i in range(m) ])/m
  return term1


def derivada_l2_regularizada(true, h, l, w, j, x):
  term1 = np.sum([ (true[i] - h(w, x, i))*(-x[i][j]**j) for i in range(m) ])/m
  term2 = l*2*w[j]
  return term1 + term2

def derivada_l1_regularizada(true, h, l, w, j, x):
  term1 = np.sum([((true[i]-h(w, x, i)) / (np.abs(true[i]-h(w, x, i))))*(-x[i][j]**j) for i in range(m)])
  return term1 + (l*(w[j]/np.abs(w[j])))

unidades = []
errores = []

def test():
  k = 1
  while (k < 100):
    unidades.append(k)
    grads = [derivada_l2(y_training_ds, h, landa, w, j, x_training_ds) for j in range(p)]
    G = [(alfa/(np.sqrt((grads[i]**2)+epsilon)))*grads[i] for i in range(p)]

    for i in range(p):
        #v[i] = gamma*v[i] + alfa*grads[i]
        w[i] = w[i] - G[i]
    err = mse(y_training_ds, h)
    errores.append(err)
    #print(err)
    k += 1


start_time = time()
test()
elapsed_time = time() - start_time

print('error: ', mse(y_training_ds, h))
print("Elapsed time: %.10f seconds." % elapsed_time)
# print(unidades)
# print(errores)
plt.plot(unidades, errores)


'''
for i in range(q-1):
    row = x_testing_ds[i]
    true = y_testing_ds[i]
    print('true: ', e**true)
    print(e**h(w, x_testing_ds, i))

   # 1.48 landa 1 alfa 0.003
   # 1.47 landa 1.5  """"
   # 1.466 landa 1.5 0.0025
'''
