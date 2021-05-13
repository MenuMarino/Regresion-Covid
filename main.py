import numpy as np
import csv
import matplotlib.pyplot as plt
from math import e, floor, ceil
from datetime import datetime

p: int = 5 # grado del polinomio de nuestro modelo de regresión no lineal

DATE = 0; CONFIRMED = 1; DEATHS = 2; ZONE = 3; ALTITUDE = 6; LAT = 4; LON = 5;

filename = 'covid.csv'
y_training_ds = []
x_training_ds = []
x_testing_ds = []
y_testing_ds = []

n: int = -1
m: int = -1

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # n = len(list(csv_reader))
    n = 9944
    training_rows = floor(n*0.7)
    m = training_rows
    testing_rows = ceil(n*0.3)
    print('total rows: ', n)
    print('training rows: ', training_rows)
    print('testing rows: ', testing_rows)
    idx: int = 0
    for row in csv_reader:
        if idx != 0: # ignore header
          # if idx == 5: # only first 5 rows
          #   break
          #print(idx, row)
          # print(row[DATE])
          marca_de_tiempo: int = datetime.strptime(str(row[DATE]), "%m/%d/%Y").timestamp()
          # print(marca_de_tiempo)
          zona: str = row[ZONE]
          if zona == 'ZONA NORTE':
            zona = 1.0
          elif zona == 'ZONA CENTRO':
            zona = 2.0
          elif zona == 'ZONA SUR':
            zona = 3.0
          else:
            print('unkown zona')
            exit(0)
          new_x = [1, float(marca_de_tiempo), float(zona), float(row[LAT]), float(row[LON])]
          new_y = float(row[DEATHS])
          if idx <= training_rows:
            #print('training row')
            x_training_ds.append(new_x)
            y_training_ds.append(new_y)
          else:
            #print('testing row')
            x_testing_ds.append(new_x)
            y_testing_ds.append(new_y)
          
          # print rows
          # print(x_training_ds[idx-1])
          # print(y_training_ds[idx-1])
        idx += 1

print('total rows after insertion: ', n)
print('training rows after insertion: ', len(x_training_ds))
print('testing rows after insertion: ', len(x_testing_ds))

w = np.random.rand(p)
landa = 0.1
alfa = 0.001
gamma = 0.9
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
    for i in range(p):
        v[i] = gamma*v[i] + alfa*grads[i]
        w[i] = w[i] - v[i]
    #err = mse(y_training_ds, h)
    #errores.append(err)
    k += 1

test()
print(h(w, x_training_ds, 1))
    
print(unidades)
print(errores)
plt.plot(unidades, errores)