import numpy as np
import csv
import matplotlib.pyplot as plt
from math import e, floor, ceil
from datetime import datetime

p: int = 6 # grado del polinomio de nuestro modelo de regresión no lineal

DATE = 0; CONFIRMED = 1; DEATHS = 2; ZONE = 3; ALTITUDE = 4; LAT = 5; LON = 6;

filename = 'covid.csv'
y_training_ds = []
x_training_ds = []
x_testing_ds = []
y_testing_ds = []

all_rows: int = -1

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # all_rows = len(list(csv_reader))
    all_rows = 9787
    training_rows = floor(all_rows*0.7)
    testing_rows = ceil(all_rows*0.3)
    print('total rows: ', all_rows)
    print('training rows: ', training_rows)
    print('testing rows: ', testing_rows)
    idx: int = 0
    for row in csv_reader:
        if idx != 0: # ignore header
          # if idx == 5: # only first 5 rows
          #   break
          print(idx, row)
          # print(row[DATE])
          marca_de_tiempo: int = datetime.strptime(str(row[DATE]), "%Y-%m-%d").timestamp()
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
          new_x = [float(marca_de_tiempo), float(zona), float(row[LAT]), float(row[LON])]
          new_y = float(row[DEATHS])
          if idx <= training_rows:
            print('training row')
            x_training_ds.append(new_x)
            y_training_ds.append(new_y)
          else:
            print('testing row')
            x_testing_ds.append(new_x)
            y_testing_ds.append(new_y)
          
          # print rows
          # print(x_training_ds[idx-1])
          # print(y_training_ds[idx-1])
        idx += 1

print('total rows after insertion: ', all_rows)
print('training rows after insertion: ', len(x_training_ds))
print('testing rows after insertion: ', len(x_testing_ds))

w = np.random.rand(p)