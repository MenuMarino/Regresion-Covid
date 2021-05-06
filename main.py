import numpy as np
import csv
import matplotlib.pyplot as plt
from math import e
from datetime import datetime

DATE = 0; CONFIRMED = 1; DEATHS = 2; ZONE = 3; ALTITUDE = 4; LAT = 5; LON = 6;

training_file = 'covid.csv'
y_training_ds = []
x_training_ds = []
x_testing_ds = []
y_testing_ds = []

with open(training_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # n = len(list(csv_reader))
    # print('rows: ', n)
    # TODO: 70% pal training y 30% pal testing
    idx = 0
    for row in csv_reader:
        if idx != 0:
          if idx == 5:
            break
          print(idx, row)
          # print(row[DATE])
          marca_de_tiempo = datetime.strptime(str(row[DATE]), "%Y-%m-%d").timestamp()
          # print(marca_de_tiempo)
          zona = row[ZONE]
          if zona == 'ZONA NORTE':
            zona = 1.0
          elif zona == 'ZONA CENTRO':
            zona = 2.0
          elif zona == 'ZONA SUR':
            zona = 3.0
          else:
            print('unkown zona')
            exit(0)
          x_training_ds.append([float(marca_de_tiempo), float(zona), float(row[LAT]), float(row[LON])])
          y_training_ds.append(float(row[DEATHS]))
          # print(x_training_ds[idx-1])
          # print(y_training_ds[idx-1])
        idx += 1

w = np.random.rand(p)