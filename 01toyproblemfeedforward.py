from cmath import isclose
from ctypes.wintypes import PINT
from re import X
import numpy as np

with open ('data\data_dark_bright_test_4000.csv', 'rt') as f:
    data_list = f.readlines()
    f.close()

def sigma(z):
    return 1 / (1 + np.exp(-z))
    
w_a = np.array([[-0.3, -0.7, -0.9, -0.9], [-1, -0.6, -0.6, -0.6], [0.8, 0.5, 0.7, 0.8]])
w_b = np.array([[2.6, 2.1, -1.2], [-2.3, -2.3, 1.1]])


#h = np.dot(w_a, )
anatol_list = []

for line in data_list:
    line = line.strip("\n")
    line = line.split(",")
    num_list = []
    for num in line:
        num = int(num)
        num_list.append(num)
    
    anatol_list.append(num_list)


for used_line in range(len(data_list)):
    x = []
    answer = anatol_list[used_line][0]
    for element in anatol_list[used_line][1::]:
        x.append(element)  
    
    x_vector = np.array(x)

    h = sigma(np.dot(w_a, x_vector))

    y = sigma(np.dot(w_b, h))
    correct = 0
    not_correct = 0
    if y[0] > y[1]:
        guess = 1
    elif y[0] < y[1]:
        guess = 0
    else:
        print("equal")
    if guess == answer:
        correct += 1
    else:
        not_correct += 1
print(correct)
print(100/(correct+not_correct)*correct)
"""
    print("x vector=", x_vector)
    print("hidden layer=", h)
    print("output layer=", y)
"""