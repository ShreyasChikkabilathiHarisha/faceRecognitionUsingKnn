from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor

my_data = genfromtxt('face.csv', delimiter=',')
new_data = [[0] * 4096] * 1298
for i in range(0, len(my_data)):
    #print i
    new_data[i] = my_data[i][0:]
#my_data = my_data[2:]
#print(new_data[1])

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.predict(new_data)
