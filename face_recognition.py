import numpy
import csv
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

my_data = genfromtxt('sample.csv', delimiter=',')
new_data = [[0] * 4096] * 1298
celeb_name=[0]*1298
j=0

with open('sample.csv') as f:
    reader = csv.reader(f, delimiter=",")
    for i in reader:
        celeb_name[j] = str(i[0])
        j+=1

for i in range(0, len(my_data)):
    new_data[i] = my_data[i][2:]

X=numpy.array(new_data)
Y=numpy.array(celeb_name)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X,Y)
knn.fit(X_train,y_train)

my_data_test = genfromtxt('face_test.csv', delimiter=',')
new_data_test = [[0] * 4096] * 1298
for i in range(0, len(my_data_test)):
    new_data_test[i] = my_data_test[i][2:]
X_test_new=numpy.array(new_data_test)
#pred = knn.predict(X_test_new)
pred = knn.predict(X_test)

print("******************************")
print(" ")
print("accuracy_score: ")
print(" ")
print(accuracy_score(y_test,pred) * 100)
print(" ")
print("******************************")


# # creating odd list of K for KNN
# myList = list(range(1,50))

# # subsetting just the odd ones
# neighbors = filter(lambda x: x % 2 != 0, myList)

# # empty list that will hold cv scores
# cv_scores = []

# # perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, Y, cv=10, scoring='accuracy')
#     cv_scores.append(scores.mean())

# # changing to misclassification error
# MSE = [1 - x for x in cv_scores]

# # determining best k
# optimal_k = neighbors[MSE.index(min(MSE))]
# print ("The optimal number of neighbors is %d" % optimal_k)



