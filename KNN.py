#!/usr/bin/env python
# coding: utf-8

# > **Knn Algorithm Pseudocode:**
# > 1. Calculate “d(x, xi)” i =1, 2, ….., n; where d denotes the Euclidean distance between the points.
# > 2. Arrange the calculated n Euclidean distances in non-decreasing order.
# > 3. Let k be a +ve integer, take the first k distances from this sorted list.
# > 4. Find those k-points corresponding to these k-distances.
# > 5. Let ki denotes the number of points belonging to the ith class among k points i.e. k ≥ 0
# > 6. If ki >kj ∀ i ≠ j then put x in class i.

# In[23]:


# Imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from collections import Counter

class KNN:
    
    def __init__(self,number_of_neighbers):
        self.number_of_neighbers=number_of_neighbers
    
    def fit(self,samples,labels):
        self.samples=samples
        self.labels=labels
    
    def predict(self,samples):
        
        distances=[]
        
        #Find the distances of each points from all the data points
        for sample in self.samples:
            distances=[self._ecludian_distance(sample, p) for p in samples]
        
        #Sort the distances 
        # we only need the index of data so we are using argsort
        #We only need the distances of number of neighbers that we specified in init 
        sorted_distances_index=np.argsort(distances)[:self.number_of_neighbers]
        
        #Now we need the lables of the sorted data point
        number_of_neighbers_label=[self.labels[i] for i in sorted_distances_index]
        
        #At last the voting from the label, the maximum number of label in the list is the predicted label
        w_label = Counter(number_of_neighbers_label).most_common(1)
        
        print(w_label[0][0])
        #print(distances)
        #print(sorted_distances_index)
        return w_label[0][0]
    

    def _ecludian_distance(self,point1,point2):
        
        distance=np.sqrt(np.sum((point1 - point2) ** 2))
        
        return distance
        
        
        
        
    



iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

#plot the dataset
plt.scatter(X[:,0], X[:,1])
plt.show()

number_of_neighbers = 3
knn_clf = KNN(number_of_neighbers)
knn_clf.fit(X_train, y_train)
predictions = knn_clf.predict(X_test)





