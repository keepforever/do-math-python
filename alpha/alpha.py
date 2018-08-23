import numpy as np
import pandas as pd
import pickle

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report

# load the dataset
iris = datasets.load_iris()
# scale data for computation
X = scale(iris.data)
y = pd.DataFrame(iris.target)

print(X[0:10,])

# build model
# my_k_means_model = KMeans(n_clusters=3, random_state=5)
# my_k_means_model.fit(X)

#pickle model for later
# pickle.dump(my_k_means_model, open('my_saved_model.pkl', 'wb'))

my_loaded_model = pickle.load(open('./my_saved_model.pkl', 'rb'))

test = [0.173673948, 0.587763531, 0.194101603, 0.133225943]
test_2 = [-0.53717756,  1.95766909, -1.17067529, -1.05003079]


print(my_loaded_model.predict([test]), 'yello')
print(my_loaded_model.predict([test_2]), 'yello_2')

# reassing lable names to predicted lables
# lable [0 => 2, 1 => 0, 2 =>1]
relabel = np.choose(my_loaded_model.labels_, [2, 0, 1]).astype(np.int64)
print(my_loaded_model.predict([test]), 'yello_balls')
print(my_loaded_model.predict([test_2]), 'yello_2_balls')


print('class report: ')
print(classification_report(y, relabel))
