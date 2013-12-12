# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn import svm
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

def array_to_file(vectors, filename):
    f_out = open(filename, "w")
    if len(vectors.shape) == 2:
        for vector in vectors:
            strings = ["%.2f" % number for number in vector]
            f_out.write("%s\n" % '\t'.join(strings))
    else:
        strings = ["%.2f" % number for number in vectors]
        f_out.write("%s\n" % '\t'.join(strings))   
    f_out.close()

# <codecell>

# Load data set
file_training = "data/Training50K.csv"

# read data values
training_data = np.genfromtxt(file_training, dtype=float, skip_header=1, delimiter='\t')

# Features (data cols from 0 to 15)
X = training_data[:,range(0,16)]

# Labels (data cols)
labels = {"y1": 15, "y2":16, "y3":17}
label = "y3"
y = training_data[:,labels[label]]
print("\nTraining SVM for %s label" %label)

# <codecell>

# K-Fold validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=1)

print("\nK-Fold validation:")
print("X train: %s" % str(X_train.shape))
print("y train: %s" % str(y_train.shape))
print("X test: %s" % str(X_test.shape))
print("y test: %s" % str(y_test.shape))

# <codecell>

# Classify
gamma_parameter = 0.01

# Create model
clf = svm.SVC(kernel='rbf', gamma=gamma_parameter, probability=False)

# Clasify
clf.fit(X_train, y_train) 


# Dump info about the model
print("\nSupported vectors length: %s" % str(clf.support_vectors_.shape))
print("Dual coef. length: %s" % str(clf.dual_coef_.shape))

# <codecell>

score = clf.score(X_test, y_test) 
print("\nScore k-fold validation: %f" % score)

# <codecell>

# Save support vectors in a file
array_to_file(clf.support_vectors_, "data/svm_models/%s_model_supported_vectors.tsv" % label)
array_to_file(clf.dual_coef_, "data/svm_models/%s_model_dual_coef.tsv" % label)

array_to_file(X_test, "data/svm_models/%s_model_X_test.tsv" % label)
array_to_file(y_test, "data/svm_models/%s_model_y_test.tsv" % label)

array_to_file(X_train, "data/svm_models/%s_model_X_train.tsv" % label)
array_to_file(y_train, "data/svm_models/%s_model_y_train.tsv" % label)

# <codecell>

# Vector index to test (from dataset)
index = 1
vector = X[index]
print("\nVector %s Labeled: %s Model prediction: %s" % (index, y[index], clf.predict(vector)))

sum_up = 0
for i in range(0, clf.support_vectors_.shape[0]):
    sum_up = sum_up + (clf.dual_coef_[0,i] * np.linalg.norm(vector - clf.support_vectors_[i]))
        
print("Decision function: %s" % clf.decision_function(vector))
if sum_up < 0.0:
    print([0])
else:
    print([1])

# <codecell>


# <codecell>


