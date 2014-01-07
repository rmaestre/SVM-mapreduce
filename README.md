SVM-mapreduce
=============

A map reduce approach to SVM classification. http://r-es.org/Concursos+V+Jornadas


Brief description of Training50K.csv (sep=\t) 
  * 8 continuous vars (x1..x8), 
  * 7 categorics vars (xc1..xc7)
  * 3 predicted var yes/no (y1..y3) (sorted by complexity, y1 > y2, ...)


SVM training - support vectors
=============

Model for y1 var
```
Training SVM for y1 label (gamma=0.001, C=50.)

K-Fold validation:
X train: (30000, 16)
y train: (30000,)
X test: (20000, 16)
y test: (20000,)

Supported vectors length: (124, 16)
Dual coef. length: (1, 124)

Score k-fold validation: 1.000000

Vector 1 Labeled: 1.0 Model prediction: [ 1.]
Decision function: [[ 1.23194725]]
[1]
```

Model for y2 var
```
Training SVM for y2 label (gamma=0.01, C=30.)

K-Fold validation:
X train: (30000, 16)
y train: (30000,)
X test: (20000, 16)
y test: (20000,)

Supported vectors length: (5819, 16)
Dual coef. length: (1, 5819)

Score k-fold validation: 0.913350

Vector 1 Labeled: 0.0 Model prediction: [ 0.]
Decision function: [[-3.46318592]]
[0]
```


Model for y3 var
```
Training SVM for y3 label (gamma=0.001, C=50.)

K-Fold validation:
X train: (30000, 16)
y train: (30000,)
X test: (20000, 16)
y test: (20000,)


Supported vectors length: (3638, 16)
Dual coef. length: (1, 3638)

Score k-fold validation: 0.949350

Vector 1 Labeled: 0.0 Model prediction: [ 0.]
Decision function: [[-2.67298744]]
[0]
```


R - Iterative approach
=============


[Decision function](http://www.math.unipd.it/~aiolli/corsi/1213/aa/user_guide-0.12-git.pdf) from the scikit-learn package.

```
# Get value from decision function (from support vector and dual coef)
value <- 0
for (i in 2:dim(support_vectors_)[1]) { 
    value <- value + dual_coef_[i] * norm(as.matrix(vector - support_vectors_[i,])))
                     
}
```

R - Mapreduce approach
=============

Using the above decision function, we apply calculate the distance of each vector to each support vector applying the dual coef. In the reduce step, we summarize the previous part from each vector.
