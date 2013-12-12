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
Training SVM for y1 label

K-Fold validation:
X train: (30000, 16)
y train: (30000,)
X test: (20000, 16)
y test: (20000,)

Supported vectors length: (1306, 16)
Dual coef. length: (1, 1306)

Score k-fold validation: 1.000000

Vector 1 Labeled: 1.0 Model prediction: [ 1.]
Decision function: [[ 1.43043731]]
[1]
```

Model for y2 var
```
Training SVM for y2 label

K-Fold validation:
X train: (30000, 16)
y train: (30000,)
X test: (20000, 16)
y test: (20000,)

Supported vectors length: (5916, 16)
Dual coef. length: (1, 5916)

Score k-fold validation: 0.914650

Vector 1 Labeled: 0.0 Model prediction: [ 0.]
Decision function: [[-1.0370797]]
[0]
```


Model for y3 var
```
Training SVM for y3 label

K-Fold validation:
X train: (30000, 16)
y train: (30000,)
X test: (20000, 16)
y test: (20000,)

Supported vectors length: (3762, 16)
Dual coef. length: (1, 3762)

Score k-fold validation: 0.948650

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
    value <- value + dual_coef_[i] * sqrt(sum((vector - support_vectors_[i,]) ^ 2))
}
```
