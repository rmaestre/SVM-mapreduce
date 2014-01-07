setwd("/Users/rmaestre/Projects/SVM-mapreduce/")

# Class to test
class <- "y1"

# Data to test the SVM model choosen through k-fold (40%)
X_data <- read.table(file.path(getwd(),'data/svm_models',sprintf("%s%s", class,'_model_x_test.tsv')), header=FALSE, sep="\t")
y_data <- read.table(file.path(getwd(),'data/svm_models',sprintf("%s%s", class,'_model_y_test.tsv')), header=FALSE, sep="\t")

# Data to calculate the class of a select vector
support_vectors_ <- read.table(file.path(getwd(),'data/svm_models',sprintf("%s%s", class,'_model_supported_vectors.tsv')), header=FALSE, sep="\t")
dual_coef_ <- read.table(file.path(getwd(),'data/svm_models',sprintf("%s%s", class,'_model_dual_coef.tsv')), header=FALSE, sep="\t")

# Get a sample vector to test it
vector_index <- 195
vector <- X_data[vector_index,]

# Get value from decision function (from support vector and dual coef)
value <- 0
for (i in 2:dim(support_vectors_)[1]) { 
    value <- value + dual_coef_[i] * sqrt(sum((vector - support_vectors_[i,]) ^ 2))
}

# Print decision function value
print(value)

# Print labeled class for y variable
print(y_data[vector_index])