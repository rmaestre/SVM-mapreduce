library(rmr2)

map = function(k, vectors){
    # Crete DS to save keys and values
    return.keys <- return.vals <- NULL
    # For each vector to map
    for(i in 1:nrow(vectors)) {
        # Get vector from Vectors input
        vector <- vectors[i,]
        # For each vector calculate distance and apply dual coef
        # Smapling is allowed (further work)
        #index <- sample(nrow(support_vectors_), 1306, replace=FALSE)
        for (s in 1:nrow(support_vectors_)) {
            # Add key and value
            return.keys <- c(return.keys, toString(vector))
            return.vals <- c(return.vals, dual_coef_[s] * norm(as.matrix(vector-support_vectors_[s,])))
        }
    # Debug info
    rmr.str(i)
    }
    # Return all keys and values
    return(keyval(return.keys, return.vals))
}

reduce = function(identifier, values){
    # Summarize all values from each vector
    summ <- sum(unlist(values))
    # Return value (v>0 or v<0)
    return(keyval(identifier, summ))
}

svm = function(input, output){
                mapreduce(  input=input, 
                            input.format=make.input.format("csv", sep = "\t"), 
                            output=output, 
                            output.format=make.output.format("csv", sep=","),
                            map=map, 
                            reduce=reduce,
                            combine=FALSE
                        ) 
                    }

# Shared variables
support_vectors_ = as.data.frame(from.dfs("y1_model_supported_vectors.tsv", format = make.input.format("csv", sep = "\t"))["val"]) 
dual_coef_ = as.data.frame(from.dfs("y1_model_dual_coef.tsv", format = make.input.format("csv", sep = "\t"))["val"]) 

# Input/output datafiles
hdfs.input_file = "y1_model_X_test_sample.tsv"
hdfs.output = "out"

# Execute map/reduce
out = svm(hdfs.input_file, hdfs.output)


# Get results
results = from.dfs(out)
results.df = as.data.frame(results)