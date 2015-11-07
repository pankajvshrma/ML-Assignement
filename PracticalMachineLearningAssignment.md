---
title: "Practical Machine Learning Assignment"
author: "Pankaj Sharma"
date: "7 November 2015"
output: html_document
---
This is an assignment of coursera Pratical Machine Learning Course.Here we are using Random Forests for modelling the data.We also set seed to make program reproductible.We use library doParallel apart from caret already shown in course.We use library pROC for plotting the ROC curve.We use package randomForest for random forests.



```r
options(warn=-1)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Note: the specification for S3 class "family" in package 'MatrixModels' seems equivalent to one from package 'lme4': not turning on duplicate class definitions for this class.
```

```r
library(doParallel)
```

```
## Loading required package: foreach
## foreach: simple, scalable parallel programming from Revolution Analytics
## Use Revolution R for scalability, fault tolerance and more.
## http://www.revolutionanalytics.com
## Loading required package: iterators
## Loading required package: parallel
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
## 
## Attaching package: 'pROC'
## 
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

First Step is to read CSV training file


```r
set.seed(123)

training<-read.csv("D:/pml-training.csv",header=TRUE)
```
Seperating class feature from training


```r
X.train<-training

Y.train =training[,length(training)]
```
Checking for NA's in data and filling them with mean if feature is numeric
Forcing Features which are factors as numeric and replacing #DIV/0! as factor 2


```r
for(i in 1:ncol(X.train))
{

if (class(X.train[,i])== 'integer' | class(X.train[,i])== 'numeric' )
  {
meanXi=mean(X.train[,i],na.rm=TRUE)
   for(j in 1:nrow(X.train))
     {
      if(is.na(X.train[j,i]))
        X.train[j,i]= meanXi
         
       }
}
if(class(X.train[,i])== 'factor')
  X.train[,i]=as.numeric(X.train[,i])

}
```
Now converting back the last column of class as factor again

```r
X.train[,ncol(X.train)]<-as.factor(X.train[,ncol(X.train)])
```
Removing columns which dont have complete observations

```r
featuresnames <- colnames(X.train[colSums(is.na(X.train)) == 0])[-(1:7)]
features <- X.train[featuresnames]
```
Dividing dataset into 70:30 and doing cross validation

```r
xdata <- createDataPartition(y=features$classe, p=3/4, list=FALSE )
training <- features[xdata,]
testing <- features[-xdata,]
```
Registering with doParallel library and making a cluster of 3 cores

```r
require(doParallel)

registerDoParallel(cores=detectCores(all.tests=TRUE))

cores <- detectCores()
cl <- makePSOCKcluster(3)
registerDoParallel(cl)
```
Fitting random Forests in parallel with a total of 600 trees

```r
fit.rf <- foreach(ntree=rep(200, 3), .combine=combine, 
                  .packages="randomForest") %dopar% {
  randomForest(training[,-ncol(training)], training$classe, importance = TRUE, ntree = ntree)
}
```
Getting importance of top five variables based on decreasing gini index  and stopping the cluster

```r
head(importance(fit.rf, type=2))
```

```
##                     MeanDecreaseGini
## roll_belt                711.7033287
## pitch_belt               439.8896870
## yaw_belt                 586.1256912
## total_accel_belt         193.3144372
## kurtosis_roll_belt         0.2628208
## kurtosis_picth_belt        0.2221048
```

```r
stopCluster(cl)
```
Printing confusion matrix with the training data.Insample error is too low.

```r
rf.pred <- predict(fit.rf,training, type="class")
rf.perf <- table(training$classe,rf.pred ,dnn=c("Actual", "Predicted"))
confusionMatrix(rf.perf)
```

```
## Confusion Matrix and Statistics
## 
##       Predicted
## Actual    1    2    3    4    5
##      1 4185    0    0    0    0
##      2    0 2848    0    0    0
##      3    0    0 2567    0    0
##      4    0    0    0 2412    0
##      5    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
Printing confusion matrix with the test data.Out of sample error is too low as Accuracy achieved was 99.55%

```r
rf.pred <- predict(fit.rf,testing, type="class")
rf.perf <- table(testing$classe,rf.pred ,dnn=c("Actual", "Predicted"))
rf.perf
```

```
##       Predicted
## Actual    1    2    3    4    5
##      1 1394    1    0    0    0
##      2    1  946    2    0    0
##      3    0    8  847    0    0
##      4    0    0   12  790    2
##      5    0    0    0    1  900
```

```r
confusionMatrix(rf.perf)
```

```
## Confusion Matrix and Statistics
## 
##       Predicted
## Actual    1    2    3    4    5
##      1 1394    1    0    0    0
##      2    1  946    2    0    0
##      3    0    8  847    0    0
##      4    0    0   12  790    2
##      5    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9945         
##                  95% CI : (0.992, 0.9964)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.993          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            0.9993   0.9906   0.9837   0.9987   0.9978
## Specificity            0.9997   0.9992   0.9980   0.9966   0.9998
## Pos Pred Value         0.9993   0.9968   0.9906   0.9826   0.9989
## Neg Pred Value         0.9997   0.9977   0.9965   0.9998   0.9995
## Prevalence             0.2845   0.1947   0.1756   0.1613   0.1839
## Detection Rate         0.2843   0.1929   0.1727   0.1611   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9995   0.9949   0.9909   0.9977   0.9988
```
Plotting the ROC curve with the sensitivity and specificity thresholds.It showed that area under curve was very close to best prediction.

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png) 

```
## 
## Call:
## plot.roc.default(x = testing$classe, predictor = rf.pred, ci = TRUE,     of = "thresholds", type = "shape", col = "blue")
## 
## Data: rf.pred in 1395 controls (testing$classe 1) < 949 cases (testing$classe 2).
## Area under the curve: 0.9991
## 95% CI (2000 stratified bootstrap replicates):
##  thresholds sp.low sp.median sp.high se.low se.median  se.high
##        -Inf 0.0000    0.0000       0 1.0000  1.000000 1.000000
##         1.5 0.9978    0.9993       1 0.9968  0.998900 1.000000
##         2.5 1.0000    1.0000       1 0.0000  0.002107 0.005269
```
Thankyou all,It was fun doing this assignment.
