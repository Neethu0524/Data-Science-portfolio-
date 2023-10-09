rm(list=ls())#clear all data from your curent Global Environment

#Diagnosing breast cancer with the kNN algorithm
#Data source: UCI Machine Learning Repository
#http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# read the background of this dataset first
#----------------------------------------------------------
#----------------------------------------------------------

#1. Exploring, preparing and transforming the data

#Read the data
wbcd <- read.csv("/Users/haha/Dropbox/Leeds/teaching/LUBS5990M\ Machine\ learning\ in\ practice/202223/lecture\ and\ seminar/+S2W15KNN/R/wisc_bc_data.csv", 
                 stringsAsFactors = FALSE, encoding = 'UTF-8')
# read.csv will convert string into factors by default, 
# so we set stringsAsFactors = FALSE here first to let you see the original raw data
# use encoding = 'UTF-8' if your OS language is not English

str(wbcd) #check the data

#Regardless of the machine learning method, 
#ID variables should always be excluded. 
#Neglecting to do so can lead to erroneous
#findings because the ID can be used to uniquely "predict" each example. 
#Therefore, a model that includes an identifier 
#will most likely suffer from overfitting, 
#and is not likely to generalize well to other data.
#Let's drop the id feature altogether.

wbcd <- wbcd[-1]

#The next variable, diagnosis, is of particular interest, 
#as it is the outcome we hope to predict. 
#This feature indicates whether the example is from a benign or malignant mass. 

table(wbcd$diagnosis)

# B for Benign, means normal samples
# M for Malignant, means abnormal samples (may have cancer)

#Many R machine learning classifiers require 
#that the target feature is coded as a factor, 
#so we will need to recode the diagnosis variable.

wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), 
                         labels = c("Benign", "Malignant"))


#Now, when we look at the prop.table() output, 
#we notice that the values 
#have been labeled Benign and Malignant, 
#with 62.7 percent and 37.3 percent of the masses, respectively:

round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

#The remaining 30 features are all numeric,
#and consist of three different measurements of ten characteristics. 
#For illustrative purposes, 
#we will only take a closer look at three of the features:

summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

#Normalizing numeric values
#We apply normalization to rescale the features to a standard range of values.
#To normalize these features, we need to create a normalize() function in R. 
#This function takes a vector x of numeric values, 
#and for each value in x, subtract the minimum value 
#in x and divide by the range of values in x.

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#Apply the normalize() function to the numeric features
#using lapply() function (and convert back to data frame)
# using index like this [2:31] looks a bit urgly, we will improve this in follwing weeks
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
summary(wbcd_n$area_mean)

#----------------------------------------------------------
#2 Training a model on the data 
#2A. Creating training and test datasets

#The question we ask is how well our learner performs on a dataset of unlabeled data. 
#If we had access to a laboratory, we could apply our learner to measurements 
#taken from the next 100 masses of unknown cancer status 
#and see how well the machine learner's predictions 
#compare to diagnoses obtained using conventional methods.
#In the absence of such data, we can simulate this scenario 
#by dividing our data into two portions: 
#a training dataset that will be used to build the kNN model 
#and a test dataset that will be used to estimate the predictive accuracy of the model.
# (this is the simplest way to divide the dataset
# in later sessions we will inroduce better methods to do that)

# again, using index like this is urgly, we will improve this in follwing weeks
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

#When we constructed our training and test data, 
#we excluded the target variable, diagnosis. 
#For training the kNN model, we will need to store these 
#class labels in factor vectors, divided to the training and test datasets:

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

#----------------------------------------------------------

#2B. Training a model on the data 

library(class)#a set of basic R functions for classification

#We use knn() function to perform classification
#We split our data into training and test datasets, 
#each with exactly the same numeric features. 
#The labels for the training data are stored 
#in a separate factor vector. 
#The only remaining parameter is k, 
#which specifies the number of neighbors to include in the vote.

K = 21
# training size = 469 so we try its square root 21 as the value of k first
#Using an odd number of K will reduce the chance of ending with a tie vote.
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k=K)


#----------------------------------------------------------

#3. Evaluating model performance
#install.packages("gmodels")
library(gmodels)#various tools for model fitting

#Evaluate how well the predicted classes in the wbcd_test_pred 
#vector match up with the known values in the wbcd_test_labels vector
#Specifying prop.chisq=FALSE will remove the chi-square values that are not needed
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq=FALSE)




