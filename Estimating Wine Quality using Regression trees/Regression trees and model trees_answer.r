rm(list=ls())#clears all data from your curent Global Environment
setwd("/Users/haha/Dropbox/Leeds/teaching/LUBS5990M\ Machine\ learning\ in\ practice/202223/lecture\ and\ seminar/+S2W18\ Regression\ trees\ and\ model\ trees/R/")


## Example: Estimating Wine Quality ----
## Step 2: Exploring and preparing the data ----
wine <- read.csv("whitewines.csv")

# examine the wine data
str(wine)

# summary statistics of the wine data
summary(wine)

# randomly pick 80% of the samples for training
# and the remainings for testing
smp_size <- floor(0.8 * nrow(wine))
set.seed(987)
train_ind <- sample(nrow(wine), smp_size)
wine_train <- wine[train_ind, ]
wine_test <- wine[-train_ind, ]

## Step 3: Training a model on the data ----
# regression tree using rpart
library(rpart)
# quality is one col name in wine_train
# . means all other cols (other than quality)
m.rpart <- rpart(quality ~ ., data = wine_train) 

# get detailed information about the tree
summary(m.rpart)

# use the rpart.plot package to create a visualization
# install.packages('rpart.plot')
library(rpart.plot)

# a basic decision tree diagram
rpart.plot(m.rpart, digits = 3)

# a few adjustments to the diagram
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)

## Step 4: Evaluate model performance ----

# generate predictions for the testing dataset
p.rpart <- predict(m.rpart, wine_test)

library(Metrics)
# mean absolute error between predicted and actual values
mae(p.rpart, wine_test$quality)
# root mean squared error between predicted and actual values
rmse(p.rpart, wine_test$quality)

# mean(wine_train$quality) = 5.89 - this can be viewed as baseline
# that is, no matter the wine characteristics, 
# the model will always give mean quality value as predicted value
baseline <- mean(wine_train$quality) 

# mean absolute error between actual values and baseline value
# if we simply use the mean value (baseline, 5.89) from training set as predicted value for every wine
# the error will be aoubt 0.65(mae) and 0.86(rmse) - so our 0.60(mae) and 0.75(rmse) error by regression tree is better
mae(baseline, wine_test$quality) #=0.65
rmse(baseline, wine_test$quality) #=0.86

## Step 5: Improving model performance ----
# train a Cubist Model Tree
library(Cubist)
library(tidyverse)
m.cubist<- cubist(x = select(wine_train, -quality), y = wine_train$quality)


# display the tree itself
summary(m.cubist)

# generate predictions for the model
p.cubist <- predict(m.cubist, select(wine_test, -quality))


# mean absolute error of predicted and true values
mae(wine_test$quality, p.cubist)
# root mean squared error between predicted and actual values
rmse(wine_test$quality, p.cubist)



## Question 1 and 2: using random forest ----
# using random forest for regression
# you can also use random forest for classification
# please see the document of the randoForest package at:
# https://cran.r-project.org/web/packages/randomForest/randomForest.pdf

#install.packages("randomForest")
library(randomForest)
m.rf <- randomForest(quality ~ ., data=wine_train, ntree=20)
p.rf <- predict(m.rf, wine_test)


mae(wine_test$quality, p.rf)
rmse(wine_test$quality, p.rf)
