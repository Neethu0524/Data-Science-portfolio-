#----------------------------------------------------------
#Identifying risky bank loans using C5.0 decision trees
#----------------------------------------------------------
#The credit dataset includes 1,000 examples of loans, 
#plus a combination of numeric and nominal features 
#indicating characteristics of the loan and the loan applicant. 
#A class variable indicates whether the loan went into default.
#Since the loan data was obtained some time ago from Germany, 
#the currency is recorded in Deutsche Marks (DM). 
#----------------------------------------------------------
#1A. Exploring, preparing and transforming the data
#----------------------------------------------------------

rm(list=ls())#clears all data from your curent Global Environment
setwd("C:/5990/")

credit <- read.csv("dataset_credit.csv", encoding = 'UTF-8')

str(credit)
summary(credit)


#Let's take a look at some of the table() output 
#for a couple of features of loans that seem 
#likely to predict a default. 

table(credit$checking_balance)
table(credit$savings_balance)

#The checking_balance and savings_balance features 
#indicate the applicant's checking and savings account balance, 
#and are recorded as categorical variables.
#It seems like a safe assumption that larger checking 
#and savings account balances should be related 
#to a reduced chance of loan default.

#Some of the loan's features are numeric,
#such as its term (months_loan_duration), 
#and the amount of credit requested (amount).

summary(credit$months_loan_duration)
summary(credit$amount)

#The loan amounts ranged from 250 DM to 18,420 DM 
#across terms of 4 to 72 months, 
#with a median duration of 18 months and amount of 2,320 DM.

#The 'default' variable indicates whether 
#the loan applicant was unable 
#to meet the agreed payment terms and went into default. 
#We also change numeric to factor (1=no, 2=yes)
#as we need factor for our C5.0 training 
credit$default <- factor(credit$default, levels = c(1, 2), 
                         labels = c("no", "yes"))
table(credit$default)
#A total of 30 percent of the loans went into default
#-----------------------------------
#1. Sampling data for training model
#-----------------------------------
#We will use 90 percent of the data for training 
#and 10 percent for testing, 
#which will provide us with 100 records to simulate new applicants.
#We need to randomly order our credit data frame prior to splitting
#to avoid order bias.

#Now we can split into training (90 percent or 900 records), 
#and test data (10 percent or 100 records) 
smp_size <- floor(0.9 * nrow(credit))

# try the sample() function first
# each time it generates a list of DIFFERENT random numbers
sample(10, 5)
sample(10, 5)

#The set.seed() function is used to generate random numbers 
#in a predefined sequence, 
#starting from a position known as a seed 
#(set here to the arbitrary value 12345). 
#The set.seed() function ensures 
#that if the analysis is repeated, an identical result is obtained.
set.seed(12345)
sample(10, 5)

set.seed(12345)
sample(10, 5)

# set the SAME seed (any constant value, as long as the same value each time) every time before running a rondom number generator function
set.seed(12345)
train_ind <- sample(nrow(credit), smp_size)
credit_train <- credit[train_ind, ]
credit_test <- credit[-train_ind, ]

#--------------------------------------------------
#2. Training a model on the data: basic 
#--------------------------------------------------
#We will use the C5.0 algorithm in the 
#C50 package for training our decision tree model.

#install.packages("C50")
library(C50)
library(tidyverse)
#The 17th column (default) in credit_train is the class variable, default, 
#so we need to exclude it from the training data frame as an independent variable, 
#but supply it as the target factor vector for classification
credit_model <- C5.0(select(credit_train, -default), credit_train$default)
credit_model
summary(credit_model)




#----------------------------------------------------------
#3. Evaluating model performance: basic
#----------------------------------------------------------
#To apply our decision tree to the test dataset, 
#we use the predict() function 
credit_pred <- predict(credit_model, credit_test)

#This creates a vector of predicted class values, 
#which we can compare to the actual class values 
#using the CrossTable() function in the gmodels package.
#Setting the prop.c and prop.r parameters to FALSE 
#removes the column and row percentages from the table. 
#The remaining percentage (prop.t) 
#indicates the proportion of records in the cell out of the total number of records.
library(gmodels)
# be careful when the order of credit_pred and credit_test$default, and the order of the dnn labels
CrossTable(credit_pred, credit_test$default, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted default', 'actual default' ))


#-----------------------------------------------------------------------
#4.Improving model performance: Adaptive boosting
#-----------------------------------------------------------------------
#The C5.0() function makes it easy to add boosting to our C5.0 decision tree. 
#We simply need to add an additional trials parameter 
#indicating the number of separate decision trees 
#to use in the boosted team. The trials parameter sets an upper limit; 
#the algorithm will stop adding trees if it recognizes 
#that additional trials do not seem to be improving the accuracy.

credit_boost10 <- C5.0(select(credit_train, -default), credit_train$default, trials = 10)
credit_boost10
summary(credit_boost10)


credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_boost_pred10, credit_test$default, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted default', 'actual default'))


