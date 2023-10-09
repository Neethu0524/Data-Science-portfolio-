rm(list=ls())#clears all data from your curent Global Environment
# set this to your own path!!!
setwd("/Users/haha/Dropbox/Leeds/teaching/LUBS5990M Machine learning in practice/202223/lecture and seminar/+S2W16 Naïve bayes/R/")


## Example: Filtering spam SMS messages ----
## Step 1 and 2: Exploring and preparing the data ---- 

# read the sms data into the sms data frame
# set up the parameter encoding = 'UTF-8' if you use non-English OS computer
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE, encoding = 'UTF-8')

# examine the structure of the sms data
str(sms_raw)

# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable more carefully
str(sms_raw$type)
table(sms_raw$type)

# build a corpus using the text mining (tm) package
#install.packages('tm',dependencies = TRUE)
library(tm)
#VCorpus(): volatile corpus - stored in memory
#PCorpus(): stored on disk - access a permanent corpus stored in a database
#VectorSource(): source is from a text vector
#DirSource(): source is from a file in a directory
#DataframeSource(): source is from a dataFrame
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# examine the sms corpus
print(sms_corpus)
inspect(sms_corpus[1:2]) # tm corpus is essentially a list

as.character(sms_corpus[[1]]) # double-bracket notation is required
lapply(sms_corpus[1:2], as.character)

# clean up the corpus using tm_map()
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# show the difference between sms_corpus and corpus_clean
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # remove numbers
# here stopwords() is just 'a list of stop words'
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # remove punctuation

# illustration of word stemming
#install.packages('SnowballC')
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns")) # return root form

sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

# examine the final clean corpus
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

## tokenisation
# create a document-term sparse matrix
# each row: a message (a sample)
# each column: a word (a feature)
# TermDocumentMatrix(): each row is a word and each column is a message
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
class(sms_dtm)
inspect(sms_dtm[1:5, 1:10])


# creating training and test datasets
# sms_dtm_train <- sms_dtm[1:4169, ] # this is what we did in last seminar
# sms_dtm_test  <- sms_dtm[4170:5559, ]
## also save the labels
# sms_train_labels <- sms_raw[1:4169, ]$type
# sms_test_labels  <- sms_raw[4170:5559, ]$type

# this time I will introduce using an index variable
# in the following sessions I will introduce better methods 
# such as random sampling and cross-validation
# use about 75% for training and remaining for testing
ratio = 0.75
p_index = round(nrow(sms_dtm)*ratio)
sms_dtm_train <- sms_dtm[1:p_index, ]
sms_dtm_test  <- sms_dtm[(p_index+1):nrow(sms_dtm), ]
sms_train_labels <- sms_raw[1:p_index, ]$type
sms_test_labels  <- sms_raw[(p_index+1):nrow(sms_raw), ]$type


# indicator features for frequent words
findFreqTerms(sms_dtm_train, 5)

# save frequently-appearing terms to a character vector
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

# create DTMs with only the frequent terms
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# naive bayes clssifier is trained on categorical data
# convert counts to categorical variable
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
# MARGIN=2 means apply function on column; MARGIN=1 means row
# search lapply() we used in last session and compare them
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

## Step 3: Training a model on the data ----
#install.packages('e1071')
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

## Step 4: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)

#install.packages('gmodels')
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance ----
sms_classifier_laplace <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred_laplace <- predict(sms_classifier_laplace, sms_test)
CrossTable(sms_test_pred_laplace, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
