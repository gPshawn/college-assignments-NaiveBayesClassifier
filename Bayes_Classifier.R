# Naive Bayes Classifier algorithm by Guilherme Passos
# Federal University of Minas Gerais

###############################

#    This is an alternative implementation of Naive Bayes Classifier in R. Even though there are many implementations
# of this algorithm on internet and in most of programming languages (R has its own as well), I decided to develop
# my onw using the classical Iris data set in order to get a strong understading of this method. Our goal here is 
# to create an algorithm that is able to sepaate the class SETOSA from the othrer two ones. For a previous 
# understanding of Naive Bayes Classifier, check this following lecture here: 
# http://software.ucv.ro/~cmihaescu/ro/teaching/AIR/docs/Lab4-NaiveBayes.pdf
# p.s: This is NOT a general implementation of Naive Bayes Classifier

###############################

# Loading the Iris data set.
data(iris)

# Separating the two classes we're gonna working on
setosa <- iris[which(iris$Species=="setosa"),1:4]
others <- iris[which(iris$Species!="setosa"),1:4]
setosa$label <-1
others$label <-2

# -----------------------------------

# Creating a train and a test set. (70% of samples go to the train set, and the 30% remaining go to the test set).
library(RSNNS)
setosa <- splitForTrainingAndTest(setosa[,1:4], setosa [,5], ratio=0.3)
others <- splitForTrainingAndTest(others[,1:4], others [,5], ratio=0.3)



# -----------------------------------

# Generating model which contains the mean and covariance matrixes from the training set we've just created. 
# These matrixes will be used to solve our classification problem.

GetModel <- function (c1,c2) {
  c1_meanAttribute <- apply(c1,2, mean)
  c2_meanAttribute <- apply(c2,2, mean)
  c1_covAttribute <- cov(c1)
  c2_covAttribute <- cov(c2)
  model <- list(meanC1 = c1_meanAttribute,meanC2 = c2_meanAttribute, 
              covC1 = c1_covAttribute,covC2 = c2_covAttribute)

  return (model)
}

# -----------------------------------

# The function pdfnvar below gives the probability density of all features in a set.
pdfnvar <- function(x, m, K, n) ((1/(sqrt((2*pi)^n*(det(K)))))*exp(-0.5*(t(x-m) %*% (solve(K)) %*% (x-m))))
pdfnvarvec <- function(x, m, K, n) apply(x, 1, pdfnvar, m, K, n)


# -----------------------------------

# Combaning and mixing the two training sets of each class. 
setosa_train <- cbind(setosa$inputsTrain,setosa$targetsTrain)
others_train <- cbind(others$inputsTrain,others$targetsTrain)
trainingSet <- rbind(setosa_train,outras_train)
trainingSet <- trainingSet[sample(nrow(trainingSet),replace=F,size=nrow(trainingSet)),]



# -----------------------------------

# In order to validate our model, let's figure out the posterior probability of each sample in our 
# traning set to belong to the two classes we're considering here. 
PosteriorProbs <- function (dataSet,model, c1, c2) {
  
  c1_likelihood <- pdfnvarvec(dataSet,model$meanC1,model$covC1,dim(c1)[2])
  c2_likelihood <- pdfnvarvec(dataSet,model$meanC2,model$covC2,dim(c2)[2])
  c1_PriorProb <- length(c1)/(length(c1)+length(c2))
  c2_PriorProb <- length(c2)/(length(c1)+length(c2))
  c1_PosteriorProb <- as.array(c1_PriorProb*c1_likelihood)
  c2_PosteriorProb <- as.array(c2_PriorProb*c2_likelihood)
  
  probabilities <- list(c1PosteriorProb =  c1_PosteriorProb, c2PosteriorProb = c2_PosteriorProb) 
  
  return (probabilities)
}

model <- GetModel(setosa$inputsTrain,others$inputsTrain)

probs<- PosteriorProbs(trainingSet[,1:4],model,setosa$inputsTrain,others$inputsTrain)


# -----------------------------------

# Now that we got the posterior probabilities of all samples to belong to one of our two classes, let's implement a 
# simple code that will basically check what is the higher posterior probability between the two classes for each
# sample.
NaiveBayesClassifier <-function (posteriorProbC1,posteriorProbC2) {

  predictions <- array(nrow(posteriorProbC1))

  for (i in 1:nrow(posteriorProbC1)){
    if (posteriorProbC1[i]>posteriorProbC2[i]) predictions[i]<-1
    else predictions [i]<-2
  }
  
  return (predictions)
}

yModelPred <- NaiveBayesClassifier(probs$c1PosteriorProb,probs$c2PosteriorProb)



# -----------------------------------

# The following confusion matrix shows how many sample in the traning set our model got right.  
yReal <-trainingSet[,5]
table(yReal,yModelPred)



# -----------------------------------

# Now we're going to check how our model will perform with the testing set. It means thar our model will predict the 
# class of samples it has never seen before. First, we're going to combine and mix our testing set.
setosa_test<- cbind(setosa$inputsTest,setosa$targetsTest)
others_test <- cbind(others$inputsTest,others$targetsTest)
testingSet<- rbind(setosa_test,others_test)
# mixing samples
testingSet <- testingSet[sample(nrow(testingSet),replace=F,size=nrow(testingSet)),]
# calculating probabilities
probs_test<- PosteriorProbs(testingSet[,1:4],model,setosa$inputsTrain,others$inputsTrain)
# getting predictions from model
yPred <- NaiveBayesClassifier(probs_test$c1PosteriorProb,probs_test$c2PosteriorProb)
yReal <- yReal <-testingSet[,5]
# confusion Matrix
table(yReal,yPred)

# -----------------------------------

# Now we're going to test the bayer classifier we've just created 30 times with differents training and testing sets
# of the Iris data set. Our goal now is to check accurancy of our method. 

acc <- vector()
hits <- vector()
data(iris)

for (i in 1:30) {
  
  # Separeting the two classes we're gonna working on
  setosa <- iris[which(iris$Species=="setosa"),1:4]
  others <- iris[which(iris$Species!="setosa"),1:4]
  setosa$label <-1
  others$label <-2
  
  
  # spliting sample into training and testing set
  setosa <- splitForTrainingAndTest(setosa[,1:4], setosa [,5], ratio=0.3)
  others <- splitForTrainingAndTest(others[,1:4], others [,5], ratio=0.3)
  # combing and mixing the two classes
  setosa_test<- cbind(setosa$inputsTest,setosa$targetsTest)
  others_test <- cbind(others$inputsTest,others$targetsTest)
  testingSet<- rbind(setosa_test,others_test)
  testingSet <- testingSet[sample(nrow(testingSet),replace=F,size=nrow(testingSet)),]
  # generating a model
  model <- GetModel(setosa$inputsTrain,others$inputsTrain)
  # getting probabilities
  probs_test<- PosteriorProbs(testingSet[,1:4],model,setosa$inputsTrain,others$inputsTrain)
  # predictions
  yPred <- NaiveBayesClassifier(probs_test$c1PosteriorProb,probs_test$c2PosteriorProb)
  yReal <- yReal <-testingSet[,5]
  # confusion Matrix
  cm <- table(yReal,yPred)
  # accuracy 
  acc[i] <-  sum(diag(cm))/sum(cm)
  hits[i]<- sum(diag(cm)) 

}
# average accuracy
mean(acc)
# accuracy standard deviation
sd(acc)

# average hits
mean(hits)
# hits standard deviation
sd(hits)
  
  
