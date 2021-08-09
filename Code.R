
### HR Analytics: Job Change of Data Scientists ####
####import the libraries
library(ggplot2)
library(caTools)
library(caret)
library(e1071)



### set working directory
getwd()
setwd("C:/Users/Priya Patle/Desktop/Class/R/Research Projects/Logistic Regression/Logistic Regression")
getwd()


### Select a dataset
train <- read.csv("aug_train.csv")
dim(train)  
str(train)


### remove the insignificant columns
unique(train$last_new_job)
train <- train[,-c(1,2,9,10,12)]
str(train)



###Check for missing values
colSums(is.na(train))

### Replace the missing values
unique(train$gender)
train$gender <- ifelse(train$gender=="","Other",train$gender)
unique(train$relevent_experience)
unique(train$enrolled_university)
train$enrolled_university <- ifelse(train$enrolled_university=="","Not Specified",train$enrolled_university)
unique(train$education_level)
train$education_level <- ifelse(train$education_level=="","Not Specified",train$education_level)
unique(train$major_discipline)
train$major_discipline <- ifelse(train$major_discipline=="","Other",train$major_discipline)
unique(train$company_type)
train$company_type <- ifelse(train$company_type=="","Other",train$company_type)


### Bivariate data visualisation #######
plot(train$target,train$city_development_index,type = "p", xlab="target",ylab = "City Devlopment Index")
ggplot(train,aes(x = target, fill = gender))+geom_bar(position = "stack")
ggplot(train,aes(x = target, fill = relevent_experience))+geom_bar(position = "stack")
ggplot(train,aes(x = target, fill = enrolled_university))+geom_bar(position = "stack")
ggplot(train,aes(x = target, fill = education_level))+geom_bar(position = "stack")
ggplot(train,aes(x = target, fill = major_discipline))+geom_bar(position = "stack")
plot(train$target,train$company_type,type = "p", xlab="target",ylab = "Company Type")
plot(train$target,train$training_hours,type = "p", xlab="target",ylab = "Training Hours")


### encoding
str(train)
train$gender <- factor(train$gender)
unique(train$gender)
train$gender <- as.numeric(train$gender)

train$relevent_experience <- factor(train$relevent_experience)
unique(train$relevent_experience)
train$relevent_experience <- as.numeric(train$relevent_experience)

train$enrolled_university <- factor(train$enrolled_university)
unique(train$enrolled_university)
train$enrolled_university <- as.numeric(train$enrolled_university)

train$education_level <- factor(train$education_level)
unique(train$education_level)
train$education_level <- as.numeric(train$education_level)

train$major_discipline <- factor(train$major_discipline)
unique(train$major_discipline)
train$major_discipline <- as.numeric(train$major_discipline)

train$company_type <- factor(train$company_type)
train$company_type <- as.numeric(train$company_type)
str(train)
### encoding is complete


#####Univariate analysis
hist(train$city_development_index,main = "Histogram of City Development Index",xlab ="City Development Index" )
hist(train$gender,main = "Histogram of Gender",xlab ="Gender")
hist(train$relevent_experience,main = "Histogram of Relevent Experience",xlab ="Relevent Experience")
hist(train$enrolled_university,main = "Histogram of University",xlab ="University")
hist(train$education_level,main = "Histogram of Education Level",xlab ="Education Level")
hist(train$major_discipline,main = "Histogram of Major",xlab ="Major")
hist(train$company_type,main = "Histogram of Company Type",xlab ="Company Type")
hist(train$training_hours,main = "Histogram of Training Hours",xlab ="Training Hours")
hist(train$target,main = "Histogram of Target",xlab ="Target")




### split the data 

split <- sample.split(train$target, SplitRatio = 0.75)
split

table(split)

training <- subset(train, split==TRUE)
test <- subset(train, split==FALSE)
nrow(training)
nrow(test)


#### now build the model

log_job <- glm(target~., data=training,family = 'binomial')
log_job
summary(log_job)



### Removing gender and major discipline as they have high p value
log_job_new <- glm(target~.-gender-major_discipline, data=training,family = 'binomial')
log_job_new
summary(log_job_new)


###predict the model

log_job_pred <- predict(log_job_new,newdata = test, type='response')
log_job_pred



### compare acutal to predicted

log_job_cbind <- cbind(test$target, log_job_pred)
head(log_job_cbind)


### 50 %###
log_job_pred_50 <- ifelse(log_job_pred>=0.5,1,0)
log_job_pred_50
log_job_cbind <- cbind(test$target, log_job_pred_50)
head(log_job_cbind,10)
## validate the data using confusion matrix or check accuracy
cm <- table(test$target, log_job_pred_50)
cm
confusionMatrix(cm)


### At 60 %  ###
log_job_pred_60 <- ifelse(log_job_pred>=0.6,1,0)
cm1 <- table(test$target, log_job_pred_60)
confusionMatrix(cm1)



### at 70 %
log_job_pred_70 <- ifelse(log_job_pred>=0.7,1,0)
cm2 <- table(test$target, log_job_pred_70)
confusionMatrix(cm2)



### at 40 %
log_job_pred_40 <- ifelse(log_job_pred>=0.4,1,0)
cm3 <- table(test$target, log_job_pred_40)
confusionMatrix(cm3)

### 40 % has the highest accuracy ###

####### checking the performance on aug_test.csv #################
ext_test <- read.csv("aug_test.csv")
log_job_cbind <- cbind(ext_test$target, log_job_pred)

head(log_job_cbind)
log_ext_50 <- ifelse(log_job_cbind>=0.5,1,0)

cm <- table(ext_test$target, log_ext_50)
cm
confusionMatrix(cm)



### ROC curve
library(ROCR)
ROCR_PRED <- prediction(log_job_pred_40, test$target)
ROCR_PRED
nrow(test)
ROCR_PRED_PERFORMANCE <- performance(ROCR_PRED, 'tpr', 'fpr')
ROCR_PRED_PERFORMANCE


### Do the plotting for visualization

plot(ROCR_PRED_PERFORMANCE)


plot(ROCR_PRED_PERFORMANCE,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))
abline(a=0, b=1)
