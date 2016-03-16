options(warn=-1)
library(mlr) # ML library
library(OpenML) # Data import and sharing results
library(ggplot2) # Plotting
library(cowplot) # Plot styling
library(rattle) # Plotting trees  
library(kknn)
library(pROC)
library(mlbench)
library(class)
library(KODAMA)
library(caret)

new_data <- as.data.frame(mlbench.threenorm(n=1000, d=2))
n <- dim(new_data)[1]   # nr of observations

# define training and test set, 
# where 1/3 of the data is used as test set
test.ind <- sample(c(1:n), size = round(n/3), replace = FALSE)
new_data.train <- new_data[-test.ind,]
new_data.test <- new_data[test.ind,]

# k nearest neighbor classifier
new_data.kknn <- kknn(classes~., train=new_data.train, test=new_data.test, 
                  k=50, distance = 2, kernel = "rectangular",
                  scale=T)

model <- train(
  classes~., 
  data=new_data, 
  method='knn',
  tuneGrid=expand.grid(.k=1:50),
  metric='Accuracy',
  trControl=trainControl(
    method='boot', 
    number=100, 
    repeats=1))

model
plot(model)

new_data.cv2 <- train.kknn(classes ~ ., new_data, nn=50, kernel="triangular")
plot(new_data.cv2, type="b")