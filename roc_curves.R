options(warn=-1)
library(mlr) # ML library
library(OpenML) # Data import and sharing results
library(ggplot2) # Plotting
library(cowplot) # Plot styling
library(rattle) # Plotting trees  
library(kknn)
library(pROC)

# Create data.

true_label <- factor(c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2))
prediction_A <- factor(c(1, 1, 2, 2, 1, 1, 2, 2,  1, 2, 2, 2, 2))
prediction_B <- factor(c(1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2))
prediction_C <- (c(0.8, 0.9, 0.7, 0.6, 0.4, 0.8, 0.4, 0.4, 0.6, 0.4, 0.4, 0.4, 0.2))

# colnames(true_label) <- c("true_label")
# colnames(prediction_A) <- c("prediction_A") 
# colnames(prediction_B) <- c("prediction_B")
# colnames(prediction_C) <- c("prediction_C")

# Create dataframes.
temp1 <- data.frame(prediction_A, true_label)
temp2 <- data.frame(prediction_B, true_label)
temp3 <- data.frame(prediction_C, true_label)

#Rename columns.
colnames(temp1) <- c("prediction_A", "true_label")
colnames(temp2) <- c("prediction_B", "true_label")
colnames(temp3) <- c("prediction_C", "true_label")

# for temp1 #######################################################################
print (temp1)
temp1.task = makeClassifTask(data = temp1 ,target = "true_label")

n = getTaskSize(temp1.task) # make a 2/3 split
print(n)
# train1.set = sample(n, size = round(2/3 * n))
train1.set = prediction_A[1:8]
print(train1.set)
test1.set = setdiff(seq_len(n), train1.set)

lrn1 = makeLearner("classif.kknn", predict.type = "prob") # output probabilities
mod1 = train(lrn1, temp1.task, subset = train1.set)
pred1 = predict(mod1, task = temp1.task, subset = test1.set)

# for temp2 #######################################################################
print (temp2)
temp2.task = makeClassifTask(data = temp2 ,target = "true_label")

n = getTaskSize(temp2.task) # make a 2/3 split
print(n)
# train2.set = sample(n, size = round(2/3 * n))
train2.set = prediction_B[1:8]
print(train2.set)
test2.set = setdiff(seq_len(n), train2.set)

lrn2 = makeLearner("classif.kknn", predict.type = "prob") # output probabilities
mod2 = train(lrn2, temp2.task, subset = train2.set)
pred2 = predict(mod2, task = temp2.task, subset = test2.set)

# for temp3 #######################################################################

print (temp3)
temp3.task = makeClassifTask(data = temp3 ,target = "true_label")

n = getTaskSize(temp3.task) # make a 2/3 split

print(n)
train3.set = sample(n, size = round(2/3 * n), replace = false, prob = [0.8, 0.9, 0.7, 0.6, 0.4, 0.8, 0.4, 0.4, 0.6, 0.4, 0.4, 0.4, 0.2])
# train3.set = prediction_C[1:8
print(train3.set)
test3.set = setdiff(seq_len(n), train3.set)


lrn3 = makeLearner("classif.kknn", predict.type = "prob", predict.threshold = 0.2) # output probabilities
mod3 = train(lrn3, temp3.task, subset = train3.set)
pred3 = predict(mod3, task = temp3.task, subset = test3.set)

# Plot.
df = generateThreshVsPerfData(list(predictionA = pred1, predictionB = pred2, predictionC = pred3), measures = list(fpr, tpr))
plotROCCurves(df)
qplot(x = fpr, y = tpr, color = learner, data = df$data, geom = "path")

# Evaluate performance over different thresholds
# df = generateThreshVsPerfData(pred1, measures = list(fpr, tpr, mmce))
# plotThreshVsPerf(df)
# auc(temp1$survived, temp1$pred1)
  
  
  