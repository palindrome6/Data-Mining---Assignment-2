library(OpenML)
library(ggplot2)
library(cowplot)
library(mlr)
library(graphics)
library(rpart)
library(party)
library(rattle)
library(mlbench)
library(caret)
library(clusterGeneration)
library(mnormt)
library(corrplot)
library(RWeka)
library(partykit)

iris = getOMLDataSet(did = 10, verbosity=0)$data

iris<-iris[!(iris$class=="fibrosis" | iris$class=="normal"),]
iristask = makeClassifTask(data = iris, target = "class")
iris_mat = data.matrix(iris)
# ggplot(data=iris, aes(x=sepallength, y=sepalwidth, color=class, shape=class)) + geom_point(size=2) + theme_cowplot() + theme(legend.position = c(0.14, 0.80))

ab = iris[,1:18]
cd = iris[,19]

str(iris)
m1 <- J48(class~., data = iris)
# if(require("party", quietly = TRUE)) plot(m1)


rpart = makeLearner("classif.rpart")
rpart = setHyperPars(rpart, cp = 0, minbucket=4)
model = train(rpart, m1)
fancyRpartPlot(model$learner.model, sub="")
