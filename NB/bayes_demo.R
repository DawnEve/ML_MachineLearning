#R语言Bayes实例：
library(e1071)

#抽样
#head(iris)
set.seed(100)
ind=sample(2,nrow(iris),replace=TRUE,prob=c(0.8,0.2));ind
table(ind)

#训练
#m <- naiveBayes(Species ~ ., data = iris)
## alternatively:
nB_model<-naiveBayes(iris[ind==1,1:4],iris[ind==1,5])
nB_model

#返回预测的标签列表
pred=predict(nB_model,iris[ind==2,-5]) 

#returns the confusion matrix
table(pred, iris[ind==2,]$Species)
