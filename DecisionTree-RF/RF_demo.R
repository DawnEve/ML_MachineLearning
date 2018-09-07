
# 
#install.packages("randomForest")
library("randomForest")
View(iris)
#划分数据
set.seed(100)
ind=sample(2,nrow(iris),replace=TRUE,prob=c(0.8,0.2));ind
table(ind)
#trainning
iris.rf=randomForest(Species~.,iris[ind==1,],ntree=50,
                     nPerm=10,mtry=3,proximity=TRUE,importance=TRUE)  
print(iris.rf)

#做预测
iris.pred=predict(iris.rf,iris[ind==2,])
iris.pred
#
table(observed=iris[ind==2,"Species"],predicted=iris.pred) 

#
#此外还可以计算各变量在分类中的重要性：
#type: either 1 or 2, specifying the type of importance measure 
#(1=mean decrease in accuracy, 2=mean decrease in node impurity).
importance(iris.rf,type=1)  #重要性评分
importance(iris.rf, type=2) #Gini指数
#变量重要性可视化：
varImpPlot(iris.rf)  #可视化
