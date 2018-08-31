#
##########
#a=2;b=6-a;b
#c=-(a/6)*log2(a/6) - (b/6)*log2(b/6);c

#R语言SVM实例
library(e1071)
#去掉一类，变成两类
x=iris[which(iris$Species!="virginica"),];
x$Species=factor(x$Species)
str(x);dim(x)
#生成抽样序列
set.seed(100)
ind=sample(2,nrow(x),replace=TRUE,prob=c(0.8,0.2));ind
#取80%作为训练集，20为测试集
x_train=x[ind==1,];dim(x_train)
x_test=x[ind==2,c(1:4)];dim(x_test) #去掉标签
#fitting model
fit=svm(Species~., data=x_train)
summary(fit)

#predict Output
predicted=predict(fit,x_test)
predicted
#给出混淆矩阵:分类100%正确
table(predicted,x[ind==2,c(5)])
#
#The end
#SVM好像无法给出ROC曲线？ //todo
#





#
#用两类画点图。和上文无关，上文是四个维度。
library(ggplot2)
ggplot(x, aes(Sepal.Length, Sepal.Width, color=Species))+geom_point()
#

#保存csv数据，供python调用
write.csv(x_train, "c://Tools/x_train.csv",row.names = FALSE)
write.csv(x_test, "c://Tools/x_test.csv",row.names = FALSE)
write.csv(y_test, "c://Tools/y_test.csv",row.names = FALSE)
#