#
#######
# KNN 可以直接进行多分类问题。

#引入数据，就是换个短名字
x=iris

#探索数据：数据结构分析
dim(x);head(x)
str(x)
table(x$Species)
round(prop.table(table(x$Species))*100,digits=1 )
summary(x)
summary(x[c("Petal.Width","Sepal.Width")])

#探索数据：图形
library(ggplot2)
ggplot(x,aes(Sepal.Length, Sepal.Width,color=Species))+geom_point()
ggplot(x,aes(Petal.Length, Petal.Width,color=Species))+geom_point()


#引入包
library(class)


#数据归一化处理:标准化方法，主要有两种，即Z标准化或0-1标准化处理
#本文用2做归一化函数，最大-最小化归一化
normalize <- function(x){
  num <- x - min(x)
  span <- max(x) - min(x)
  return(num/span)
}

#标准化函数1(未使用)
standard <- function(x) {
  (x-mean(x))/sd(x)
}

#1
x_norm <- as.data.frame(lapply(x[,1:4], normalize))
head(x_norm)
summary(x_norm)
#2
#将该函数应用到数据框cancer中的每一列（除Diagnosis变量）
# cancer_standard <- sapply(X = cancer[,3:32], FUN = standard)
# summary(cancer_standard)


#生成抽样序列
set.seed(100)
ind=sample(2,nrow(x_norm),replace=TRUE,prob=c(0.8,0.2));ind
#取80%作为训练集，20为测试集
x_train=x_norm[ind==1,];dim(x_train) #121
x_test=x_norm[ind==2,];dim(x_test) #29 去掉标签

#fitting model
fit <-knn(train=x_train, test=x_test, cl=x[ind==1,]$Species, k=5)
summary(fit)
#模型评价:采用交叉联表
table(fit, x[ind==2,]$Species)
#




# 问题：knn中的k怎么选择呢？遍历后用最优
accs=c()
for(i in 1:round(sqrt(nrow(x_train)))){
  fit <-knn(train=x_train, test=x_test, cl=x[ind==1,]$Species, k=i)
  Freq=table(fit, x[ind==2,]$Species)
  acc <- sum(diag(Freq))/sum(Freq)
  accs=c(accs, acc)
}
#可视化
plot(1:round(sqrt(nrow(x_train))),accs,type = 'b', pch = 20, col= 'blue',
     main = 'k vs. accuracy',
     xlab = 'k', ylab = 'accuracy')
#所以取7 8 10 11结果最好
fit.best <-knn(train=x_train, test=x_test, cl=x[ind==1,]$Species, k=7)
Freq=table(fit.best, x[ind==2,]$Species)
#预判正确率
sum(diag(Freq))/sum(Freq) #0.9310345
#简单而易用的knn算法能够有97%的把握，给出分类信息。
