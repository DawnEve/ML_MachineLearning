###########
# logistic 回归
setwd("C:\\Users\\admin\\Desktop\\ML_MachineLearning\\logistic")
dt1=read.csv("logit_sample.csv",header=T)
head(dt1)
# age case total
#1   9    0   376
#2  10    0   200
#3  11    0    93
#4  12    2   120
#5  13    2    90
#6  14    5    88

# 年龄与患病率
library(ggplot2)
ggplot(dt1,aes(age, case/total))+geom_point()#+geom_smooth()
#formula=y~x #y是否患病(0/1),x是危险因素矩阵
#本文只好采用矩阵形式
case=dt1$case
control=dt1$total-dt1$case
dt2=cbind(case,control);head(dt2)
#拟合
glm.out=glm(dt2~age, family=binomial(link=logit),data=dt1)
summary(glm.out)
#str(glm.out)
# 拟合的图形
plot(case/total~age, data=dt1,pch=16,col=1)
lines(dt1$age,glm.out$fitted.values, type="l", col=4,lwd=2)
title(main="Fitted Logistic Regression Line")
#
p=1-pchisq(43.084,23);p# 0.006925646
#
#参考：简单logistic回归的R语言实例 < chgaoming  生物和医学统计学  2017-12-19

#用上述参数计算给定age的p值
lp=function(age){
  t=exp(-8.8360+0.4272*age)
  return(t/(1+t))
}

#预测几个age的p值，并画出来
ages=c(10,15,20,25,30,35,40)
y=lp(ages)
points(ages,y, col=2, pch=16,cex=2.5)
#

