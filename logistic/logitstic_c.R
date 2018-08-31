###########
# logistic 判别
setwd("C:\\Users\\admin\\Desktop\\ML_MachineLearning\\logistic")
life=read.csv("alldt.txt",header=T,sep='\t')
head(life)
dim(life)#50  4
#开始拟合
glm.sol=glm(Y~X1+X2+X3, family=binomial, data=life)
summary(glm.sol)
# 发现常数项没有通过检验，p值为0.6823。
#用step做变量筛选
glm.new=step(glm.sol)
glm.sol2=glm(formula = Y~X2+X3, family = binomial, data = life)
summary(glm.sol2)
#由summary()结果的最下方Residual deviance实际上就是-2Log L(-2倍的似然对数)
# 此时，所有参数都通过了检验(p<0.1)
#回归模型为 t=exp(-1.6419-0.7070*x2+2.7844*x3); p=t/(1+t)

#再做预测分析
pred=predict(glm.sol2, data.frame(X2=2, X3=0))
str(pred)
p=exp(pred)/(1+exp(pred));p #0.04496518 
#
pred=predict(glm.sol2, data.frame(X2=2, X3=1))
p=exp(pred)/(1+exp(pred));p #0.4325522 
#因此巩固治疗比没有巩固治疗提高了9.619715倍。

#还可以做回归诊断
influence.measures(glm.sol)
#表明第5、46号样本可能有问题。
#
#对应模型的显著性检验。也可以查看更详细的Residual deviance过程：
anova(glm.sol2, test="Chisq")

#给出回归系数的置信区间
CI=confint(glm.sol2)
parameter=cbind(glm.sol2$coefficients,CI)
parameter
#给出优势比odds ratio的置信区间
OR=exp(glm.sol2$coefficients)
OR_CI=cbind(OR, exp(confint(glm.sol2)))
OR_CI


#模型预测
life2=life[,-1]
probability=predict(object=glm.sol2, newdata=life2[,c('X2','X3')], type="response")
#probability 一堆概率值
pre_test=cbind(life2,probability);head(pre_test)
#主观定义0.5做分界线
pre_test=transform(pre_test, predict=ifelse(probability<=0.5, 0,1))
head(pre_test)
#比较预测结果准确程度
table(pre_test$Y, pre_test$predict)
#

#画出 ROC曲线
library(ROCR)
pred <- prediction(pre_test$probability, pre_test$Y)
perf <- performance(pred,"tpr","fpr")
str(perf)

##AUC值,ROC曲线下面积为AUC，用来评价分类器的综合性能，该数值取0-1之间，越大越好。
#https://blog.csdn.net/Hellolijunshy/article/details/79991385
auc <- performance(pred,'auc');auc_value=auc@y.values[[1]]
auc_value=round(auc_value,2)
auc_value #0.84

plot(perf,colorize=TRUE,main=auc@y.name)
abline(a=c(0,0),b=c(1,1),col="gray")
text(0.5,0.4,paste("AUC: ", round(auc_value, digits=2)), col="blue")
#


