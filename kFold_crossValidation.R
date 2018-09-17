# K折交叉验证R语言实现

###########
# 1. 数据输入
setwd(choose.dir())
data1=read.csv("NB/breast-cancer-wisconsin.txt", header = F,na.strings = NA)
#去掉第一列id
data1=data1[,-1]
#减去V7为?的行
data1=data1[-which(data1$V7=="?"),]
data1$V7=as.numeric(data1$V7)
#定义最后一列为分类变量
data1$V11=ifelse(data1$V11==2,"B","M") #2是良性，4恶性
data1$V11=factor(data1$V11) #分类变量要变成因子类型，否则还是不能预测
#重新命名行名,有必要吗？
rownames(data1)=1:nrow(data1)
#检查数据
dim(data1)
head(data1)
str(data1)
#


###########
#2.分组、拟合、预测
#分组
ind=sample(2,nrow(data1),replace = T,prob=c(0.8,0.2));head(ind)
cancer_train=data1[ind==1,];dim(cancer_train)
cancer_test=data1[ind==2,]; dim(cancer_test)

#生成logistic模型
fit=glm(V11~., family = binomial(link="logit"), data=cancer_train)
summary(fit)

#预测
pre=predict.glm(fit, type="response", newdata = cancer_test[,-10])
head(pre) #预测到的是百分比




###########
#3.模型检验
n=nrow(cancer_train) #训练数据的行数，也就是样本数量
#计算Cox-Snell拟合优度
R2=1-exp((fit$deviance - fit$null.deviance)/n);R2
cat("Cox-Snell R2=",R2,"\n")
#计算Nagelkerke拟合优度，我们在最后输出这个拟合优度值
R2_<-R2/(1-exp((-fit$null.deviance)/n));R2_
cat("Nagelkerke R2=",R2_,"\n")
##模型的其他指标
residuals(fit)     #残差
coefficients(fit)  #系数，线性模型的截距项和每个自变量的斜率，由此得出线性方程表达式。或者写为coef(fit)
anova(fit)         #方差
summary(fit) #这个是模型汇总



###########
#4.准确率和精度
# 使用哪个分界线好呢？需要做ROC曲线
pre2=ifelse(pre>0.5,"M","B");head(pre2)
#混淆矩阵，显示结果依次为TP、FN、FP、TN
tb=table(pre2, cancer_test[,10]);tb

#正确率（accuracy）
accu=sum(diag(tb))/sum(tb)*100;accu



###########
#5.ROC曲线的几个方法
# 方法1
library(ROCR)
pred=prediction(pre,ifelse(cancer_test$V11=="M",1,0)) #预测概率和真实值
performance(pred, "auc")@y.values[[1]] #AUC值
perf=performance(pred, 'tpr','fpr')
plot(perf, colorize=T)
# 方法2
library(pROC)
mroc=roc(ifelse(cancer_test$V11=="M",1,0), pre)
mroc
#画出ROC曲线，标出坐标，并标出AUC的值
plot(mroc, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE) 
#方法3，按ROC定义
fpr=rep(0,1000)
tpr=rep(0,1000)
for(i in 1:1000){
  p0=i/1000;
  ypre=ifelse(pre>p0,"M","B")
  tb=table(ypre,cancer_test$V11) 
  fpr[i]=tb[2]/sum(tb[2],tb[4])
  tpr[i]=tb[1]/sum(tb[1],tb[3])
}
plot(fpr,tpr,type="l",col=2)
#points(c(0,1),c(0,1),type="l",lty=2) #这一句干啥呢？






###########
#6.更换测试集和训练集的选取方式，采用十折交叉验证
library("caret")
#随机分成十组
set.seed(1)
folds=createFolds(y=data1$V11, k=10)
#构建for循环，得10次交叉验证的测试集精确度、训练集精确度
accuracy1c=NULL
accuracy2c=NULL
for(i in 1:10){
  fold_test=data1[folds[[i]],]
  fold_train=data1[-folds[[i]],]
  print(paste0("**组号:",i))
  #建模
  fold_fit=glm(V11~., family=binomial(link="logit"), data=fold_train)
  ####
  print("***测试集精确度***")
  fold_predict_=predict(fold_fit, type="response", newdata=fold_test)
  fold_predict=ifelse(fold_predict_>0.5,"M","B")
  fold_test$predict=fold_predict
  #
  fold_error=sum( ifelse((fold_test$V11!=fold_test$predict)==T,1,0) )
  fold_accuracy=1-fold_error/nrow(fold_test)
  print(fold_accuracy)
  
  #方法二：计算准确度
  #print(table(fold_test$predict,fold_test$V11)) #用混淆矩阵
  #print(sum(diag(tb) )/sum(tb))#对角线/总样本数
  
  ####
  print("***训练集精确度***")
  fold_predict2_=predict(fold_fit, type="response", newdata=fold_train)
  fold_predict2=ifelse(fold_predict2_>0.5,"M","B")
  fold_train$predict=fold_predict2
  #
  fold_error2=sum( ifelse((fold_train$V11!=fold_train$predict)==T,1,0) )
  fold_accuracy2=1-fold_error2/nrow(fold_train)
  print(fold_accuracy2)
  
  #输出准确度数组
  accuracy1c=c(accuracy1c,fold_accuracy)
  accuracy2c=c(accuracy2c,fold_accuracy2)
}
#输出准确度的平均值
mean_accuracy=mean(accuracy1c); #test
print(paste0("the mean accuracy is ", round(mean_accuracy,4) ))

#回测做参考
# mean_accuracy2=mean(accuracy2c);mean_accuracy2 #train



#########
#剩下的没啥用
hist(accuracy1c,n=10)
max(accuracy1c) #最高的准确度
which(accuracy1c==max(accuracy1c)) #第四个分组效果最好
