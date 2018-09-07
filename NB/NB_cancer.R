#NB 算法
setwd("C:\\Users\\admin\\Desktop\\ML_MachineLearning\\NB")
data1=read.csv("breast-cancer-wisconsin.txt",header=F)

# 不加这一句就不行，因为贝叶斯分类器的结论不能是数字，必须是分类变量。
#https://stackoverflow.com/questions/19961441/naivebayes-in-r-cannot-predict-factor0-levels
data1$V11=ifelse(data1$V11==2,"B","M") #2是良性，4恶性
data1$V11=factor(data1$V11) #分类变量要变成因子类型，否则还是不能预测

head(data1);dim(data1)

#分组
set.seed(1)
ind=sample(2,nrow(data1), replace=T, prob=c(0.8,0.2));head(ind);table(ind)
# 去掉第一列的id。
train_df=data1[ind==1,2:11]
test_df=data1[ind==2,2:11]
head(train_df)
#训练
library(e1071)
#fit=naiveBayes(V11~., data=train_df)
fit=naiveBayes(train_df[,1:9],train_df[,10])
fit
#测试
pred=predict(fit, test_df[,1:9])
pred
tb=table(pred, test_df[,10]);tb
#正确率
sum(diag(tb))/sum(tb) #96.35 的正确率，很牛逼了。
#


