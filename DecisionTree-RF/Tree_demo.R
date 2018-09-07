#
#
#决策树，及其可视化
library(rpart)
#library(rpart.plot)
library(rattle)
#数据划分
head(iris)
set.seed(100)
ind=sample(2,nrow(iris),replace=TRUE,prob=c(0.8,0.2));ind
table(ind)
# trainning[1]
fit <- rpart(Species ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
             data = iris[ind==1,])
fit
#summary(fit)

# 可视化
#http://bbs.pinggu.org/thread-1391518-1-1.html
plot(fit,margin = 0.1) #神奇的参数 margin = 0.1
text(fit, use.n = TRUE) #pic 1 
#fancyRpartPlot(fit) #fancy图

# 输出决策树规则：rattle程序包里面的asRules函数：
asRules(fit)

#
###########
#2.做预测[2]
pred=predict(fit, newdata=iris[ind==2,]);dim(pred)
pred
#给出的是p值，还需要重新按归类：
pred.1=apply(pred, 1, function(x){
  colnames(pred)[which(x==max(x)) ]
})
pred.1=as.character(pred.1);pred.1
#做交叉表查看准确程度。
table(observed=iris[ind==2,"Species"],predicted=pred.1)
#

