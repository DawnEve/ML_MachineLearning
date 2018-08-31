
#LOGIT REGRESSION | R DATA ANALYSIS EXAMPLES
#https://stats.idre.ucla.edu/r/dae/logit-regression/

#
library(aod)
library(ggplot2)
mydata <- read.csv("binary.csv")

#探索数据
head(mydata)
#   admit gre  gpa rank
#1     0 380 3.61    3
#2     1 660 3.67    3
#3     1 800 4.00    1
#数据Y是录取admit，其他3个是自变量
summary(mydata)
sapply(mydata,sd)
## two-way contingency table of categorical outcome and predictors we want
## to make sure there are not 0 cells
xtabs(~admit+rank, data=mydata)
str(mydata)

par(mfrow=c(1,3)) 
for(i in 2:4){
  boxplot(mydata[,i],main=colnames(mydata)[i])
}

#把category变成分类变量
mydata$rank <- factor(mydata$rank)
str(mydata)
#开始logistic拟合
mylogit <- glm(admit ~ gre + gpa + rank, data = mydata, family = "binomial")
summary(mylogit)
#
##可见全部参数都通过验证(p<0.05)
#解读：
# 第一部分是 call，提示我们采用的模型
# 第二部分是 离差残差, 是模型fit的一个度量。
# 第三部分最重要，系数、标准差、z统计量(也叫Wald z-statistic)、p值
#   gre和gpa显著，以及rank的三个值。
#   gre增加1，则录取(比不录取)胜算log odds增加 0.002
#   rank2比着rank1，录取胜算log odds增加-0.67
# 第四部分是评价模型fit程度的。

#获得各个参数的CI值
## CIs using profiled log-likelihood
confint(mylogit)
## CIs using standard errors
confint.default(mylogit)


#We can test for an overall effect of rank using the wald.test function of the aod library.
wald.test(b = coef(mylogit), Sigma = vcov(mylogit), Terms = 4:6)
#p-value of 0.00011 indicating that the overall effect of rank is statistically significant.
#
l <- cbind(0, 0, 0, 1, -1, 0)
wald.test(b = coef(mylogit), Sigma = vcov(mylogit), L = l)
#p-value of 0.019, indicating that the difference between the coefficient for rank=2 
#and the coefficient for rank=3 is statistically significant.
#
## odds ratios only
exp(coef(mylogit))
## odds ratios and 95% CI
exp(cbind(OR = coef(mylogit), confint(mylogit)))

#造一批数据
newdata1 <- with(mydata, data.frame(gre = mean(gre), gpa = mean(gpa), rank = factor(1:4)))
## view data frame
newdata1

#做预测
#that the type of prediction is a predicted probability (type="response").
newdata1$rankP <- predict(mylogit, newdata = newdata1, type = "response")
newdata1

#refer
# https://stats.idre.ucla.edu/r/dae/logit-regression/

