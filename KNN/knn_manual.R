#需要预测的新样本特征值
new_sample <- c(1.8, 1.9, 1.7, 1.7)
names(new_sample) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
iris_features <- iris[,1:4]

#样本之间的距离测算函数，这里使用了欧氏距离
dist_eucl <- function(x1, x2) sqrt(sum((x1-x2) ^ 2))
distances <- apply(iris_features, 1, function(x) dist_eucl(x, new_sample))

#找出k个最近邻样本点
kneighbor <-head(iris[order(distances),], 5)[,5]
kcount <- table(kneighbor)
dataframe_kcount <- as.data.frame(kcount)
order_count <- dataframe_kcount[order(dataframe_kcount$Freq, decreasing = TRUE),]

#找出频数最高的样本作为新样本点的类别
MLneighbor <- as.character(order_count[1,1])
print(paste("预测新样本为：",MLneighbor,"类"))


# 微信公众号：机器学习之分类算法：K最近邻(KNN)[r实现] <原创： 骧颦  左手r右手python  6月28日
