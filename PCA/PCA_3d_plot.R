#step1: 对iris做PCA分析，按列做
df.pca<-prcomp(iris[,1:4])
summary(df.pca)
str(df.pca)

#check: PC sdev, percentage
barplot(df.pca$sdev)


#check: 旋转矩阵 ?? 怎么理解? 
# 4个原变量怎么变成新变量的，线性组合系数
# PC1=0.36*SL-0.08*SW+0.85PL+0.35PW
df.pca$rotation
library(pheatmap)
pheatmap( df.pca$rotation, scale="none",
         border_color = NA,
         cluster_rows = F, cluster_cols = F,
         main="rotation matrix")


#step2: 获取PC
pca.result<-df.pca$x 
pca.result<-data.frame(pca.result)
head(pca.result)
pca.result$Species<-iris$Species

#step3: 总共数据是150，准备150个颜色和150个形状
colors0 <- c("#999999", "#E69F00", "#56B4E9")
colors <- colors0[as.numeric(pca.result$Species)]
shapes0<-16:18
shapes<-shapes0[as.numeric(pca.result$Species)]

#step4: 2d plot
library(ggplot2)
ggplot(pca.result, aes(PC1, PC2, color=Species))+
  geom_point()+theme_bw()

#step4: 3d plot
library("scatterplot3d")
s3d <- scatterplot3d(pca.result[,1:3],
                     pch = shapes,
                     color=colors,
                     angle=80, #x和y轴的夹角
                     cex.symbols = 1)
usrP=par("usr");usrP
legend(x=usrP[1],y=usrP[4]+2, #"top", 
       legend = levels(pca.result$Species),
       col = c("#999999", "#E69F00", "#56B4E9"),
       pch = c(16, 17, 18), box.col = NA,
       inset = -0.1, xpd = TRUE, horiz = TRUE)

# ref: https://zhuanlan.zhihu.com/p/375110294