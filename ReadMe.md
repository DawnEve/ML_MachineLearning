# Machine Learning with Python and R (造轮子+用轮子)

Note: Python and R code is edited with Jupyter Notebook.
[My github](https://github.com/DawnEve/ML_MachineLearning)




# 第0章 模型的选择


```
2. EM算法：https://www.zhihu.com/question/27976634


1. 马尔可夫模型 todo

-https://www.cnblogs.com/baiboy/p/hmm1.html
-举例法经典HMM扫盲帖：时间序列(七): 高冷贵族: 隐马尔可夫模型 < 原创： Pegasus  夏洛克AIOps  2017-08-24
-有一点算是MM：捉迷藏 | HMM 隐马尔科夫链 < Math & Lucy  伴露  2016-06-13

```




## 1. k fold cross validation //todo


## 2. AUC 曲线 //todo







# 第一章 有监督的分类


## 1. KNN (K-近邻)
数据集： http://archive.ics.uci.edu/ml/index.php











## 2. Tree & RF (决策树)

主要算法: ID3, C4.5, CART;

(1) 基于R

干货 | 基于R语言和SPSS的决策树算法介绍及应用

moojnn  大数据魔镜  2015-11-06


(2) 基于Py

ID3 无法直接处理数值型的feature。 CART 是不是更好呢？









## 3. NB: naive bayes (朴素贝叶斯)

(2) py版的 NaiveBayes: 离散型、连续型 NB/Naive_bayes.ipynb









## 4. LR: Logistic Regression(逻辑斯蒂回归)

[ROC曲线]How to Perform a Logistic Regression in R

https://datascienceplus.com/perform-logistic-regression-in-r/


[分数据]Simple Guide to Logistic Regression in R

https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/


(2) py 实现 logistic/LR_demo.ipynb

梯度上升 -> 随机梯度上升(占用更少的计算资源)









## 5. SVM (支持向量机)

pluskid 的SVM系列教程 http://blog.pluskid.org/?page_id=683

(1)支持向量机通俗导论（理解SVM的三层境界） https://blog.csdn.net/v_july_v/article/details/7624837

(2)http://scikit-learn.org/stable/modules/svm.html

http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

Spark机器学习系列之13：支持向量机SVM（Python） https://blog.csdn.net/qq_34531825/article/details/52881804





### 核函数

(3)核函数是SVM的力量之源！

核函数需要花时间研究，也是创新点。
https://blog.csdn.net/qq_34531825/article/details/52895621

机器学习--SVM（支持向量机）核函数原理以及高斯核函数 https://blog.csdn.net/wenqiwenqi123/article/details/79313876

引入核函数后的损失函数 https://blog.csdn.net/wenqiwenqi123/article/details/79314166


(4)String Kernel SVM

https://www.cnblogs.com/emanlee/archive/2011/12/07/2278830.html


(5)代码 Understanding Support Vector Machine algorithm from examples (along with code)

https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/












## 6. 集成学习 //todo


### adaBoost (adaptive boosting)

path: /adaBoost/




















# 第二章 回归预测



## 1. 数值型回归: 回归 //todo


### 岭回归

### lasso

### 前向逐步回归






## 2. 树回归 //todo















# 第三章 无监督学习



## 1. K-means (K均值聚类) //todo











# 第四章 降维技术


## 1. PCA //todo


## 2. 利用 SVD 简化数据

## 3. t-SNE 算法

path: /t-SNE/

python实现

R实现


## 4. UMAP //todo










# 第五章 Neural network (神经网络)

## 1. ANN / CNN

手写数字识别


















# 附录A 其他工具

## 1. MapReduce //todo

## 2. 线性代数与python3

path: /pythonPKG/ 关于numpy, pandas, matplotlib/seaborn 的使用.

path: txtBlog/data/Math/




## 3. 概率论

## 4. 机器学习资源











# 附录B 数据集

## 1. 数据集 data/breast-cancer-wisconsin.txt [link](https://archive.ics.uci.edu/ml/index.php) 

[breast-cancer-wisconsin.data](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data) | 
[breast-cancer-wisconsin.names](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names)


|  Attribute    |          Domain |
| ------------- |: -------------------:|
|   1. Sample code number            | id number
|   2. Clump Thickness               | 1 - 10 肿块密度
|   3. Uniformity of Cell Size       | 1 - 10 细胞大小均一性
|   4. Uniformity of Cell Shape      | 1 - 10 细胞形状均一性
|   5. Marginal Adhesion             | 1 - 10 边界粘附
|   6. Single Epithelial Cell Size   | 1 - 10 单个上皮细胞大小
|   7. Bare Nuclei                   | 1 - 10 裸核
|   8. Bland Chromatin               | 1 - 10 微受激染色质
|   9. Normal Nucleoli               | 1 - 10 正常核
|  10. Mitoses                       | 1 - 10 有丝分裂
|  11. Class:                        | (2 for benign良性, 4 for malignant恶性)

除掉id列，一个9个输入属性，和一个Class肿瘤判定结果。



## 2. Horse Colic Data Set(可用于分类, 有缺失值)  [link](https://archive.ics.uci.edu/ml/datasets/Horse+Colic)

logistic 回归: 从疝气病预测病马的死亡率 (斧头书chapter5.3, P85)


## 3. iris data

150行4列特征1列分类。共3类鸢尾花。

来自R语言内置数据集。



