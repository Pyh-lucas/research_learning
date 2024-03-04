Python配套博客：https://www.cnblogs.com/nickchen121/p/10718112.html
Sklearn（机器学习）配套博客：https://www.cnblogs.com/nickchen121/p/11686958.html
预训练语言模型的前世今生：https://www.cnblogs.com/nickchen121/p/15105048.html

sklearn中的模块：
preprocessing
inpute：填补缺失值
feature_selection：特征选择
decomposition：降维算法



###  1、查看数据分布报告-PandasProfiling
[【精选】【Python】Pandas profiling 生成报告并部署的一站式解决方案-CSDN博客](https://blog.csdn.net/fengdu78/article/details/122138477)
[python——ydata-profiling介绍与使用_ALittleHigh的博客-CSDN博客](https://blog.csdn.net/whitedrogen/article/details/132541295)

### 2、缺失值/异常值处理


### 3、编码
[[01-编码]]
[[03-相关性分析]]

### 4、相关性分析

[[04-缺失值处理]]

### 5、特征选择&特征提取
[[05-特征选择vs特征提取]]

#### Trick

查看连续性数据的数据分布，类别型数据的不平衡现象
	![Pasted image 20240229144721](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403040851890.png?imageSlim)
### insights[](https://www.kaggle.com/code/shadechen/eda-otuna-ensemble-lgb-xgb-cat#insights)
---
1. Some categorical feature columns is quite imbalanced (Sex), consider resampling to balance the data.
2. ==Some numerical feature have symtoms of Negative skewed or positive skewed, consider applying following:==
    1. ==data transformation==
    2. ==outlier removal==
    3. ==data normalization==
    4. ==Reduce data diemension==
3. Some categorical feature can apply feature engineering to transformed as bool type(example: Y as True, N as False)

### 6、模型评估
[[06-模型评估]]

### 7、不平衡
[[07-不平衡数据集]]

### 8、参数调优
[[08-参数调优]]

### 9、离散化
[[09-离散化]]

### 10、特征工程
[[10-特征工程]]