[Learn Machine Learning Explainability | Kaggle](https://www.kaggle.com/learn/machine-learning-explainability)

	Extract human-understandable insights from any model.
---
## 1. Use cases 什么时候需要？

	 "black boxes"： you can't understand the logic behind those predictions

### 01 ==目前可解释性能够获得这些insights==
- What features in the data did the model think are most important?什么特征是最重要的
- For any single prediction from a model, how did each feature in the data affect that particular prediction?对于模型中的任何一个预测，数据中的每个特征如何影响该特定预测？
- How does each feature affect the model's predictions in a big-picture sense (what is its typical effect when considered over a large number of possible predictions)?从宏观上看，每个特征如何影响模型的预测（当考虑大量可能的预测时，它的典型效果是什么）？


### 02 ==这些insights的好处==

- Debugging
- Informing feature engineering
- Directing future data collection
- Informing human decision-making
- Building Trust

##### Debugging
不可靠的脏数据可能在数据预处理时候添加潜在的错误信息，同时可能出现目标泄露的问题，这些问题很常见。==Debug可以发现模型挖掘的pattern和现实世界是否不一致，这是追踪错误的第一步==
	The world has a lot of unreliable, disorganized and generally dirty data. You add a potential source of errors as you write preprocessing code. Add in the potential for [target leakage](https://www.kaggle.com/alexisbcook/data-leakage), and it is the norm rather than the exception to have errors at some point in a real data science project.

##### Informing Feature Engineering
	特征工程是提升模型精度最有效的方法
[Feature engineering](https://www.kaggle.com/learn/feature-engineering) is usually the most effective way to improve model accuracy. Feature engineering usually involves repeatedly creating new features using transformations of your raw data or features you have previously created.

	尽管优势可以凭借对潜在主题的直觉来完成，然而当特征增多/缺乏正在研究主题的背景知识时，需要这些知识指导
Sometimes you can go through this process using nothing but intuition about the underlying topic. But you'll need more direction when you have 100s of raw features or when you lack background knowledge about the topic you are working on.

	kaggle中的一场比赛中，特征名称未给出，因此背景知识没法用。两个功能之间的差异，特别是“f527 - f528”，创造了一个非常强大的新功能。将这种差异作为一项功能的模型比没有它的模型要好得多。但是，当您从数百个变量开始时，您会如何考虑创建此变量？
A Kaggle competition to [predict loan defaults](https://www.kaggle.com/c/loan-default-prediction) gives an extreme example. This competition had 100s of raw features. For privacy reasons, the features had names like `f1`, `f2`, `f3` rather than common English names. This simulated a scenario where you have little intuition about the raw data.

	因此可解释的技能能够帮助找到重要特征
The techniques you'll learn in this micro-course would make it transparent that `f527` and `f528` are important features, and that their role is tightly entangled. This will direct you to consider transformations of these two variables, and likely find the "golden feature" of `f527 - f528`.




##### Directing Future Data Collection
	线下组织可以通过你目前挖掘出的模型见解，了解那些特征更重要，从而帮助企业未来的数据收集
	
#写作参考 
You have no control over datasets you download online. But many businesses and organizations using data science have opportunities to expand what types of data they collect. Collecting new types of data can be expensive or inconvenient, so they only want to do this if they know it will be worthwhile. Model-based insights give you a good understanding of the value of features you currently have, which will help you reason about what new values may be most helpful.


##### Informing Human Decision-Making

Some decisions are made automatically by models. Amazon doesn't have humans (or elves) scurry to decide what to show you whenever you go to their website. But many important decisions are made by humans. ==For these decisions, insights can be more valuable than predictions.==

##### Building Trust
展示模型和现实理解相同
Many people won't assume they can trust your model for important decisions without verifying some basic facts. This is a smart precaution given the frequency of data errors. In practice, showing insights that fit their general understanding of the problem will help build trust, even among people with little deep knowledge of data science.

---
## 2. Permutation Importance（你的模型认为哪些特征是重要的？）

	关于“哪些特征对预测影响最大”的问题，一般使用feature importance进行衡量。目前有很多种方法从略微不同的version回答了这个问题，然而同时具有一定缺点。
	此处使用Permutation Importance来衡量，该方法的优势fast to calculate,widely used and understood, and consistent with properties we would want a feature importance measure to have.

### 01 原理

	想要预测一个人20岁时的身高。我们的数据包括有用的特征（10 岁时的身高）、预测能力不强的特征（拥有的袜子）以及我们在本解释中不会重点关注的其他一些特征。
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071424406.png?imageSlim)
==置换重要性是在拟合模型后计算的。==因此，我们不会改变模型或改变我们对给定的身高、袜子数等值的预测。

相反，我们将提出以下问题：==如果我随机随机洗牌验证数据的一列，将目标和所有其他列保留在原位，这将如何影响现在洗牌数据中预测的准确性？==
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071425167.png?imageSlim)
随机重新排序单个列会导致预测的准确性降低，因为生成的数据不再与现实世界中观察到的任何数据相对应。==如果我们对模型进行预测所严重依赖的列进行洗牌，则模型准确性尤其受到影响==。在这种情况下，在 10 岁时改变身高会导致可怕的预测。如果我们改用自有的袜子，那么由此产生的预测就不会受到太大的影响。

**因此，Permutation Importance的流程如下：**
- **获取经过训练的模型。**
- **对单列中的值进行随机排序，使用生成的数据集进行预测。使用这些预测和真实目标值来计算损失函数因洗牌而遭受的损失。==这种性能下降衡量了您刚刚洗牌的变量的重要性。==**
- **将数据恢复到原始顺序（撤消步骤 2 中的随机排序）。现在对数据集中的下一列重复步骤 2，直到计算出每列的重要性。**

	这种思想其实和卓伦师兄说的检测数据good/bad case的方法思想是一样的


### 02 🐎
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071430177.png?imageSlim)

==eli5 已经不能用了==
```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```


[4.2. Permutation feature importance — scikit-learn 1.4.1 documentation](https://scikit-learn.org/stable/modules/permutation_importance.html)
解读负值[Learn Machine Learning Explainability | Kaggle](https://www.kaggle.com/learn/machine-learning-explainability/discussion/356240)
[Stop Permuting Features. Permutation importance may give you… | by Denis Vorotyntsev | Towards Data Science](https://towardsdatascience.com/stop-permuting-features-c1412e31b63f)


![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071430741.png?imageSlim)

### 03 结果解读

- The values towards the top are the most important features, and those towards the bottom matter least.==越顶部的越重要==

- The first number in each row shows how much model performance decreased with a random shuffling (in this case, using "accuracy" as the performance metric).第一个数字表示模型的性能指标在随机打乱后如何==“下降”==

- Like most things in data science, there is some randomness to the exact performance change from a shuffling a column. ==We measure the amount of randomness in our permutation importance calculation by repeating the process with multiple shuffles. The number after the **±** measures how performance varied from one-reshuffling to the next.==+-后面的数字衡量的是连续两次随机打乱时，指标的波动性

- You'll occasionally see negative values for permutation importances. In those cases, the predictions on the shuffled (or noisy) data happened to be more accurate than the real data. This happens when the feature didn't matter (should have had an importance close to 0), but random chance caused the predictions on shuffled data to be more accurate. This is more common with small datasets, like the one in this example, because there is more room for luck/chance.出现负数的哪些特征意味着这些特征并不重要，通常出现在小数据集中

	In our example, the most important feature was **Goals scored**. That seems sensible. Soccer fans may have some intuition about whether the orderings of other variables are surprising or not.

---
## 3. Partial Plots（特征是如何影响预测的？）

### 01 Partial Dependence Plots 部分依赖关系图

特征重要性显示哪些变量对预测影响最大，而==部分依赖关系图显示特征如何影响预测（单位变动如何影响整体数值，不是shuffle）==

This is useful to answer questions like:

- Controlling for all other house features, what impact do longitude and latitude have on home prices? To restate this, how would similarly sized houses be priced in different areas?控制所有其他房屋特征，经度和纬度对房价有什么影响？也就是说，不同地区类似大小的房屋将如何定价？
    
- Are predicted health differences between two groups due to differences in their diets, or due to some other factor?两组之间的预测健康差异是由于饮食差异还是由于其他因素？

If you are familiar with linear or logistic regression models, partial dependence plots can be interpreted similarly to the coefficients in those models. Though, partial dependence plots on sophisticated models can capture more complex patterns than coefficients from simple models. If you aren't familiar with linear or logistic regressions, don't worry about this comparison.如果您熟悉线性回归或逻辑回归模型，则可以将部分依赖图解释为与这些模型中的系数类似。但是，与简单模型中的系数相比，复杂模型上的部分依赖性图可以捕获更复杂的模式。如果您不熟悉线性回归或逻辑回归，请不要担心这种比较。

We will show a couple examples, explain the interpretation of these plots, and then review the code to create these plots.


### 02 原理

与排列重要性一样，部分依赖图是在拟合模型后计算的。Like permutation importance, **partial dependence plots are calculated after a model has been fit.** The model is fit on real data that has not been artificially manipulated in any way.

在我们的足球示例中，球队可能在许多方面有所不同。他们传球了多少次，射门次数多了，进球了多少球，等等。乍一看，似乎很难解开这些特征的影响。

	为了了解部分图如何分离出每个特征的影响，我们首先考虑一行数据。例如，该行数据可能表示一支球队有 50% 的时间控球、100 次传球、10 次射门和 1 个进球。

我们将使用已经训练好了的模型来预测我们的结果（他们的球员赢得“比赛最佳球员”的概率）。==但是我们反复改变一个变量的值来做出一系列预测==。如果球队只有40%的时间控球，我们可以预测结果。然后我们预测他们有 50% 的时间有球。然后再次预测 60%。等等。==然后，We trace out predicted outcomes (on the vertical axis在横轴上) as we move from small values of ball possession to large values (on the horizontal axis在纵轴上).==

	在此描述中，我们仅使用了单行数据。要素之间的交互可能会导致单行的绘图不典型。

因此，我们用原始数据集中的多行重复该实验，并在纵轴上绘制平均预测结果。

### 03 🐎

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071504859.png?imageSlim)
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071505030.png?imageSlim)
	Here is the code to create the Partial Dependence Plot using the scikit-learn library.


```python
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Create and plot the data
disp1 = PartialDependenceDisplay.from_estimator(tree_model, val_X, ['Goal Scored'])
plt.show()
```

y 轴被解释为预测值相对于基线或最左侧值的预测值的变化
The y axis is interpreted as **change in the prediction** from what it would be predicted at the baseline or leftmost value.

从这张特殊的图表中，我们看到进球大大增加了你赢得“全场最佳球员”的机会。但除此之外的额外目标似乎对预测影响不大
From this particular graph, we see that scoring a goal substantially increases your chances of winning "Man of The Match." But extra goals beyond that appear to have little impact on predictions.

==针对Goal Score的变动，y轴是“赢得全场最佳球员”变动的幅度。也就是说，此处随着Goal Score从左到右的变动，“赢得全场最佳球员”的幅度提升==
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071634229.png?imageSlim)
另一个变量：
	这张图似乎太简单了，无法代表现实。但那是因为模型太简单了。您应该能够从上面的决策树中看到（<101.5时，不会发生变化），这完全代表了模型的结构。
	![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071642623.png?imageSlim)
	您可以轻松地比较不同模型的结构或含义。这是与随机森林模型相同的图。
	![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071647800.png?imageSlim)
	This model thinks you are more likely to win Man of the Match if your players run a total of 100km over the course of the game. Though running much more causes lower predictions.该模型认为，如果您的球员在游戏过程中总共跑了 100 公里，您更有可能赢得比赛最佳球员。尽管运行得越多会导致预测值降低。
	In general, the smooth shape of this curve seems more plausible than the step function from the Decision Tree model. Though this dataset is small enough that we would be careful in how we interpret any model.一般来说，这条曲线的平滑形状似乎比决策树模型中的阶跃函数更合理。尽管这个数据集足够小，但我们在解释任何模型时都会小心。

### 04 2D部分依赖图（寻找属性间的交互效应）

[[10-特征工程]] #交互效应

==如果您对特征之间的interaction感到好奇，2D 部分依赖关系图也很有用==。一个例子可以澄清这一点。If you are curious about interactions between features, 2D partial dependence plots are also useful. An example may clarify this.

我们将再次使用决策树模型来绘制此图。它将创建一个非常简单的图，但您应该能够将您在图中看到的内容与树本身相匹配。We will again use the Decision Tree model for this graph. It will create an extremely simple plot, but you should be able to match what you see in the plot to the tree itself.

```python
fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('Goal Scored', 'Distance Covered (Kms)')]
# Similar to previous PDP plot except we use tuple of features instead of single feature
disp4 = PartialDependenceDisplay.from_estimator(tree_model, val_X, f_names, ax=ax)
plt.show()
```
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071651844.png?imageSlim)
This graph shows predictions for any combination of Goals Scored and Distance covered.此图显示了对“进球数”和“覆盖距离”的任意组合的预测。线上的指标显示的时预测变量的性能指数

For example, we see the highest predictions when a team scores at least 1 goal and they run a total distance close to 100km. If they score 0 goals, distance covered doesn't matter. Can you see this by tracing through the decision tree with 0 goals?例如，当一支球队至少打进 1 个球并且他们的总距离接近 100 公里时（在二维坐标中确定一个点），我们会看到最高的预测。如果他们的进球数为0，那么覆盖的距离并不重要。你能通过追踪 0 个进球的决策树来看到这一点吗？

But distance can impact predictions if they score goals. Make sure you can see this from the 2D partial dependence plot. Can you see this pattern in the decision tree too?但是，如果他们进球，距离会影响预测。确保您可以从 2D 部分依赖关系图中看到这一点。你能在决策树中看到这种模式吗？

---
## 4. SHAP Values

	Understand individual predictions

### 01 功能

You've seen (and used) techniques to extract general insights from a machine learning model. But what if you want to break down how the model works for an individual prediction?你已了解（并使用）从机器学习模型中提取general见解的技术。但是，==如果您想分解模型如何用于单个预测，该怎么办？==

SHAP 值（SHapley Additive exPlanations 的首字母缩写）对预测进行细分，以显示每个特征的影响。使用场景如下：

- 一个模型说，银行不应该借钱给别人，法律要求银行解释每次贷款被拒绝的依据
- 医疗保健提供者希望确定哪些因素导致每位患者患某种疾病的风险，以便他们可以通过有针对性的健康干预措施直接解决这些风险因素

在本课程中，您将使用 SHAP 值来解释各个预测。在下一课中，你将了解如何将这些内容聚合为强大的模型级见解。

### 02 原理

SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.  SHAP 值能够解释，当给定特征具体值时，与该特征采用某个基线值时，我们所做的预测的对比影响

An example is helpful, and we'll continue the soccer/football example from the permutation importance and partial dependence plots lessons.
In these tutorials, we predicted whether a team would have a player win the Man of the Match award.、

We could ask:我们可能会问
当一支球队已经进了3球，预测的结果如何？
How much was a prediction driven by the fact that the team scored 3 goals?

But it's easier to give a concrete, numeric answer if we restate this as:
在已经进了3球的情况下，现有预测是多少，而非其他进球的baseline阈值
How much was a prediction driven by the fact that the team scored 3 goals, instead of some baseline number of goals.

Of course, each team has many features. So if we answer this question for number of goals, we could repeat the process for all other features.当然，每个团队都有很多特点。因此，我们可以对每个特征重复计算SHAP

SHAP values do this in a way that guarantees a nice property. Specifically, you decompose a prediction with the following equation:

	sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values

That is, the SHAP values of all features sum up to explain why my prediction was different from the baseline. This allows us to decompose a prediction in a graph like this:也就是说，所有特征的 SHAP 值相加以解释为什么我的预测与基线不同。这允许我们在图中分解预测，如下所示：
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071721487.png?imageSlim)

How do you interpret this?

We predicted 0.7, whereas the base_value is 0.4979. Feature values causing increased predictions are in pink, and their visual size shows the magnitude of the feature's effect. Feature values decreasing the prediction are in blue. The biggest impact comes from `Goal Scored` being 2. Though the ball possession value has a meaningful effect decreasing the prediction.我们的预测值是 0.7，而基准值是 0.4979。导致预测值增加的特征值为粉红色，其视觉大小表示特征影响的程度。降低预测值的特征值为蓝色。影响最大的是进球数为 2 的特征值。尽管控球率值对降低预测值也有一定影响。(我认为这里的prediction可能指的是具体的值，也就是说起到对值本身起到反作用，而不是说这个特征冗余了)

If you subtract the length of the blue bars from the length of the pink bars, it equals the distance from the base value to the output.如果用粉色长条的长度减去蓝色长条的长度，就等于从基础值到输出值的距离。

There is some complexity to the technique, to ensure that the baseline plus the sum of individual effects adds up to the prediction (which isn't as straightforward as it sounds). We won't go into that detail here, since it isn't critical for using the technique. [This blog post](https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d) has a longer theoretical explanation.这项技术有一定的复杂性，要确保基线加上单个影响的总和等于预测值（这并不像听起来那么简单）。由于这对使用该技术并不重要，我们在此就不赘述了。这篇博文有较长的理论解释。

### 03 🐎

[shap/shap: A game theoretic approach to explain the output of any machine learning model. (github.com)](https://github.com/shap/shap)
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071726823.png?imageSlim)

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071726418.png?imageSlim)

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
```

The `shap_values` object above is a list with two arrays. The first array is the SHAP values for a negative outcome (don't win the award), and the second array is the list of SHAP values for the positive outcome (wins the award). We typically think about predictions in terms of the prediction of a positive outcome, so we'll pull out SHAP values for positive outcomes (pulling out `shap_values[1]`).
上述 shap_values 对象是一个包含两个数组的列表。第一个数组是负面结果（未获奖）的 SHAP 值，第二个数组是正面结果（获奖）的 SHAP 值列表。我们通常会从正面结果的预测角度来考虑预测，因此我们会取出正面结果的 SHAP 值（取出 shap_values[1]）。

It's cumbersome to review raw arrays, but the shap package has a nice way to visualize the results.
查看原始数组很麻烦，但 shap 软件包有一个很好的方法来可视化结果。

```python
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
```

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071729125.png?imageSlim)

If you look carefully at the code where we created the SHAP values, you'll notice we reference Trees in shap.TreeExplainer(my_model). But the SHAP package has explainers for every type of model.==如果仔细查看我们创建 SHAP 值的代码，就会发现我们在 shap.TreeExplainer(my_model) 中引用了树。但是，SHAP 软件包为每种类型的模型都提供了解释器。==

- `shap.DeepExplainer` works with Deep Learning models.适用于深度学习模型。
- `shap.KernelExplainer` works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values.适用于所有模型，不过它比其他解释器慢，而且提供的是近似值而不是精确的 Shap 值。

Here is an example using KernelExplainer to get similar results. The results aren't identical because KernelExplainer gives an approximate result. But the results tell the same story.下面是一个使用 KernelExplainer 获得类似结果的示例。由于 KernelExplainer 得出的是近似值，因此结果并不完全相同。但结果说明的问题是一样的。

```python
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071737655.png?imageSlim)
 ![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071737625.png?imageSlim)

---
## 5. Advanced Uses of SHAP Values

	Aggregate SHAP values for even more detailed model insights

	We started by learning about permutation importance and partial dependence plots for an overview of what the model has learned.我们首先学习了排列重要性和部分依赖图，以了解模型所学到的知识。
	We then learned about SHAP values to break down the components of individual predictions.然后，我们学习了 SHAP 值，以分解单个预测的组成部分。
	Now we'll expand on SHAP values, seeing how aggregating many SHAP values can give more detailed alternatives to permutation importance and partial dependence plots.现在，我们将扩展 SHAP 值，了解将多个 SHAP 值聚合在一起如何为排列重要度图和部分依赖性图提供更详细的替代方案。

### 01 原理回顾

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071746722.png?imageSlim)
shap包中包含了两种可视化方案，在概念上和前面所学的permutation importance 和 partial dependence plot类似

### 02 Summary Plots（给出模型整体概览）

	置换重要度非常好，因为它创建了简单的数字度量，以了解哪些特征对模型很重要。这有助于我们轻松地进行特征之间的比较，而且您还可以向非专业观众展示由此生成的图表。

	但它并不能告诉您每个特征的重要性。如果某个特征的包络重要性处于中等水平，这可能意味着（1）它对少数预测有较大影响，但总体上没有影响，（2）或对所有预测都有中等影响。

通过 SHAP 汇总图，我们可以鸟瞰特征重要性及其驱动因素。我们将以足球数据为例进行分析：

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071749838.png?imageSlim)

==This plot is made of many dots. Each dot has three characteristics:==

- Vertical location shows what feature it is depicting垂直位置显示所描述的特征
- Color shows whether that feature was high or low for that row of the dataset颜色显示数据集中该行的特征值是大还是小（每一个点代表了数据集中某一行在该特征上的取值是大是小）
- Horizontal location shows whether the effect of that value caused a higher or lower prediction.水平位置显示该值的影响是导致预测值升高还是降低（也就是说，对于某一个点，她本身在该特征上的取值（颜色），最终导致预测的取值升高还是降低）

For example, the point in the upper left was for a team that scored few goals, reducing the prediction by 0.25.

==Some things you should be able to easily pick out:==

- The model ignored the `Red` and `Yellow & Red` features.模型忽略了红牌和黄红牌特征。
- Usually `Yellow Card` doesn't affect the prediction, but there is an extreme case where a high value caused a much lower prediction.通常黄牌不会影响预测结果，但有一个极端的例子，黄牌值高会导致预测结果大大降低
- High values of Goal scored caused higher predictions, and low values caused low predictions进球数的高值会导致较高的预测值，而低值则会导致较低的预测值。

If you look for long enough, there's a lot of information in this graph. You'll face some questions to test how you read them in the exercise.如果你观察的时间足够长，这张图中会有很多信息。在练习中，你会遇到一些问题来测试你是如何读懂它们的。

==代码如下：==

	导入数据集
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071800637.png?imageSlim)

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], val_X)
```

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071802565.png?imageSlim)

The code isn't too complex. But there are a few caveats.

- When plotting, we call `shap_values[1]`. For classification problems, there is a separate array of SHAP values for each possible outcome. In this case, we index in to get the SHAP values for the prediction of "True".绘图时，我们调用 shap_values[1]。对于分类问题，每个可能的结果都有一个单独的 SHAP 值数组。在这种情况下，我们通过索引来获取==预测结果为 "真 "的 SHAP 值==。
- Calculating SHAP values can be slow. It isn't a problem here, because this dataset is small. But you'll want to be careful when running these to plot with reasonably sized datasets. The exception is when using an `xgboost` model, which SHAP has some optimizations for and which is thus much faster.计算 SHAP 值可能会比较慢。由于数据集较小，在这里这不是问题。但在使用合理大小的数据集进行绘图时，您需要小心谨慎。使用 xgboost 模型是个例外，SHAP 对该模型进行了一些优化，因此速度要快得多。


	This provides a great overview of the model, but we might want to delve into a single feature. That's where SHAP dependence contribution plots come into play.这为我们提供了一个很好的模型概览，但我们可能想深入研究某个特征。这就是 SHAP 依赖性贡献图发挥作用的地方。


### 03 SHAP Dependence Contribution Plots（深入研究某个特征）

We've previously used Partial Dependence Plots to show how a single feature impacts predictions. These are insightful and relevant for many real-world use cases. Plus, with a little effort, they can be explained to a non-technical audience.我们之前曾使用部分依赖图（Partial Dependence Plots）来展示单一特征对预测的影响。这些都很有洞察力，而且与现实世界中的许多用例息息相关。此外，只需稍加努力，我们就能向非技术人员解释它们。

==But there's a lot they don't show. For instance, what is the distribution of effects? Is the effect of having a certain value pretty constant, or does it vary a lot depending on the values of other feaures. SHAP dependence contribution plots provide a similar insight to PDP's, but they add a lot more detail.但是，它们也有很多没有显示的地方。例如，效果的分布是怎样的？具有某个值的效果是非常恒定的，还是会因其他特征值的不同而变化很大。SHAP 依赖性贡献图提供了与 PDP 类似的洞察力，但它们增加了更多细节。==
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071807531.png?imageSlim)
Start by focusing on the shape, and we'll come back to color in a minute. Each dot represents a row of the data. The horizontal location is the actual value from the dataset, and the vertical location shows what having that value did to the prediction. The fact this slopes upward says that the more you possess the ball, the higher the model's prediction is for winning the Man of the Match award.首先==关注形状==，稍后我们再来看颜色。每个点代表一行数据。==水平位置是数据集中的实际值，垂直位置表示拥有该值对预测的影响。==这个向上倾斜的事实说明，你控球越多，模型对赢得比赛先生奖的预测就越高。

The spread suggests that other features must interact with Ball Possession %. For example, here we have highlighted two points with similar ball possession values. That value caused one prediction to increase, and it caused the other prediction to decrease.这种分布表明，==其他特征可能与控球率相互作用==。例如，在这里我们突出显示了两个控球率值相似的点。该值导致一个预测值上升，而另一个预测值下降。
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071811664.png?imageSlim)

For comparison, a simple linear regression would produce plots that are perfect lines, without this spread.相比之下，简单的线性回归绘制出的曲线图是完美的直线，没有这种差异。（如果是特征是线性回归的，就不会出现这种的差异）

This suggests we delve into the ==interactions==, and the plots include color coding to help do that. While the primary trend is upward, you can visually inspect whether that varies by dot color.这就需要我们深入研究交互作用，图中的颜色编码可以帮助我们做到这一点。虽然主要趋势是向上的，但您可以==通过点的颜色==直观地查看是否有变化。（看完形状看颜色）

Consider the following very narrow example for concreteness.请看下面这个非常狭窄的例子。
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071813514.png?imageSlim)
These two points stand out spatially as being far away from the upward trend. They are both colored purple, indicating the team scored one goal. You can interpret this to say **In general, having the ball increases a team's chance of having their player win the award. But if they only score one goal, that trend reverses and the award judges may penalize them for having the ball so much if they score that little.**
从空间上看，这两点与上升趋势相距甚远。它们都被染成紫色，表示该队进了一球。您可以将其理解为——==一般来说，拥有球权会增加球队球员获奖的机会（从倾斜的形状看出来的）==。但如果他们只进了一个球，这一趋势就会逆转，如果他们只进了那么一点点球，评委们可能会因为他们经常控球而惩罚他们。

==Outside of those few outliers, the interaction indicated by color isn't very dramatic here. But sometimes it will jump out at you.除了这几个异常值之外，用颜色表示的交互作用在这里并不十分显著。但有时它会让你眼前一亮。==


==代码==
	We get the dependence contribution plot with the following code. The only line that's different from the `summary_plot` is the last line.

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)

# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")
```
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403071820542.png?imageSlim)

If you don't supply an argument for `interaction_index`, Shapley uses some logic to pick one that may be interesting.如果你没有创造interaction_index的想法，Shapley会使用一些逻辑来选择一个可能有趣的论据。

这不需要编写大量代码。但这些技术的诀窍在于批判性地思考结果，而不是编写代码本身。

==是谁这么天才==