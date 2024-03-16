
Use Keras and Tensorflow to train your first neural network


## Introduction

在前两课中，我们学习了如何从密集层堆栈中构建全连接网络。刚开始创建时，网络的所有权重都是随机设置的--网络还什么都不 "知道"。在本课中，我们将了解如何训练神经网络；我们将了解神经网络是如何学习的。

与所有机器学习任务一样，我们从一组训练数据开始。训练数据中的每个示例都包含一些特征（输入）和一个预期目标（输出）。训练网络意味着调整其权重，使其能够将特征转化为目标。

以 80 种谷物数据集为例，我们需要一个网络，它能根据每种谷物的 "糖"、"纤维 "和 "蛋白质 "含量，预测出该谷物的 "卡路里"。如果我们能成功地训练一个网络来做到这一点，那么它的权重就必须以某种方式代表这些特征与训练数据中所表达的目标之间的关系。

In addition to the training data, we need two more things:

- A "loss function" that measures how good the network's predictions are.衡量网络预测结果好坏的 "损失函数"
- An "optimizer" that can tell the network how to change its weights.一个 "优化器"，告诉网络如何改变权重

---
## The Loss Function

We've seen how to design an architecture for a network, but we haven't seen how to tell a network what problem to solve. This is the job of the loss function.我们已经了解了如何设计网络架构，但还不知道如何告诉网络要解决什么问题。这就是损失函数的工作。

The loss function measures the disparity between the the target's true value and the value the model predicts.损失函数测量目标真实值与模型预测值之间的差距。

Different problems call for different loss functions. We have been looking at regression problems, where the task is to predict some numerical value -- calories in 80 Cereals, rating in Red Wine Quality. Other regression tasks might be predicting the price of a house or the fuel efficiency of a car.不同的问题需要不同的损失函数。我们一直在研究回归问题，其中的任务是预测一些数值--80 种谷物的卡路里，红葡萄酒质量的等级。其他回归任务可能是预测房子的价格或汽车的燃油效率。

A common loss function for regression problems is the mean absolute error or MAE. For each prediction y_pred, MAE measures the disparity from the true target y_true by an absolute difference abs(y_true - y_pred).回归问题常用的损失函数是平均绝对误差或 MAE。对于每个预测结果 y_pred，MAE 用绝对差值 abs（y_true - y_pred）来衡量与真实目标 y_true 之间的差距。

The total MAE loss on a dataset is the mean of all these absolute differences.数据集的总 MAE 损失是所有这些绝对差值的平均值。

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141431980.png?imageSlim)
Besides MAE, other loss functions you might see for regression problems are the mean-squared error (MSE) or the Huber loss (both available in Keras).除了 MAE，您可能会在回归问题上看到的其他损失函数还有均方误差 (MSE) 或 Huber 损失（Keras 中均有提供）

During training, the model will use the loss function as a guide for finding the correct values of its weights (lower loss is better). In other words, the loss function tells the network its objective.在训练过程中，模型将使用损失函数作为指导，找到正确的权重值（损失越小越好）。换句话说，损失函数告诉网络它的目标。

---
## The Optimizer - Stochastic Gradient Descent随机梯度下降

We've described the problem we want the network to solve, but now we need to say _how_ to solve it. This is the job of the **optimizer**. The optimizer is an algorithm that adjusts the weights to minimize the loss.我们已经描述了希望网络解决的问题，但==现在需要说明如何解决。这就是优化器的工作==。==优化器是一种调整权重以最小化损失的算法。==

Virtually all of the optimization algorithms used in deep learning belong to a family called **stochastic gradient descent**. They are iterative algorithms that train a network in steps. One **step** of training goes like this:深度学习中使用的几乎所有优化算法都属于随机梯度下降算法。它们是迭代算法，分步训练网络。其中一个训练步骤是这样的：

1. Sample some training data and run it through the network to make predictions.采样一些训练数据，并通过网络进行预测。
2. Measure the loss between the predictions and the true values.测量预测值与真实值之间的损失。
3. Finally, adjust the weights in a direction that makes the loss smaller.最后，调整权重，使损失变小

Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)然后反复这样做，直到损失小到你想要的程度（或者直到损失不再减少）。

![Fitting a line batch by batch. The loss decreases and the weights approach their true values.](https://storage.googleapis.com/kaggle-media/learn/images/rFI1tIk.gif)

Training a neural network with Stochastic Gradient Descent.

Each iteration's sample of training data is called a **minibatch** (or often just "batch"), while a complete round of the training data is called an **epoch**. The number of epochs you train for is how many times the network will see each training example.每次迭代的训练数据样本称为**minibatch**（或通常简称为 "批次"），而一轮完整的训练数据称为**epoch**。训练的epoch次数就是网络查看每个训练示例的次数。

The animation shows the linear model from Lesson 1 being trained with SGD. The pale red dots depict the entire training set, while the solid red dots are the minibatches. Every time SGD sees a new minibatch, it will shift the weights (`w` the slope and `b` the y-intercept) toward their correct values on that batch. Batch after batch, the line eventually converges to its best fit. You can see that the loss gets smaller as the weights get closer to their true values.动画展示了使用 SGD 训练第一课中的线性模型。淡红色点表示整个训练集，红色实心点表示小批量。每当 SGD 看到一个新的迷你批次，它就会将该批次的权重（`w`斜率和`b`y-截距）向正确值移动。一批又一批，这条线最终收敛到最佳拟合。可以看到，随着权重越来越接近真实值，损失也越来越小。

---
## Learning Rate and Batch Size

Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way). The size of these shifts is determined by the **learning rate**. A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.请注意，这条线在每个批次的方向上都只有很小的移动（而不是一直移动）。这些移动的大小由学习率决定。学习率越小，意味着网络在权重收敛到最佳值之前，需要看到更多的迷你批次。

The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds. Their interaction is often subtle and the right choice for these parameters isn't always obvious. (We'll explore these effects in the exercise.)==学习率和迷你批次的大小是对 SGD 训练过程影响最大的两个参数。它们之间的相互作用往往很微妙，正确选择这些参数并不总是显而易见的。(我们将在练习中探讨这些影响）==。

Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results. **Adam** is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.幸运的是，==对于大多数工作来说，不需要进行广泛的超参数搜索就能获得令人满意的结果。Adam 是一种 SGD 算法，它具有自适应学习率，无需调整参数就能解决大多数问题==（从某种意义上说，它是一种 "自调整 "算法）。Adam 是一款出色的通用优化器。

---
## Adding the Loss and Optimizer

After defining a model, you can add a loss function and optimizer with the model's `compile` method:

```python
model.compile(
    optimizer="adam",
    loss="mae",
)
```

Notice that we are able to specify the loss and optimizer with just a string. ==You can also access these directly through the Keras API -- if you wanted to tune parameters, for instance -- but for us, the defaults will work fine.==
请注意，我们只需使用一个字符串即可指定损耗和优化器。您也可以通过 Keras API 直接访问这些参数，例如，如果您想调整参数，但对我们来说，默认值就可以了。

**What's In a Name?**  
The **gradient** is a vector that tells us in what direction the weights need to go. More precisely, it tells us how to change the weights to make the loss change _fastest_. We call our process gradient **descent** because it uses the gradient to _descend_ the loss curve towards a minimum. **Stochastic** means "determined by chance." Our training is _stochastic_ because the minibatches are _random samples_ from the dataset. And that's why it's called SGD!
梯度是一个向量，它告诉我们权重需要向哪个方向移动。更准确地说，它告诉我们如何改变权重才能使损失变化最快。我们将这一过程称为梯度下降法，因为它利用梯度将损失曲线下降到最小值。随机的意思是 "由机会决定"。我们的训练是随机的，因为迷你批次是从数据集中随机抽取的样本。这就是它被称为 SGD 的原因！


---
## Example - Red Wine Quality

Now we know everything we need to start training deep learning models. So let's see it in action! We'll use the _Red Wine Quality_ dataset.

This dataset consists of physiochemical measurements from about 1600 Portuguese red wines. Also included is a quality rating for each wine from blind taste-tests. How well can we predict a wine's perceived quality from these measurements?该数据集包含约 1600 种葡萄牙红葡萄酒的生化测量数据。其中还包括盲品测试中对每种葡萄酒的质量评分。我们能从这些测量结果中预测出葡萄酒的感知质量吗？

We've put all of the data preparation into this next hidden cell. It's not essential to what follows so feel free to skip it. One thing you might note for now though is that we've rescaled each feature to lie in the interval [0,1][0,1]. As we'll discuss more in Lesson 5, neural networks tend to perform best when their inputs are on a common scale.它对接下来的内容并不重要，所以请随意跳过。==但有一点您可能会注意到，我们已将每个特征重新标定为位于区间 [0,1] 内。正如我们将在第 5 课中详细讨论的那样，当神经网络的输入在一个共同的范围内时，其性能往往最佳。==

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141724541.png?imageSlim)

How many inputs should this network have? We can discover this by looking at the number of columns in the data matrix. Be sure not to include the target (`'quality'`) here -- only the input features.

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141726476.png?imageSlim)

==Eleven columns means eleven inputs.==

We've chosen a three-layer network with over 1500 neurons. This network should be capable of learning fairly complex relationships in the data.

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141727698.png?imageSlim)

Deciding the architecture of your model should be part of a process. Start simple and use the validation loss as your guide. You'll learn more about model development in the exercises.==决定模型的结构应该是一个过程的一部分。从简单的开始，以验证损失为指导。您将在练习中学习到更多关于模型开发的知识。==

After defining the model, we compile in the optimizer and loss function.定义模型后，我们将编译优化器和损失函数。

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141729915.png?imageSlim)
Now we're ready to start the training! We've told Keras to feed the optimizer 256 rows of the training data at a time (the `batch_size`) and to do that 10 times all the way through the dataset (the `epochs`).现在我们准备开始训练！我们告诉 Keras 每次==向优化器输入 256 行训练数据（"batch_size"），并在整个数据集（"epochs"）中执行 10 次。==

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141730516.png?imageSlim)

You can see that Keras will keep you updated on the loss as the model trains.

Often, a better way to view the loss though is to plot it. The `fit` method in fact keeps a record of the loss produced during training in a `History` object. We'll convert the data to a Pandas dataframe, which makes the plotting easy==.通常，查看损失的更好方法是绘制损失图。事实上，"fit "方法会在一个 "History "对象中记录训练过程中产生的损失。我们将把数据转换为 Pandas 数据帧，这样就能轻松绘制曲线。==


```python
import pandas as pd

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();
```

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141732201.png?imageSlim)

Notice how the loss levels off as the epochs go by. When the loss curve becomes horizontal like that, it means the model has learned all it can and ==there would be no reason continue for additional epochs.==请注意随着时间的推移，损失是如何趋于平稳的。当损失曲线变成这样的水平时，说明模型已经学到了它所能学到的一切，==没有理由再继续学习更多的时间了。==