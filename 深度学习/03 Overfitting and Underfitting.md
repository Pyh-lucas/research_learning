Improve performance with ==extra capacity or early stopping==.通过额外的容量或提前停机来提高性能。

## Introduction

Recall from the example in the previous lesson that Keras will keep a history of the training and validation loss over the epochs that it is training the model. In this lesson, we're going to learn how to interpret these learning curves and how we can use them to guide model development. In particular, we'll examine at the learning curves for evidence of _underfitting_ and _overfitting_ and look at a couple of strategies for correcting it.==回顾上一课的示例，Keras 会在训练模型的历时中保留训练和验证损失的历史记录。在本课中，我们将学习如何解释这些学习曲线，以及如何使用它们来指导模型开发。特别是，我们将研究学习曲线，找出不充分拟合和过度拟合的证据，并研究几种纠正策略。==

- 使用拟合记录来解释学习曲线，从而指导模型开发
- 找出欠拟合&过拟合的证据，进行纠正

---
## Interpreting the Learning Curves解释学习曲线

You might think about the information in the training data as being of two kinds: _signal_ and _noise_. The signal is the part that generalizes, the part that can help our model make predictions from new data. The noise is that part that is _only_ true of the training data; the noise is all of the random fluctuation that comes from data in the real-world or all of the incidental, non-informative patterns that can't actually help the model make predictions. The noise is the part might look useful but really isn't.==你可以把训练数据中的信息分为两种：信号和噪音。==信号是可以泛化的部分，是可以帮助我们的模型对新数据进行预测的部分。噪音则是仅适用于训练数据的部分；噪音是来自真实世界数据的所有随机波动，或者是所有偶然出现的、无信息的模式，它们实际上无法帮助模型进行预测。噪音是那些看起来有用但实际上没用的部分。

We train a model by choosing weights or parameters that minimize the loss on a training set. You might know, however, that to accurately assess a model's performance, we need to evaluate it on a new set of data, the _validation_ data. (You could see our lesson on [model validation](https://www.kaggle.com/dansbecker/model-validation) in _Introduction to Machine Learning_ for a review.)我们通过选择权重或参数来训练模型，使训练集上的损失最小化。不过，你可能知道，为了准确评估模型的性能，我们需要在一组新数据（即验证数据）上对其进行评估。

When we train a model we've been plotting the loss on the training set epoch by epoch. To this we'll add a plot the validation data too. These plots we call the **learning curves**. To train deep learning models effectively, we need to be able to interpret them.==当我们训练一个模型时，我们会在训练集上逐次绘制损失图。在此基础上，我们还将添加验证数据图。==这些图我们称之为学习曲线。==为了有效地训练深度学习模型，我们需要能够解释它们==。

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141741679.png?imageSlim)

	Validation的下降说明模型学到了Signal，差距的大小代表学到了多少噪声
Now, the training loss will go down either when the model learns signal or when it learns noise. But the validation loss will go down only when the model learns signal. (Whatever noise the model learned from the training set won't generalize to new data.) So, when a model learns signal both curves go down, but when it learns noise a _gap_ is created in the curves. The size of the gap tells you how much noise the model has learned.==当模型学习到信号或噪声时，训练损失都会减少。但只有当模型学习到信号时，验证损失才会减少。==(无论模型从训练集中学到了什么噪音，都不会泛化到新数据上）。==因此，当模型学习到信号时，两条曲线都会下降，但当模型学习到噪声时，两条曲线就会出现差距。差距的大小可以告诉你模型学到了多少噪声==。

Ideally, we would create models that learn all of the signal and none of the noise. This will practically never happen. Instead we make a trade. We can get the model to learn more signal at the cost of learning more noise. So long as the trade is in our favor, the validation loss will continue to decrease. After a certain point, however, the trade can turn against us, the cost exceeds the benefit, and the validation loss begins to rise.==理想的情况是，我们创建的模型能学习到所有信号，而没有噪音。这实际上永远不会发生。==相反，我们要进行平衡。我们可以让模型学习更多信号，但代价是学习更多噪音。只要交易对我们有利，验证损失就会继续减少。==但到了一定程度后，交易可能会对我们不利，成本超过收益，验证损失就会开始上升。==

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141753665.png?imageSlim)
This trade-off indicates that there can be two problems that occur when training a model: not enough signal or too much noise. **Underfitting** the training set is when the loss is not as low as it could be because the model hasn't learned enough _signal_. **Overfitting** the training set is when the loss is not as low as it could be because the model learned too much _noise_. The trick to training deep learning models is finding the best balance between the two.这种权衡表明，在训练模型时可能会出现两个问题：信号不足或噪音过多。训练集不充分是指由于模型没有学习到足够的信号，导致损失没有那么低。过度拟合训练集是指由于模型学习了太多噪音，导致损失没有那么低。==训练深度学习模型的诀窍在于找到两者之间的最佳平衡点。==

**We'll look at a couple ways of getting more signal out of the training data while reducing the amount of noise
我们将研究几种既能从训练数据中获得更多信号，又能减少噪音的方法。**

## Capacity 通过加宽/加深拓展学习能力

A model's **capacity** refers to the size and complexity of the patterns it is able to learn. For neural networks, this will largely be determined by how many neurons it has and how they are connected together. If it appears that your network is underfitting the data, you should try increasing its capacity.模型的容量指的是它能够学习的模式的大小和复杂程度。==对于神经网络来说，这主要取决于它有多少个神经元以及这些神经元是如何连接在一起的==。==如果你的网络似乎**对数据拟合不足**，你应该尝试增加它的容量。==

You can increase the capacity of a network either by making it _wider_ (more units to existing layers) or by making it _deeper_ (adding more layers). Wider networks have an easier time learning more linear relationships, while deeper networks prefer more nonlinear ones. Which is better just depends on the dataset您可以通过加宽网络（在现有层中加入更多单元）或加深网络（增加更多层）来提高网络容量。==更宽的网络更容易学习更多线性关系，而更深的网络则更喜欢非线性关系。孰优孰劣取决于数据集==

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141757120.png?imageSlim)

## Early Stopping

We mentioned that when a model is too eagerly learning noise, the validation loss may start to increase during training. To prevent this, we can simply stop the training whenever it seems the validation loss isn't decreasing anymore. Interrupting the training this way is called **early stopping**.我们提到过，==当模型过于急切地学习噪声时，验证损失可能会在训练过程中开始增加。为了避免这种情况，我们可以在验证损失似乎不再减少时停止训练。这种中断训练的方法称为**早期停止**。==

![](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141759180.png?imageSlim)

Once we detect that the validation loss is starting to rise again, we can reset the weights back to where the minimum occured. This ensures that the model won't continue to learn noise and overfit the data.==一旦检测到验证损失开始回升，我们就可以将权重重置回最小值。这样就能确保模型不会继续学习噪声和过拟合数据。==

Training with early stopping also means we're in less danger of stopping the training too early, before the network has finished learning signal. So besides preventing overfitting from training too long, early stopping can also prevent _underfitting_ from not training long enough. Just set your training epochs to some large number (more than you'll need), and early stopping will take care of the rest.提前停止训练还意味着，在网络完成信号学习之前，我们不会过早停止训练。==因此，除了防止训练时间过长导致的过拟合，提前停止训练还能防止训练时间不够导致的欠拟合==。==**只需将训练历元设置为某个较大的数字（比你需要的更多）**，早期停止就能解决剩下的问题==。

### Adding Early Stopping

In Keras, we include early stopping in our training through a callback. ==A **callback** is just a function you want run every so often while the network trains. ==The early stopping callback will run after every epoch. (Keras has [a variety of useful callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) pre-defined, but you can [define your own](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback), too.)
在 Keras 中，我们通过回调将早期停止纳入训练。回调**只是在网络训练时希望每隔一段时间运行的函数。早期停止回调将在每个epoch后运行。(Keras 预先定义了[各种有用的回调]，但你也可以[定义自己的回调]）

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)
```

These parameters say: "If there hasn't been at least an improvement of 0.001 in the validation loss over the previous 20 epochs, then stop the training and keep the best model you found." It can sometimes be hard to tell if the validation loss is rising due to overfitting or just due to random batch variation. The parameters allow us to set some allowances around when to stop==.这些参数表示 "如果验证损失比前 20 个 epoch 没有至少 0.001 的改进，那么停止训练，保留找到的最佳模型"。有时很难判断验证损失的增加是由于过度拟合还是随机批次变化造成的。我们可以通过参数设置来确定何时停止训练。==

As we'll see in our example, we'll pass this callback to the `fit` method along with the loss and optimizer.正如我们将在示例中看到的，我们将把这个回调与损失和优化器一起传递给 `fit` 方法


## Example - Train a Model with Early Stopping

Let's continue developing the model from the example in the last tutorial. We'll increase the capacity of that network but also add an early-stopping callback to prevent overfitting.在原有模型中加入及时停止的call back来阻止过拟合

Here's the data prep again

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141858475.png?imageSlim)

Now let's increase the capacity of the network. We'll go for a fairly large network, but rely on the callback to halt the training once the validation loss shows signs of increasing.现在我们来增加网络的容量。我们将使用一个相当大的网络，但一旦验证损失有增加的迹象，就依靠回调来停止训练。

```python
from tensorflow import keras
from tensorflow.keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
```

After defining the callback, add it as an argument in `fit` (you can have several, so put it in a list). Choose a large number of epochs when using early stopping, more than you'll need.定义回调后，将其作为参数添加到 `fit` 中（可以有多个参数，因此将其放在一个列表中）。在使用早期停止时，选择较大的epoch数，比你需要的更多。

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
```

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141902826.png?imageSlim)

And sure enough, Keras stopped the training well before the full 500 epochs!果不其然，Keras 在 500 个epochs 之前就停止了训练！