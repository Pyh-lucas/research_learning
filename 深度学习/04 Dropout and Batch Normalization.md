
Add these special layers to prevent overfitting and stabilize training.添加特殊层可防止过度拟合并稳定训练

## Introduction

There's more to the world of deep learning than just dense layers. There are dozens of kinds of layers you might add to a model. (Try browsing through the [Keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/) for a sample!) Some are like dense layers and define connections between neurons, and others can do preprocessing or transformations of other sorts.**==深度学习的世界不仅仅只有密集层。你可以在模型中添加几十种层。(试着浏览 Keras 文档以获取示例！）有些层与密集层类似，定义神经元之间的连接，有些层则可以进行预处理或其他类型的转换。==**

In this lesson, we'll learn about a two kinds of special layers, not containing any neurons themselves, but that add some functionality that can sometimes benefit a model in various ways. Both are commonly used in modern architectures在本课中，我们将学习==两种特殊层，它们本身不包含任何神经元，但可以添加一些功能，有时可以以各种方式使模型受益==。这两种层在现代架构中都很常用。

---
## Dropout

The first of these is the "dropout layer", which can help correct overfitting.

In the last lesson we talked about how overfitting is caused by the network learning spurious patterns in the training data. To recognize these spurious patterns a network will often rely on very a specific combinations of weight, a kind of "conspiracy" of weights. Being so specific, they tend to be fragile: remove one and the conspiracy falls apart.在上一课中，我们谈到过拟合是如何由网络学习训练数据中的虚假模式造成的。==为了识别这些虚假模式，网络通常会依赖于非常特殊的权重组合，即权重的一种 "阴谋"。由于权重组合非常特殊，它们往往很脆弱：去掉一个权重组合，整个 "阴谋 "就会分崩离析。==

This is the idea behind **dropout**. To break up these conspiracies, we randomly _drop out_ some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious patterns in the training data. Instead, it has to search for broad, general patterns, whose weight patterns tend to be more robust.这就是 **dropout** 背后的理念。为了打破这些阴谋，我们在==每一步训练中都会随机剔除某一层的部分输入单元，从而使网络更难学习训练数据中的虚假模式。取而代之的是，它必须寻找宽泛、一般的模式，其权重模式往往更加稳健。==

![An animation of a network cycling through various random dropout configurations.](https://storage.googleapis.com/kaggle-media/learn/images/a86utxY.gif)

Here, 50% dropout has been added between the two hidden layers.在这里，两个隐藏层之间添加了 50% 的滤波。（动图每次只选2个神经元）

You could also think about dropout as creating a kind of _ensemble_ of networks. The predictions will no longer be made by one big network, but instead by a committee of smaller networks. Individuals in the committee tend to make different kinds of mistakes, but be right at the same time, making the committee as a whole better than any individual. (If you're familiar with random forests as an ensemble of decision trees, it's the same idea.)你也可以把 "滤除 "看作是创建一种网络的 "组合"。预测将不再由一个大网络做出，而是由一个由较小网络组成的委员会做出。委员会中的个体往往会犯不同类型的错误，但同时又是正确的，这就使得委员会作为一个整体比任何个体都要好。(如果你熟悉作为决策树集合的随机森林，这也是同样的道理）。

### Adding Dropout

In Keras, the dropout rate argument `rate` defines what percentage of the input units to shut off. Put the `Dropout` layer just before the layer you want the dropout applied to:在 Keras 中，滤除率参数 rate 定义了==要关闭的输入单元的百分比==。==将 "滤除 "层放在要应用滤除的层==之前：

``` python
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```

---
## Batch Normalization

The next special layer we'll look at performs "batch normalization" (or "batchnorm"), which can help correct training that is slow or unstable.下一个特殊层将执行 "批量规范化"（或 "批量规范"），它可以帮助纠正缓慢或不稳定的训练。

With neural networks, it's generally a good idea to put all of your data on a common scale, perhaps with something like scikit-learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). The reason is that SGD will shift the network weights in proportion to how large an activation the data produces. Features that tend to produce activations of very different sizes can make for unstable training behavior==.对于神经网络来说，将所有数据放在一个统一的比例尺上通常是个好主意==，或许可以使用类似 scikit-learn 的 StandardScaler 或 MinMaxScaler 这样的工具。原因在于，SGD 会根据数据产生的激活程度按比例移动网络权重。往往会产生不同大小激活的特征会导致不稳定的训练行为。

Now, if it's good to normalize the data before it goes into the network, maybe also normalizing inside the network would be better! In fact, we have a special kind of layer that can do this, the **batch normalization layer**. A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, and then also putting the data on a new scale with two trainable rescaling parameters. Batchnorm, in effect, performs a kind of coordinated rescaling of its inputs.现在，如果在数据进入网络之前对其进行归一化处理是件好事，==那么在网络内部进行归一化处理也许会更好！事实上，我们有一种特殊的层可以做到这一点，即批量归一化层==。**批量归一化层会在每个批次数据进入网络时对其进行处理，首先用其自身的平均值和标准差对批次数据进行归一化处理，然后用两个可训练的重缩放参数将数据放到一个新的刻度上**。批正则实际上是对其输入数据进行一种协调的重定标。

Most often, batchnorm is added as an ==aid to the optimization process (though it can sometimes also help prediction performance)==. Models with batchnorm tend to need fewer epochs to complete training. Moreover, batchnorm can also fix various problems that can cause the training to get "stuck". Consider adding batch normalization to your models, especially if you're having trouble during training.通常，==批正值是作为优化过程的辅助工具添加的（尽管有时它也能帮助提高预测性能）==。使用批规范的模型往往需要更少的历时来完成训练。此外，批归一化还能解决可能导致训练 "卡壳 "的各种问题。请考虑在您的模型中添加批处理归一化，尤其是在训练过程中遇到困难时。

### Adding Batch Normalization 放批量归一化的三个位置

It seems that batch normalization can be used at almost any point in a network. You can put it after a layer...

```python
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),
```

... or between a layer and its activation function:

```python
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),
```

And if you add it as the first layer of your network it can act as a kind of adaptive preprocessor, standing in for something like Sci-Kit Learn's `StandardScaler`.==如果把它添加到网络的第一层，它就可以充当一种自适应预处理器，替代 Sci-Kit Learn 的 StandardScaler。==


## Example - Using Dropout and Batch Normalization

Let's continue developing the _Red Wine_ model. Now we'll increase the capacity even more, but add dropout to control overfitting and batch normalization to speed up optimization. This time, we'll also leave off standardizing the data, to demonstrate how batch normalization can stabalize the training.让我们继续开发红葡萄酒模型。现在，==我们将进一步提高容量，但会增加 dropout 以控制过拟合，并增加批量归一化以加快优化速度==。**这一次，我们还将不对数据进行标准化，以演示批量归一化如何稳定训练。**

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141937423.png?imageSlim)

==When adding dropout, you may need to increase the number of units in your `Dense` layers.==

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```

There's nothing to change this time in how we set up the training.

```python
model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=100,
    verbose=0,
)

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
```

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141940580.png?imageSlim)

You'll typically get better performance if you standardize your data before using it for training. That we were able to use the raw data at all, however, shows how effective batch normalization can be on more difficult datasets
如果在使用数据进行训练前对其进行标准化处理，通常会获得更好的性能。不过，我们能够使用原始数据，说明批量标准化在更困难的数据集上是多么有效