
Add hidden layers to your network to uncover complex relationships.

## Introduction

In this lesson we're going to see how we can build neural networks capable of learning the complex kinds of relationships deep neural nets are famous for.在本课中，我们将了解如何构建神经网络，==使其能够学习深度神经网络所擅长的复杂关系==。

The key idea here is _modularity_, building up a complex network from simpler functional units. We've seen how a linear unit computes a linear function -- now we'll see how to combine and modify these single units to model more complex relationships.这里的关键概念是模块化，即从较简单的功能单元构建复杂的网络。我们已经了解了线性单元是如何计算线性函数的，现在我们来看看如何组合和修改这些单个单元，以建立更复杂的关系模型。

## Layers

Neural networks typically organize their neurons into **layers**. When we collect together linear units having a common set of inputs we get a **dense** layer.神经网络通常将神经元组织成**层**。当我们把具有一组共同输入的线性单元集合在一起时，我们就得到了一个**密集**层。

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141339951.png?imageSlim)

You could think of each layer in a neural network as performing some kind of relatively simple transformation. Through a deep stack of layers, a neural network can transform its inputs in more and more complex ways. In a well-trained neural network, each layer is a transformation getting us a little bit closer to a solution.你可以把神经网络中的每一层都看作是在进行某种相对简单的转换。通过层层深入，神经网络能以越来越复杂的方式转换输入。在训练有素的神经网络中，每一层都是一次变换，让我们离解决方案更近一点

> **Many Kinds of Layers**  
> A "layer" in Keras is a very general kind of thing. A layer can be, essentially, any kind of _data transformation_. Many layers, like the [convolutional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) and [recurrent](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN) layers, transform data through use of neurons and differ primarily in the pattern of connections they form. Others though are used for [feature engineering](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) or just [simple arithmetic](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add). There's a whole world of layers to discover -- [check them out](https://www.tensorflow.org/api_docs/python/tf/keras/layers)!
> Keras 中的 "层 "是一种非常通用的东西。从本质上讲，层可以是任何一种数据转换。许多层，如[卷积]层和[递归]层，通过使用神经元来转换数据，其主要区别在于它们形成的连接模式。其他层则用于[特征工程]或仅仅是[简单运算]


## The Activation Function

It turns out, however, that two dense layers with nothing in between are no better than a single dense layer by itself. Dense layers by themselves can never move us out of the world of lines and planes. What we need is something _nonlinear_. What we need are activation functions.然而，事实证明，两个中间空无一物的密集层并不比一个单独的密集层更好。==密集层本身永远无法让我们走出线条和平面的世界。我们需要的是非线性的东西。==我们需要的是激活函数。

![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141344708.png?imageSlim)

An **activation function** is simply some function we apply to each of a layer's outputs (its _activations_). The most common is the _rectifier_ function max(0,x).所谓**激活函数**，就是我们应用于层的每个输出（其_激活_）的函数。最常见的是整型函数 max(0,x)

![A graph of the rectifier function. The line y=x when x>0 and y=0 when x<0, making a 'hinge' shape like '_/'.](https://storage.googleapis.com/kaggle-media/learn/images/aeIyAlF.png)

The rectifier function has a graph that's a line with the negative part "rectified" to zero. Applying the function to the outputs of a neuron will put a _bend_ in the data, moving us away from simple lines.整流函数的图形是一条将负部分 "整流 "为零的直线。将该函数应用于神经元的输出时，数据会发生弯曲，从而脱离简单的直线。

When we attach the rectifier to a linear unit, we get a **rectified linear unit** or **ReLU**. (For this reason, it's common to call the rectifier function the "ReLU function".) Applying a ReLU activation to a linear unit means the output becomes `max(0, w * x + b)`, which we might draw in a diagram like:当我们将整流器连接到线性单元时，就得到了整流线性单元或 ReLU。(因此，我们通常称整流函数为 "ReLU 函数"）。将 ReLU 激活应用于线性单元，意味着输出变为 max(0，w * x + b)，我们可以将其绘制成类似的图表：
![image.png](https://peiyihan-1324725457.cos.ap-beijing.myqcloud.com/Obsidian/202403141409158.png?imageSlim)
## Stacking Dense Layers

Now that we have some nonlinearity, let's see how we can stack layers to get complex data transformations.

![An input layer, two hidden layers, and a final linear layer.](https://storage.googleapis.com/kaggle-media/learn/images/Y5iwFQZ.png)

A stack of dense layers makes a "fully-connected" network.密集层的堆叠构成了一个 "全连接 "网络。

The layers before the output layer are sometimes called **hidden** since we never see their outputs directly.输出层之前的层有时被称为 "隐藏层"，因为我们无法直接看到它们的输出。

Now, notice that the final (output) layer is a linear unit (meaning, no activation function). That makes this network appropriate to a regression task, where we are trying to predict some arbitrary numeric value. Other tasks (like classification) might require an activation function on the output.现在，请注意最后（输出）层是一个线性单元（即没有激活函数）。这使得该网络适用于回归任务，即我们试图预测某个任意数值。其他任务（如分类）可能需要在输出上使用激活函数。

## Building Sequential Models

The `Sequential` model we've been using will connect together a list of layers in order from first to last: the first layer gets the input, the last layer produces the output. This creates the model in the figure above:我们一直在使用的 "顺序 "模型会按照从第一层到最后一层的顺序将一系列层连接起来：第一层获取输入，最后一层产生输出。这就创建了上图中的模型：

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```

Be sure to pass all the layers together in a list, like `[layer, layer, layer, ...]`, instead of as separate arguments. To add an activation function to a layer, just give its name in the `activation` argument.
请务必将所有层以列表形式（如 [layer, layer, layer, ...]）传递，而不是以单独参数的形式传递。要为某个层添加激活函数，只需在激活参数中给出函数名称即可。