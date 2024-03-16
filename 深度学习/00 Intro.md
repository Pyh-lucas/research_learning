[Learn Intro to Deep Learning | Kaggle](https://www.kaggle.com/learn/intro-to-deep-learning)

	Deep learning is an approach to machine learning characterized by deep stacks of computations. This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and hierarchical patterns found in the most challenging real-world datasets.深度学习是一种以深度计算堆栈为特征的机器学习方法。正是这种计算深度使得深度学习模型能够在最具挑战性的现实世界数据集中发现各种复杂的分层模式。

	凭借其强大的功能和可扩展性，神经网络已成为深度学习的定义模型。神经网络由神经元组成，每个神经元只能单独进行简单的计算。神经网络的强大功能来自于这些神经元所能形成的复杂连接。

## The Linear Unit

So let's begin with the fundamental component of a neural network: the individual neuron. As a diagram, a **neuron** (or **unit**) with one input looks like:

![Diagram of a linear unit.](https://storage.googleapis.com/kaggle-media/learn/images/mfOlDR6.png)

The Linear Unit: y=wx+b

The input is `x`. Its connection to the neuron has a **weight** which is `w`. Whenever a value flows through a connection, you multiply the value by the connection's weight. For the input `x`, what reaches the neuron is `w * x`. A neural network "learns" by modifying its weights.

The `b` is a special kind of weight we call the **bias**. The bias doesn't have any input data associated with it; instead, we put a `1` in the diagram so that the value that reaches the neuron is just `b` (since `1 * b = b`). The bias enables the neuron to modify the output independently of its inputs.

The `y` is the value the neuron ultimately outputs. To get the output, the neuron sums up all the values it receives through its connections. This neuron's activation is `y = w * x + b`, or as a formula y=wx+b

	虽然单个神经元通常只能作为更大网络的一部分发挥作用，但以单个神经元模型作为基线开始通常是有用的。单神经元模型是线性模型。


## Multiple Inputs

The _80 Cereals_ dataset has many more features than just `'sugars'`. What if we wanted to expand our model to include things like fiber or protein content? That's easy enough. We can just add more input connections to the neuron, one for each additional feature. To find the output, we would multiply each input to its connection weight and then add them all together.

A linear unit with three inputs.

The formula for this neuron would be y=w0x0+w1x1+w2x2+b. A linear unit with two inputs will fit a plane, and a unit with more inputs than that will fit a hyperplane.两个输入会拟合一个平面，更多的输入将会拟合一个超平面

## Linear Units in Keras

The easiest way to create a model in Keras is through `keras.Sequential`, which creates a neural network as a stack of _layers_. We can create models like those above using a _dense_ layer (which we'll learn more about in the next lesson).在 Keras 中创建模型的最简单方法是通过 `keras.Sequential`，它将神经网络创建为_layer_的堆栈。我们可以使用_密集_层（我们将在下一课详细了解）创建类似上述的模型。

We could define a linear model accepting three input features (`'sugars'`, `'fiber'`, and `'protein'`) and producing a single output (`'calories'`) like so:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

With the first argument, `units`, we define how many outputs we want. In this case we are just predicting `'calories'`, so we'll use `units=1`.通过第一个参数 "单位"，我们可以定义需要多少输出。在本例中，我们只预测 `'calories'`，所以使用 `units=1`。

With the second argument, `input_shape`, we tell Keras the dimensions of the inputs. Setting `input_shape=[3]` ensures the model will accept three features as input (`'sugars'`, `'fiber'`, and `'protein'`).通过第二个参数 "input_shape"，我们可以告诉 Keras 输入的尺寸。设置 `input_shape=[3]` 可以确保模型接受三个特征作为输入（"糖"、"纤维 "和 "蛋白质"）。

This model is now ready to be fit to training data!

>**Why is `input_shape` a Python list?**  
	The data we'll use in this course will be tabular data, like in a Pandas dataframe. We'll have one input for each feature in the dataset. The features are arranged by column, so we'll always have `input_shape=[num_columns]`. The reason Keras uses a list here is to permit use of more complex datasets. Image data, for instance, might need three dimensions: `[height, width, channels]`.
	我们在本课程中使用的数据将是表格数据，就像 Pandas 数据框中的数据一样。我们将为数据集中的每个特征设置一个输入。特征按列排列，因此我们将始终使用 `input_shape=[num_columns]`。Keras 在此使用列表的原因是允许使用更复杂的数据集。例如，图像数据可能需要三个维度： [高度、宽度、通道]`。