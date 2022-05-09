<h1 align="center">第五课第四周「Transformer」</h1>

# 笔记

## 目录

* [笔记](#笔记)
   * [目录](#目录)
   * [Transformer网络直观了解](#Transformer网络直观了解)
   * [自注意力](#自注意力)
   * [Multi-Head注意力](#Multi-Head注意力)
   * [Transformer网络](#Transformer网络)

## Transformer网络直观了解

今年来的研究中，Transformer这个结构彻底改变了NLP领域，当前绝大多数高效的NLP算法都是基于Transformer结构。整体上这是一个相对复杂的网络结构。

我们的时序模型的复杂度会随着序列任务的复杂度的增加而增加。我们最早引入的是RNN结构，但是由于存在当输入较长依赖的序列后会出现梯度消失的问题，我们就引入了GRU，之后是LSTM模型。这些模型引入了一些**“门”**的概念来控制信息流。同样每一个单元都会耗费一些计算压力，因此每一个模型的计算复杂度都在逐步递增。而在Transformer结构中，我们会把对于整个序列的这一系列计算并行进行，因此可以一次性提取整个句子，而不是从左往右逐个计算。

![](https://raw.githubusercontent.com/kakack/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week4/md_images/01.png)

Transformer 架构的主要创新是结合了基于注意力的表示和 CNN 卷积神经网络处理方式的使用。也就是将原本基于时序的序列能像在CNN中计算所有像素组成的矩阵一样并行计算。在Transformer结构中有两个重要概念：

 -  自注意力：假如有一个由五个单词组成的句子，最终会为这五个单词计算五中表示：$A_1,A_2,A_3,A_4,A_5$。这是一种基于注意力的方法，可以并行计算句子中所有单词的表示。

- Multi-Head注意力：这是一种对于自注意力过程的全遍历，最终得到多种版本的表示方法。

  最终这些丰富的表示方法能帮助创建非常高效的机器翻译或者其他NLP算法。



## 自注意力

之前已经见识过如何将注意力机制用于像RNN这样的序列模型。为了像使用CNN一样使用注意力机制，需要首先计算自注意力，也就是为输入句子的每一个单词创建基于注意力的表示方式。比如在例子中：Jane visite l'Afrique en septembre.将句子中五个单词表达为$A_1,A_2,A_3,A_4,A_5$。

比如对于单词l'Afrique或者说Africa，我们如何确定这个词在整个句子中真正的含义，是指一个大洲或者一个目的地或者一个历史名胜，这也是表达式$A_3$需要达到的目的，它会审视上下文确定该词真正表达的含义。

在之前的RNN中，注意力机制的表达式是这样：

$$\alpha^{<t, t'>}=\frac{exp(e^{<t,t'>})}{\sum^{T_x}_{t'=1}exp(e^{<t,t'>})}$$

而在Transformer注意力中表达式是这样：

$$A(q,K,V)=\sum_i\frac{exp(e^{<q \cdot k^{<i>}>})}{\sum_j exp(e^{<q \cdot k^{<j>}>})}v^{<i>}$$或者：$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

对于一个单词，如l'Afrique，会有三个值：$q^{<i>}$（query）、$K^{<i>}$（key）和$V^{<i>}$（value），这些向量是用来计算每个单词的注意力值的。继续以l'Afrique为例，则此时$i=3$。

假设有$x^{<3>}$是l'Afrique的嵌入矩阵，那么：

$$q^{<3>}=W^Q \cdot x^{<3>}$$

$$k^{<3>}=W^K\cdot x^{<3>}$$

$$v^{<3>}=W^V\cdot x^{<3>}$$

其中$W^Q$、$W^K$和$W^V$是算法的参数。计算得到的这些值广义上可以看做类似在数据库冲查询得到key-val键值对结果的过程。$q^{<3>}$就像是发起了一个提问：什么是l'Afrique？

![](https://raw.githubusercontent.com/kakack/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week4/md_images/02.png)

接下去要做的就是将$q^{<3>}$和$k^{<1>}$、$k^{<2>}$、$k^{<3>}$和$k^{<4>}$做内积，来尝试回答什么是l'Afrique这个问题。这一系列操作的目的就是帮助我们提取计算$A^{<3>}$所需的最有用的信息。假设我们已知$k^{<1>}$表示的是一个人名，$k^{<2>}$表示的是visit这个动作，那么$k^{<3>}$最有可能表示visit的目的地。

最后把得到的softmax内积跟各个$v^{<i>}$相乘再相加求和，得到最终的$A^{<3>}$，也可以表达成$A(q^{<3>},K,V)$。

这个做法最大的优势是，l'Afrique这个词不再是某一个预设好的embedding，而是让自注意力机制认识到l'Afrique是一个visit的目的地，从而计算出更丰富更有用的表达形式。同样可以用这个方法计算句子中其他所有词的注意力值并组成一个大矩阵。

这类注意力另一个名称是scaled dot-product attention，因为其中分母是scale the dot-product，因此不会爆炸。



## Multi-Head注意力

Multi-Head注意力就是自注意力的full loop，我们把在一个序列上计算一次自注意力称为a head。在multi-head注意力里，将同样的query、key和value作为输入。

![](https://raw.githubusercontent.com/kakack/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week4/md_images/03.png)

首先将q、k和v跟对应的权重矩阵相乘，即得到$W^Q_1q^{<i>}$、$W^K_1k^{<i>}$和$W^V_1v^{<i>}$也就是第$i$个单词的一组新的query、key和value。把整个序列中所有单词都计算一遍，得到了第一个head，如同上文中自注意力中所做的一样，于是我们得到了第一个head。第一个head可能回答了问题：What's happenning。之后计算第二个head，用到的权重矩阵变成了$W^Q_2q^{<i>}$、$W^K_2k^{<i>}$和$W^V_2v^{<i>}$，后续计算方法同上。第二个head可能回答了问题：When is something happenning。以此类推，我们通常用$h$表示head个数。

我们可以把每个head当做一个不同的特征，当你将这些特征传递给一个新的网络时，你可以计算出非常丰富的句子表示。

## Transformer网络

在学习了自注意力和multi-head注意力后，我们将它们组合到一起成为一个Transformer架构来完成从法语到英语的翻译功能。

再次用Jane visite l'Afrique en septembre这句话举例。并且加入了句子开头符$SOS$和句子终结符$EOS$。

![](https://raw.githubusercontent.com/kakack/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week4/md_images/04.png)



第一步，将这个句子中的embedding喂入一个拥有multi-head注意力的encoder block（Q、K、V），计算embedding和权重矩阵乘积并在网络中向前传递。在论文中这个encoder block会重复N次，通常N会取6。之后将输出输入到decoder block中。Decoder block的任务是输出英语翻译结果。第一个block为下一个注意力block输出Q矩阵，而encoder的输出则用于生成K和V。第二层的主要作用是预测句子中下一个输出的词。

额外补充：

- 输入的位置编码：在之前的注意力表达式中，单词在句子中的位置并没有体现，但位置信息在翻译中却是十分重要的。所以表示位置信息的方法是使用一个$sin$和$cos$等式的组合。（例子见图中蓝色部分）

  $$PE_{pos, 2i}=sin\frac{pos}{1000^{\frac{2i}{d}}}$$

  $$PE_{pos, 2i+1}=cos\frac{pos}{1000^{\frac{2i}{d}}}$$

  

  Mask multi-head Attention只在训练时候重要。在训练过程中会屏蔽句子最后一部分内容以模仿网络在测试时或预测期间需要做什么。换句话说，Mask multi-head Attention所做的只是反复假装网络已经完美翻译。说出前几个词并隐藏剩余的词，看看是否给出了完美的第一部分翻译，神经网络是否能准确预测序列中的下一个词。
