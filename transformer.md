# Transformer

## self-attention

输入输出和RNN是类似的，输入一个sequence，输出一个sequence；

但是self-attention的每个输出都是与每个输入有关的，RNN输出只和当前时间前的输入有关（双向RNN貌似可以解决这个问题）；

而且self-attention可以实现并行计算较于RNN；

<img src="https://pic2.zhimg.com/80/v2-e3ef96ccae817226577ee7a3c28fa16d_720w.jpg" alt="img" style="zoom:50%;" />



<img src="https://pic3.zhimg.com/80/v2-8537e0996a586b7c37d5e345b6c4402a_720w.jpg" alt="img" style="zoom:50%;" />

输入是x1 - x4一个sequence，首先×矩阵W得到embedding a1 - a4；接着进入attention层，每个embedding × Wq得到qi，embedding × Wk得到ki，embedding × Wv得到vi；从而每个输入sequence元素被分裂为三个向量；

<img src="https://pic4.zhimg.com/80/v2-197b4f81d688e4bc40843fbe41c96787_720w.jpg" alt="img" style="zoom:50%;" />

使用每个query ![[公式]](https://www.zhihu.com/equation?tex=q) 去对每个key ![[公式]](https://www.zhihu.com/equation?tex=k) 做attention，attention计算这里就是内积，表征两个向量有多近；
$$
\alpha_{1i} = q_1 \cdot k_i /\sqrt{d}
$$
d是q和k的维度，除以根号d相当于一个归一化操作；

![img](https://pic2.zhimg.com/80/v2-58f7bf32a29535b57205ac2dab557be1_720w.jpg)

之后通过SoftMax层，从α到α‘；

<img src="https://pic3.zhimg.com/80/v2-b7e1ffade85d4dbe3350f23e6854c272_720w.jpg" alt="img" style="zoom:50%;" />

之后每个α’ × 对应的 vi，求和得到bi；分别对a1 - a4做相同的操作，就得到最终的b1 - b4输出sequence

可以看到，每个输出bi都是有着global info的；如果想要输出不包含x1的info，只需要学习到输出b1 = 0即可；

<img src="https://pic2.zhimg.com/80/v2-67bc90b683b40488e922dcd5abcaa089_720w.jpg" alt="img" style="zoom:50%;" />

上述attention操作封装起来，宏观来看，self-attention layer与RNN起到相似的作用

<img src="https://pic2.zhimg.com/80/v2-b081f7cbc5ecd2471567426e696bde15_720w.jpg" alt="img" style="zoom:50%;" />

下面考虑使用vector解释self-attention

embedding 
$$
I = [I^{1} ,I^{2} ,I^{3} ,I^{4} ]
$$
Q K V
$$
Q= [Q^{1} ,Q^{2} ,Q^{3} ,Q^{4} ]
K= [K^{1} ,K^{2} ,K^{3} ,K^{4} ]
V= [V^{1} ,V^{2} ,V^{3} ,V^{4} ]
$$


<img src="https://pic3.zhimg.com/80/v2-6cc342a83d25ac76b767b5bbf27d9d6e_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic2.zhimg.com/80/v2-52a5e6b928dc44db73f85001b2d1133d_720w.jpg" alt="img" style="zoom:50%;" />

![img](https://pic4.zhimg.com/80/v2-1b7d30f098f02488c48c3601f8e13033_720w.jpg)

可以看到 A = KT × Q 

A_hat = Softmax(A)

O = V × A_hat

<img src="https://pic2.zhimg.com/80/v2-8628bf2c2bb9a7ee2c4a0fb870ab32b9_720w.jpg" alt="img" style="zoom:50%;" />

self-attention实际是一堆矩阵乘法，当然从微观对每一个vector分析会更加深刻的理解其机制；

