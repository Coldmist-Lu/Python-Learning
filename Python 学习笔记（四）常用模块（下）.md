# Python 学习笔记（四）常用模块（下）

* 本笔记 # 后为该语句的输出结果，或该变量的值。若 # 后接 ! 号，意思是该语句不能这样写。
* 对于多行的输出结果，我会用""" """进行注释。
* 对于一些输出结果，笔记中为方便理解会在一个代码块写出所有的输出语句，实际调试中应该仅保留一个输出语句（格式化输出print除外），否则前面的输出会被最后一个输出语句覆盖。



* 本笔记将接着上一篇笔记对Python的第三方库进行详细叙述，将对numpy、pandas、matplotlib、sklearn模块等一一介绍。本笔记的内容主要基于深度之眼的Python基础训练营课程，在顺序和例子上面进行了一些修改和总结。
* 本文对Python的基本语法特性将不做详细回顾，因此对于Python的基本语法的请参看笔记（一）基础编程和笔记（二）高级编程。
* 本笔记主要介绍Python的第三方库。



## numpy：科学计算库

### 动机

#### for循环的低效

* Python中的for循环在进行一些计算中是十分低效的。
* 首先介绍一个语法糖 %timeit 用于统计运行时间。该方法会将程序运行多次来计算平均时间。
* 考察下面的例子：求多个数的倒数。

```python
def compute_reciprocals(values):
    res = []
    for value in values:
        res.append(1/value)
    return res

values = list(range(1, 1000000))
%timeit compute_reciprocals(values) 
# 150 ms ± 5.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

* 如果使用 numpy 库：

```python
import numpy as np

values = np.arange(1, 1000000)
%timeit 1/values
# 6.3 ms ± 544 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

* 因此，numpy 库的运算速度快了很多（25倍左右），原因是什么呢？

#### C的高效

* numpy 库是由C语言编写的：
  * C语言属于编译型语言（非解释型语言），对代码进行整体编译，速度更快；
  * Numpy数组形似C语言中的数组，数据类型**必须统一**，而Python列表支持任意数据类型的填充。
    * 这样的存储结构更契合高效的底层处理方式。
  * Python语言无法实现真正的多线程并行计算，而C语言可以。



### 场景

* 我们应该在什么时候使用 numpy 库而不是for循环呢？
  * 大体上来说，当我们需要使用一些向量化、矩阵化操作时，会优先考虑使用 numpy 。
  * 如两个向量的点乘、矩阵乘法。



### 构建

* numpy 库的所有运算都是基于 numpy 数组（准确地说是 numpy.ndarray）进行计算的。
* 因此，在介绍所有计算之前，我们需要对 ndarray 数组的各种构建方式做一了解。

#### 从列表到 ndarray

* numpy 提供将列表转换为数组的方法：
* **np.array(list)**

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x) # [1 2 3 4 5]

type(x) # numpy.ndarray
type(x[0]) # numpy.int32
```

* 可看出，上述方法建立的是 int32 类型的 ndarray。
* 如果我们需要建立其他类型的 ndarray ，就必须自己设置数据类型：

```python
x = np.array([1, 2, 3, 4, 5], dtype='float32')
print(x) # [1. 2. 3. 4. 5.] 浮点型结果后面会有.符号
type(x[0]) # numpy.float32
```

* 这样就建立了一个 float32 类型的数据。通常我们会指定数据类型，以防计算时出错。
* 用这一函数也可以建立二维数组：

```python
x = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])
print(x)
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
```

#### 其他方法创建

* **np.zeros()** 创建全零数组

```python
np.zeros(5, dtype=int) # array([0, 0, 0, 0, 0])
```

* **np.ones()** 创建全1数组

```python
np.ones((2, 4), dtype=float) 
"""
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.]])
"""
```

* **np.full()** 创建全相同元素数组

```python
np.full((3, 5), 8.8)
"""
array([[8.8, 8.8, 8.8, 8.8, 8.8],
       [8.8, 8.8, 8.8, 8.8, 8.8],
       [8.8, 8.8, 8.8, 8.8, 8.8]])
"""
```

* **np.eye()** 创建单位矩阵

```python
np.eye(3)
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""
```

* **np.arange(start, end, step)** 线性序列数组

```python
np.arange(1, 15, 2) # 从1开始，到15结束（不包括15），步长为2
# array([ 1,  3,  5,  7,  9, 11, 13])
```

* **np.linspace(start, end, num)** 等差序列数组
  * 与上一函数的区别在于，这个函数给出的是数组的数量。

```python
np.linspace(0, 3, 5)
# array([0.  , 0.75, 1.5 , 2.25, 3.  ])
```

* **np.logspace(start, end, num)** 等比序列数组
  * 起始值是10的start次方，结束值是10的end次方。

```python
np.logspace(0, 9, 10)
"""
array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,
       1.e+08, 1.e+09])
"""
```

* **np.random.random()** 创建0~1之间均匀分布的随机数数组

```python
np.random.random((3, 3))
"""
array([[0.78846103, 0.85618135, 0.77730669],
       [0.43419035, 0.53395233, 0.98567417],
       [0.31198376, 0.61695423, 0.66247612]])
"""
```

* **np.random.normal()** 创建正态分布随机数数组

```python
np.random.normal(0, 1, (3, 3)) # 0为均值，1为标准差
"""
array([[ 2.20620677, -1.67833076,  0.87802333],
       [ 1.64411512, -0.26525969,  1.1537198 ],
       [ 0.01700017,  0.58944009,  2.03325854]])
"""
```

* **np.random.randint()** 创建随机整数构成的数组

```python
np.random.randint(0, 10, (3, 3)) # [0, 10)
"""
array([[0, 2, 4],
       [4, 2, 1],
       [7, 4, 6]])
"""
```

* **np.random.permutation()** 将数组打乱，并生成一个新数组返回（不改变原数组）

```python
x = np.array([10, 20, 30, 40])
x_shuffle = np.random.permutation(x)
x # array([10, 20, 30, 40])
x_shuffle # array([30, 10, 40, 20])
```

* **np.random.shuffle()** 将原数组打乱，返回打乱后的数组（改变原数组）

```python
x = np.array([10, 20, 30, 40])
x_shuffle = np.random.shuffle(x)
x # array([40, 10, 30, 20])
x_shuffle # array([40, 10, 30, 20])
```

* **np.random.choice()** 随机采样生成数组

```python
x = np.arange(10, 25, dtype = float)
np.random.choice(x, size = (4, 3)) # 按指定形状采样
"""
array([[18., 21., 17.],
       [20., 21., 18.],
       [16., 20., 10.],
       [17., 14., 18.]])
"""
np.random.choice(x, size = (4, 3), p = x / np.sum(x))
"""
array([[24., 23., 15.],
       [24., 18., 11.],
       [18., 19., 15.],
       [14., 16., 18.]])
"""
```



### 属性

* 如何查看一个数组的属性？本节将解决这一问题。
* 本节将基于以下数组介绍：

```python
x = np.random.randint(10, size = (3, 4))
x
"""
array([[2, 0, 5, 7],
       [2, 3, 6, 2],
       [7, 2, 8, 1]])
"""
```

#### 形状 shape

* **array.shape** 返回数组的每一维的数量。

```python
x.shape # (3, 4) 行数为3，列数为4
```

#### 维度 ndim

* **array.ndim** 返回数组的维数。

```python
x.ndim # 2 二维数组
```

#### 大小 size

* **array.size** 返回数组所包含的元素数量。

```python
x.size # 12
```

#### 数据类型 dtype

* **array.dtype** 返回数组中的数据类型。

```python
x.dtype # dtype('int32')
```



### 索引与切片

* 对列表的访问主要以索引和切片为主。ndarray 的访问与 list 的访问类似，下面详细介绍。

#### 一维数组的索引

* 以下讲解基于下面的例子：

```python
x = np.arange(10)
x # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

* 直接使用 **array[n]** 的方法就可以实现索引：
  * n < 0 实现的是反向索引。这一点上与 list 是完全一致的。

```python
x[0] # 0
x[5] # 5
x[-1] # 9
```

#### 多维数组的索引（以二维为例）

* 以下介绍基于下例：

```python
x = np.random.randint(0, 20, (2, 3))
x
"""
array([[15, 10,  9],
       [ 6, 12,  0]])
"""
```

* 两种直接索引的方式：
  * **array[a, b, ...]**
  * **array[a] [b] ...**
  * 上述两种方法是等价的。

```python
x[1, 1] # 12
x[1][1] # 12
x[1][-1] # 0 同样支持反向索引
```

* 小注意点：numpy 数组的数据类型是固定的，因此如果插入不同的数据类型会将结果进行向下取整。

```python
x[1][1] = 10.9
x
"""
array([[15, 10,  9],
       [ 6, 10,  0]]) # 原来的12替换为10
"""
```

#### 一维数组的切片

* 以下讲解基于下例：

```python
x = np.arange(10)
x # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

* 一维数组的基本切片方式和列表是一致的。只不过返回的依然是数组。

```python
x[:3] # array([0, 1, 2])
x[3:] # array([3, 4, 5, 6, 7, 8, 9])
x[::-1] # array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
```

#### 多维数组的切片

* 以下讲解基于下例：

```python
x = np.random.randint(0, 20, (2, 3))
x
"""
array([[15, 10,  9],
       [ 6, 12,  0]])
"""
```



