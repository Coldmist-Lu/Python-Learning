# Python 学习笔记（三）常用模块

* 本笔记 # 后为该语句的输出结果，或该变量的值。若 # 后接 ! 号，意思是该语句不能这样写。
* 对于多行的输出结果，我会用""" """进行注释。
* 对于一些输出结果，笔记中为方便理解会在一个代码块写出所有的输出语句，实际调试中应该仅保留一个输出语句（格式化输出print除外），否则前面的输出会被最后一个输出语句覆盖。



* 本笔记将对Python的常用模块进行详细叙述，将从模块调用、python内置模块到常用的numpy、pandas、matplotlib模块等一一介绍。本笔记的内容主要基于深度之眼的Python基础训练营课程，在顺序和例子上面进行了一些修改和总结。
* 本文对Python的基本语法特性将不做详细回顾，因此对于Python的基本语法的请参看笔记（一）基础编程和笔记（二）高级编程。

* 本笔记在原课程的基础上添加了许多帮助文档上显示的信息，并且添加了一个章节来叙述如何阅读帮助文档。 



目录

[toc]



## 模块总述

* 模块也称包、库，是一种已经被封装好的一部分代码，专门用于解决一个特定问题。它可能是一个单独的.py文件，也可能是多个python文件的组合。有了模块，我们在使用一些代码的时候就不用从头写起，可以直接使用别人的函数和类，进而站在巨人的肩膀上前行。许多热门的算法和框架都是以模块的形式使用的。
* 那么我们现在来学习如何调用和使用模块。



### 模块的分类

* 广义的模块可分为：python内置库、第三方库、自定义库等。

#### python内置库

* 顾名思义，python自带的一些库，无需安装，只要导入（声明）一下就可以使用，如 time (时间库) 、random (随机库)、collection (容器数据类型) 、 itertools (迭代器函数)等。

#### 第三方库

* 第三方库是一些公司或机构编写并发表的供他人学习使用的开源库，这样的库数量及其庞大，我们简单介绍几个：
* numpy、pandas（数据分析库）、matplotlib（数据可视化库）、scikit-learn（机器学习库）、Tensorflow（深度学习框架）

#### 自定义文件

* 我们也可以自己定义自己的库，可以使用单个py文件，或者使用多个py文件构成包。
* 如果文件夹内有多个py文件，需要再加一个--init--.py文件（内容可为空，前面的-符号应该是_，因为笔记中__打不出来）。



### 模块导入

#### 导入整个模块

* 导入模块方式：**import  模块名**
* 调用方式：模块名.函数名或类名
* 下面是一个导入内置库time实现程序计时的例子：

```python
import time # 导入 time 模块

start = time.time() # 调用time模块的time()函数，输出结果返回给start变量
time.sleep(3) # 调用time模块的sleep()函数
end = time.time() # 再次调用time()函数

print('程序运行用时：{:.2f}秒'.format(end-start)) # 获得该程序执行的时间
```

* 下面是一个导入自定义库的例子：

```python
import fun1 # 导入自己定义的fun1模块
fun1.f1() # 调用自己定义的f1()函数
```

#### 从模块中导入类或函数

* 导入模块方式：**from 模块名 import 类名或函数名**
* 调用方式：函数名或类名
* 下面是一个导入内置库itertools中product函数的例子，该函数对两个对象做笛卡尔积并用元组返回。

```python
from itertools import product # 从itertools库中导入product函数

ls = list(product('AB', '123')) # 直接调用product函数
print(ls) # [('A', '1'), ('A', '2'), ('A', '3'), ('B', '1'), ('B', '2'), ('B', '3')]
```

* 下面是一个导入自定义模块的例子。需要注意这里function是一个文件夹（也可以看成是一个模块），fun1是function文件夹中的一个.py文件。

```python
from function.fun1 import f1 # 从function文件夹的fun1.py文件中导入函数f1

f1() # 调用f1
```

* 该方法支持一次导入多个函数或多个文件：

```python
from function import fun1, fun2 # 从function文件夹的fun1.py和fun2.py两个文件中的导入所有函数

fun1.f1() # 调用fun1.py文件中的f1()函数
fun2.f1() # 调用fun2.py文件中的f2()函数
```

#### 导入模块中的所有类和函数

* 导入模块方式：**from 模块名 import ***

* 调用方式：函数名或类名

* 下面给出一个random模块的调用例子：

```python
from random import *

print(randint(1, 100)) # 产生一个[1, 100]之间的随机整数
print(random()) # 产生一个[0, 1)之间的随机小数
```



### 模块的查找路径

* 下面给出系统在导入模块时的查找路径，通过路径的学习可以有效规避一些查找问题。

#### 1、内存中已加载模块

* 系统会首先搜索系统中有没有已加载这个模块，如果有，则直接导入。
* 这一机制会带来一个问题。那就是如果我们在内存中已经加载好这个模块了，对模块在硬盘上进行修改或删除，是**不会起作用的**。
* 下面的例子展示了这个现象：

```python
import fun1
fun1.f1() # 第一次调用

# 此时修改硬盘上的fun1模块文件
import fun1 # 可以执行，因为内存上已经加载好这个模块了
fun1.f1() # 第二次调用，依然可以调用，会产生和第一次调用一样的结果。

# 此时删除硬盘上的fun1模块文件
import fun1 # 可以执行，因为内存上已经加载好这个模块了
fun1.f1() # 第三次调用，依然可以调用，会产生和第一次调用一样的结果。
```

#### 2、内置模块（built-in）

* python在启动时，解释器会默认加载一些模块存放在sys.modules中。
* sys.modules 变量是一个当前载入解释器的模块组成的字典，该字典以模块名为键，他们的位置为值。

```python
import sys

print(len(sys.modules)) # 打印已加载的模块数
print("math" in sys.modules) # 判断是否在模块中
print("numpy" in sys.modules)
```

#### 3、sys.path路径中包含的模块

* sys模块中有一个path变量，该变量是由许多路径组成的列表。系统会到这些列表中查找所需的模块。

```python
import sys

sys.path # 显示所有系统路径
```

* sys.path 的第一个路径是当前执行文件所在的文件夹。



### 其他模块的导入

* 如果一个模块不在系统的查找路径内，也不在系统路径中，那么我们又想使用这个模块，该怎么办呢？
* 我们可以将这个模块的路径添加到系统路径中。添加的方法如下：

```python
import sys

sys.path.append('C:\\Users\\Twist\\Desktop') # windows的路径都是双斜杠。不同电脑路径会有不同。

import fun3

fun3.f3()
```

* 以上我们就完成了对模块使用方法的基本了解。



## 帮助文档

* python内置的帮助文档是我们学习模块的最好工具。由于python中有非常多的模块，以及非常多的类和函数，我们不可能像C语言、C++那样把所有函数全部记下他们的定义和参数的意义。而python具有matlab、R语言等都有的特性，那就是提供了帮助文档供使用者了解每个函数是如何使用的，以及传参细节。下面就基本的文档阅读方法做介绍：



### 查阅帮助文档

* 在Jupyter Notebook中查阅帮助文档的方法是：
  * 函数名 ??
* 还有一种通用的方法就是：
  * help(函数名)
* 需要注意的是上述方法中，如果需要查阅特定模块中的特定函数名，可以将函数名修改成：函数名.模块名



### 函数定义的解释

* 一般帮助文档会先给出一个函数定义，这里会给出参数列表以及传参的要求。

* (制作时这部分省略，最后补充。)



## time：用于处理时间的标准库

### 获取时间 localtime、gmtime、ctime

* **time.localtime()**  获取本地时间（如在中国，获取北京时间）

  * 函数定义：

  ```
  localtime([seconds]) -> (tm_year,tm_mon,tm_mday,tm_hour,tm_min,
                            tm_sec,tm_wday,tm_yday,tm_isdst)
  ```
  
  * localtime() 函数默认从1970年1月1日8时0分0秒（称为the Epoch）开始计时。
  * localtime() 可以接受一个整型数字参数seconds，该数字反映的是所需要得到的时刻相对于 the Epoch 有多少秒。
  * 若不给出该参数，则默认使用现在的时间作为参数。
  * localtime() 函数返回一个时间元组类型数据，将会清晰地显示现在的时间信息（年月日时分秒等）
  
* **time.gmtime()** UTC世界统一时间

  * 函数定义：

  ```
  gmtime([seconds]) -> (tm_year, tm_mon, tm_mday, tm_hour, tm_min,
                         tm_sec, tm_wday, tm_yday, tm_isdst)
  ```
  
  * gmtime() 的用法和 localtime 完全一致，只不过显示的是世界同一时间。
  * 北京时间比世界统一时间早8个小时。
  * 例子：

```python
import time 

t_local = time.localtime()
t_UTC = time.gmtime()
print(t_local)
# time.struct_time(tm_year=2020, tm_mon=6, tm_mday=19, tm_hour=23, tm_min=24, tm_sec=33, tm_wday=4, tm_yday=171, tm_isdst=0) 显示了年、月、日、时、分、秒等信息
print(t_UTC)
# time.struct_time(tm_year=2020, tm_mon=6, tm_mday=19, tm_hour=15, tm_min=24, tm_sec=33, tm_wday=4, tm_yday=171, tm_isdst=0)
```

* 有时我们仅仅需要一个简单的时间，不用那么复杂，那么我们可以用 **time.ctime()** 函数获取一个本地时间的简单字符串。
  
  * 函数定义：
  
  ```
  ctime([seconds]) -> string # 原文本中seconds参数不可省，其实可省。
  ```
  
  * ctime() 函数与 localtime、gmtime 完全一致，只不过返回的是一个字符串形式。

```python
time.ctime() # 'Fri Jun 19 23:27:49 2020'
```



### 时间戳 time、perf_counter、process_time

* 时间戳函数的设计是为了实现计时器的功能。我们很多时候需要知道程序大概要运行多久，那么我们可以通过时间戳来进行设计。

* **time.time()**   返回自纪元以来的秒数。

  * 函数定义：

  ```
  time() -> floating point number
  ```

  * time() 函数返回的秒数是一个浮点型数据。小数部分是代表一秒内的精度。

* **time.perf_counter()**   随意选取一个时间点，记录现在到该时间的秒数。较 time.time() 精度稍高一些。
  
  * 函数定义：
  
  ```
  perf_counter() -> float
  ```
  
  * perf_counter 的全称是 performance counter（性能计数器），大概指的是从一个固定时间点开始提取一个秒数。
  * 每次调用 perf_counter 都会提取一次秒数。从而可以实现计时。
  
* **time.process_time()**   随意选取一个时间点，记录现在到该时间的秒数。和上一个函数的区别是，不会将系统休眠时间算入。
  
  * 函数定义：
  
  ```
  process_time() -> float
  ```
  
  * 该函数每次返回的时间其实不完全是随便取的，而是 kernel 和 CPU 的使用时间之和。
  
* 需要注意的是，这些时间戳每次使用时随意选取仅仅在第一次调用时随机选，但是后续调用时依然按照第一次选取的时间进行。（否则就没法进行计时了嘛）。

* 下面给出一个小例子：

```python
t_1_start = time.time()
t_2_start = time.perf_counter()
t_3_start = time.process_time()
print(t_1_start) # 1592580969.8131123
print(t_2_start) # 5.5e-06
print(t_3_start) # 1.796875

res = 0
for i in range(1000000):
    res += i
    
time.sleep(5)
t_1_end = time.time()
t_2_end = time.perf_counter()
t_3_end = time.process_time()

print("time方法：{:.3f}秒".format(t_1_end-t_1_start)) # time方法：5.147秒
print("perf_counter方法：{:.3f}秒".format(t_2_end-t_2_start)) # perf_counter方法：5.147秒
print("process_time方法：{:.3f}秒".format(t_3_end-t_3_start)) # process_time方法：0.172秒
```

明显可以看出，process_time减去了休眠的大概5秒钟。



### 自定义格式化输出 strftime

* **time.strftime()**  该函数可以自定义时间格式输出：
  
  * 函数定义：
  
  ```
  strftime(format[, tuple]) -> string
  ```
  
  * 首先输入一个格式字符串代表所需格式（具体格式请参看帮助文档），以及一个时间元组，就能按照这个格式输出。若不提供元组信息，那么就会使用 localtime 输出。
  * 请参看帮助文档获取详细信息。

```python
lctime = time.localtime()
time.strftime("%Y-%m-%d %A %H:%M:%S", lctime) # '2020-06-19 Friday 23:38:42'
```



### 睡觉觉 sleep

* 利用 **time.sleep()** 函数可以实现一定时间的系统休眠。上面的例子已经演示过了。sleep函数需要提供一个参数，可以设定睡眠时间。
* 函数定义：

```
sleep(seconds)
```



## random：用于获取随机化数据的标准库

* Python产生的随机数依然是伪随机数，即根据随机化种子生成。所以从某种意义上来说具有确定性。
* random库中的random是一个类，方法中有的静态方法也可以直接调用。

### 基本随机化

#### 随机数种子 seed

* **random.seed()** 

* 函数定义：

```
random.seed(a=None, version=2)
```

* seed() 方法用于初始化随机数种子。通过初始化该种子，可实现对其他初始化函数的随机化。
* 如果不给出随机化种子，则按照系统时间初始化。每做完一次随机数提取后，应该重新初始化该种子。
* 需要注意的是，对于同一个随机化函数，如果随机数种子也一致，那么函数得到的结果也一致。

#### 随机数产生 random

* **random.random()**

  * 函数定义：

  ```
  random() -> x in the interval [0, 1).
  ```

  * random() 方法用于产生一个0~1之间的随机数。下面我们用这个方法与seed结合验证一下刚才的结论：

  ```python
  import random
  random.seed(10) # 初始化种子为10
  random.random() # 0.5714025946899135
  random.seed(10) # 初始化种子为10
  random.random() # 0.5714025946899135 产生的随机数不变
  ```



### 随机整数 randint、randrange

* **random.randint(a, b)**

  * randint() 方法返回一个 [a, b] 之间的随机整数，包括两端点。
  * 给出一个小例子：

  ```python
  import random
  
  numbers = [random.randint(0, 10) for i in range(10)]
  numbers
  # [9, 0, 3, 7, 7, 4, 10, 2, 0, 8]
  ```

* **random.randrange(a)** 

  * 函数定义：

  ```
  random.randrange(start, stop=None, step=1, _int=<class 'int'>)
  ```

  * randrange(a) 方法产生一个 [0, a) 之间的随机整数，不包括右端点。
  * 给出一个小例子：

  ```python
  import random
  
  numbers = [random.randrange(10) for i in range(10)]
  numbers
  # [7, 5, 1, 3, 5, 0, 6, 2, 9, 5]
  ```

  * randrange(a, b) 方法产生一个 [a, b) 之间的随机整数，不包括右端点。
  * randrange(a, b, step) 方法产生一个按照 step 步长划分的 [a, b) 之间的随机整数，不包括右端点。



### 随机浮点数 random、uniform

* **random.random()** 产生 (0.0, 1.0) 之间的随机浮点数。
* **random.uniform(a, b)** 产生 [a, b] 之间的随机浮点数。
* 上述两者都符合均匀分布。用法和整数几乎一致，略。



### 序列用函数 choice、choices、shuffle、sample

#### 随机取样

* **random.choice()**  从序列类型中随机取一个元素

  * 函数定义：

  ```python
  random.choice(seq)
  ```

  * 常见用法：

  ```python
  import random
  
  random.choice(['win', 'lose', 'draw']) # 'draw'  在列表元素中随便取一个
  ```

* **random.choices()**

  * 函数定义：

  ```python
  random.choices(population, weights=None, *, cum_weights=None, k=1)
  ```

  * population参数和seq差不多，weights参数可以给出每个元素的相对权重，cum_weights是累积权重（两者给一个即可）。
  * k代表的是返回的随机选择的数量。可以根据所需要的选择结果进行修改。默认返回一个。

  ```python
  import random
  
  random.choices(['win', 'lose', 'draw'], k=5) # ['lose', 'win', 'draw', 'lose', 'draw']
  random.choices(['win', 'lose', 'draw'], weights = [4,4,1], k=5)
  # ['draw', 'win', 'lose', 'win', 'lose']
  ```

#### 随机打乱

* **random.shuffle() ** 将一个序列数据打乱

  * 函数定义：

  ```python
  random.shuffle(x, random=None)
  ```

  * 小例子：

  ```python
  numbers = ['one', 'two', 'three', 'four']
  random.shuffle(numbers)
  numbers # ['one', 'four', 'two', 'three']
  ```

#### 不放回随机抽样

* **random.sample()**

  * 函数定义：

  ```python
  random.sample(population, k)
  ```

  * population为序列，k为抽样的个数。
  * 小例子：

  ```python
  random.sample([10, 20, 30, 40, 50], 3) # [20, 30, 50]
  ```





