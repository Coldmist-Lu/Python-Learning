# Python 学习笔记（二）高级编程

* 本笔记 # 后为该语句的输出结果，或该变量的值。若 # 后接 ! 号，意思是该语句不能这样写。
* 对于多行的输出结果，我会用""" """进行注释。
* 对于一些输出结果，笔记中为方便理解会在一个代码块写出所有的输出语句，实际调试中应该仅保留一个输出语句（格式化输出print除外），否则前面的输出会被最后一个输出语句覆盖。



* 本笔记将对Python的一些高级特性进行详细叙述，其内容主要基于深度之眼的Python基础训练营课程，在顺序和例子上面进行了一些修改和总结。
* 本文对Python的基础语法将不做详细回顾，因此对于Python的基本数据类型、控制语句结构、函数等不清楚的请参看笔记（一）基础编程。



目录

[toc]

## 数据结构的底层实现

### 列表的底层实现 —— 细谈列表的深浅拷贝

* 在基础编程阶段，我们在列表的复制章节中强调了深浅拷贝的问题。这里我们要对这一问题进行进一步详细的说明。

#### 最浅层的拷贝方法 —— `=`

* 请看下面的例子：

```python
list1 = [1, 2, 3, 4, 5]
list2 = list1

list1.append(6)
list1 # [1, 2, 3, 4, 5, 6]
list2 # [1, 2, 3, 4, 5, 6]
```

* 上面的例子中，list1相当于一个指针，指向这个引用数组的头位置。当执行list2=list1语句时，将list2也指向了list1的头位置，因此当list1的元素发生改变时，通过list2指针访问的依然还是这个数组，因此list2的访问结果也会随之改变。

#### 其实还是浅拷贝 —— list.copy()    list[:]    list(list)

* 我们之前介绍过三种处理上述问题的方法，这里我们再复习一下：

```python
list1 = [1, 2, 3, 4, 5]
list2 = list1.copy()
list3 = list1[:]
list4 = list(list1)

list1.append(6)
list1 # [1, 2, 3, 4, 5, 6]
list2 # [1, 2, 3, 4, 5]
list3 # [1, 2, 3, 4, 5]
list4 # [1, 2, 3, 4, 5]
```

* 可以看出，这三种方法都"成功"地实现了拷贝。对list1加以修改看似不会对其他列表产生影响。但是，这样真的就万事大吉了吗？
* 来看下面的例子：

````python
list1 = [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}]
list2 = list1.copy() # 或 list1[:] 或 list(list1)

list1[1].append(55)
list1 # [1, [22, 33, 44, 55], (5, 6, 7), {'name': 'Sarah'}]
list2 # [1, [22, 33, 44, 55], (5, 6, 7), {'name': 'Sarah'}]
````

* 显然，对于这样一种较为复杂的情况，对list1中某一元素的修改促成了list2的变动。这是为什么呢？

* 这里请回顾一下C语言中的引用数组。即一个数组中，所有的元素都是指针，每个指针都指向一个不同的地址区域。python中列表的底层实现就是一种引用数组。
* 详细地分析一下：
    * list1这里是一个引用数组，list1是数组头指针，它的第一个元素指向了一片区域，这一区域为数字1；第二个元素指向了一片列表区域，该列表共有三个元素（分别是22，33，44）。
    * 当我们对list1进行拷贝操作时，仅仅对这些指针元素进行了拷贝。因此list2的第一个元素的指针依然指向数字1，第二个元素的元素的指针依然指向这一片列表区域。此时如果我们通过list1中第二个元素的指针对列表区域进行修改（如例子中添加55元素），那么自然用list2访问时，该列表区域也会是修改后的结果。这也就说明了为什么list1元素发生变动会影响list2。
* 但是这一问题不总会发生，来看看更多的情况吧：

```python
list1 = [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}]
list2 = list1.copy() # 或 list1[:] 或 list(list1)

list1.append(100) # 新增一个元素
list1 # [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}, 100]
list2 # [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}] # 未影响到list2

# 举一反三，删除一个元素会发生浅拷贝的问题吗？哈哈，不会的。

list1[0] = 10 # 修改数字类型元素
list1 # [10, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}, 100]
list2 # [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}] # 未影响到list2

list1[1].remove(44) # 操作list1中的列表元素
list1 # [10, [22, 33], (5, 6, 7), {'name': 'Sarah'}, 100]
list2 # [1, [22, 33], (5, 6, 7), {'name': 'Sarah'}] # 影响到list2

list1[2] += (8, 9) # 操作list1中的元组
list1 # [10, [22, 33], (5, 6, 7, 8, 9), {'name': 'Sarah'}, 100]
list2 # [1, [22, 33], (5, 6, 7), {'name': 'Sarah'}] # 未影响到list2

list1[3]['sex'] = 'female' # 操作list1中的字典
list1 # [10, [22, 33], (5, 6, 7, 8, 9), {'name': 'Sarah', 'sex': 'female'}, 100]
list2 # [1, [22, 33], (5, 6, 7), {'name': 'Sarah', 'sex': 'female'}] # 影响到list2
```

* 总结一下：
    * 在用list.copy()、list[]、list(list) 三种方法进行列表拷贝时：
    * 对于可变类型的元素（如列表、字典等），这样拷贝是浅层的拷贝，修改原件会影响副本。
    * 对于非可变类型的元素（如元组、字符串、数字等），这样的拷贝时深层的拷贝，修改原件不会影响副本。
    * 究其本质，是list采用了引用数组的方式访问列表中的各个元素，当访问可变类型的元素时，复制的方法仅仅复制了引用指针，而未重新复制这个元素，所以会造成影响，而非可变元素本身不能进行修改，因此进行其他“修改”之后其地址会发生变化，此时原件的指针就会指向一个新地址，这样原件和副本的指针就能做到互补冲突互不干扰啦。

#### 泾渭分明 —— copy模块

* 最妥帖的方法是使用copy模块保证深拷贝。下面是一个例子：

```python
import copy
list1 = [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah'}]
list2 = copy.deepcopy(list1) # 调用深拷贝函数
list1[-1]['age'] = 18 # 修改list1的字典
list2[1].append(55) # 修改list2的列表

# 互不干扰，泾渭分明
list1 # [1, [22, 33, 44], (5, 6, 7), {'name': 'Sarah', 'age': 18}]
list2 # [1, [22, 33, 44, 55], (5, 6, 7), {'name': 'Sarah'}]
```



### 字典的底层实现 —— 哈希表的快速查找

#### 字典到底有多快？

* 看下面的例子：
* 首先使用列表完成查找操作：

```python
import time
ls_1 = list(range(1000000))
ls_2 = list(range(500)) + [-10] * 500

start = time.time() # 计时开始
# 蛮力算法计算ls_2中有多少元素在ls_1中：
count = 0
for n in ls_2:
    if n in ls_1:
        count += 1
end = time.time() # 计时结束
print("共查找{0}个元素，在ls_1列表中的有{1}个，共耗时{2}秒".format(len(ls_2), count, 
                                                   round(end-start, 2)))
# 共查找1000个元素，在ls_1列表中的有500个，共耗时6.56秒
```

* 接着使用字典完成查找操作：

```python
import time
d = {i:i for i in range(1000000)} # 快速生成字典（具体语法在后面讲）
ls_2 = list(range(500)) + [-10] * 500

start = time.time() # 计时开始
# 蛮力算法计算ls_2中有多少元素在d中：
count = 0
for n in ls_2:
    try:
        d[n]  # 调用一下字典的查询，如果没有该索引，就会报错，有的话count+1
    except:
        pass
    else:
        count += 1
end = time.time() # 计时结束
print("共查找{0}个元素，在d字典中的有{1}个，共耗时{2}秒".format(len(ls_2), count, 
                                                   round(end-start, 2)))
# 共查找1000个元素，在d字典中的有500个，共耗时0.0秒
```

* 从上述结果可以看出，字典的查找效率比列表要高得多。它是如何实现的呢？

#### 字典的底层实现 —— 稀疏数组/哈希表

* 字典其实是在内部创建一个散列表，散列表也成为哈希表、稀疏数组。后面我们不加区分。

* 字典是按照以下方法创建的：

    * 第一步：创建一个散列表（稀疏数组，长度N >> 表中需要放入的元素n）

        * 该散列表是一个动态数组，其长度是不断变化的，从而能够适应元素数量的改变。

    * 第二步：计算散列值

        * 通过 hash() 计算键的散列值

        ```python
        hash('Python') # 4850355859126490695
        hash(1024) # 1024
        hash((1, 2)) # 3713081631934410656
        ```

        * 该散列值再进行进一步的处理和运算得到一个在散列表中的位置。极个别的时候会产生冲突，内部有对应的解决方法。

    * 第三步：在对应位置存入值

* 字典是按照以下方法访问的：

    * 第一步：计算要访问键的散列值
    * 第二步：根据计算的散列值，通过一定的规则，确定其在散列表中的位置
    * 第三步：读取该位置上存储的值（若不存在，则返回KeyError）

* 具体的散列值的计算方法、散列表位置的确定方法、冲突的解决方法并不是学习中需要深入探讨的问题，因此我们也不需要过多考虑。

* 字典数据类型的特点：

    * 用空间换时间，实现了数据的快速查找；
    * 因为散列值对应位置的顺序与键在字典中的顺序可能不同，因此表现出来字典是无序的。



### 字符串的底层实现 —— 紧凑数组

#### 紧凑的字符串

* 字符串通过紧凑数组实现存储
    * 数据在内存中是连续存放的，效率更高，节省空间；
    * 为什么列表采用引用数组，而字符串采用紧凑数组呢？这是由于字符串每个字符的大小是一致的，而列表存储的元素是多种多样的，而且可能不断变化，并不知道每个元素需要预留多大的空间。所以采用引用数组更合适。

#### 是否可变的重新探讨

* 不可变类型：数字、字符串、元组

    * 在生命周期内始终保持内容不变。
    * 换句话说，改变以后，该元素的id（地址）就会发生改变。
    * 不可变对象的 += 操作，实际上创建了一个新的对象。
    * 使用id函数可以验证这一问题：

    ```python
    x = 1
    y = 'Python'
    
    id(x) # 4557722976
    id(y) # 4560313120
    
    x += 2
    y += '3.7'
    
    id(x) # 4557723040
    id(y) # 4595666096
    ```

    * 上述计算结果中，每个人的机器算出来的id应该是不一样的。但可以看出，经过变化的不可变类型，其地址发生了改变，已经不再是自己了。
* 再思考一个问题：元组一定是不可变的吗？其实并不一定：
  
    ```python
    t = (1, [2])
    t[1].append(3)
    
    print(t) # (1, [2, 3])
    ```
    
    * 当不可变类型中存在可变的元素时，该类型本质上是可变的。但通常情况下依然称元组为非可变类型。
* 可变类型：列表、字典、集合
    * id保持不变，但内容可以变
    * 可变对象的 += 操作，实际上在原对象的基础上就地修改



### 列表操作的几个问题

#### 删除列表中的特定元素

* 存在运算删除法：
    * 每次存在（in）运算都要从头遍历、查找，效率低。

```python
alist = ['d', 'd', 'd', '2', '2', '2', 'd', 'd', '4']
s = 'd'
while True:
    if s in alist:
        alist.remove(s)
    else:
        break
        
print(alist) # ['2', '2', '2', '4']
```

* 一次性遍历法：
    * 并不能正确的删除所有特定元素。

```python
alist = ['d', 'd', 'd', '2', '2', '2', 'd', 'd', '4']
for s in alist:
    if s == 'd':
        alist.remove(s) # 每次删除s后列表的元素位置和索引都会发生改变。
        
print(alist) # ['2', '2', '2', 'd', 'd', '4']
```

* 解决方法：使用负向索引

```python
alist = ['d', 'd', 'd', '2', '2', '2', 'd', 'd', '4']
for i in range(-len(alist), 0):
    if alist[i] == 'd':
        alist.remove(alist[i]) # 每次删除完以后是反向索引的位置不会发生改变。
        
print(alist) # ['2', '2', '2', '4']
```



#### 多维列表的创建

* 假设我们创建了如下的5*10的二维列表：

```python
ls = [[0] * 10] * 5
ls
"""
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""
```

* 进行如下的赋值操作：

```python
ls[0][0] = 1
ls
"""
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""
```

* 可以发现，本来我们希望只赋一个位置的值，结果对5行都赋值了。
* 该问题源于创建操作的错误。由于使用了*5操作，其实仅仅创建了一行列表（每行的指针都指向同一个列表），所以我们对列表中的一个元素进行修改时，每一行都会被修改。
* 那正确的创建方法是什么呢？这个问题我们放在下一章进行讨论。



## 解析语法与条件表达

### 解析语法

#### 引例：多维列表赋值问题的解决

* 可以使用如下的方式对多维列表进行赋值：

```python
ls = [ [0]*10 for i in range(5) ]
ls
"""
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""

ls[0][0] = 1
ls
"""
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""
```

* 上述创建列表的方式有点类似于用一个for循环进行创建。循环一共执行5次，每一次都创建一个独立的含10个元素的空列表，最后将其合并形成二维列表。
* 上例中的创建语句被称之为"列表解析语句"。

#### 解析语法的基本结构 —— 以列表解析为例

* 列表解析也称为列表推导，两者不加区分。

* 下面是列表解析的基本结构：

    [ expression **for value in iterable** if condition ]

* 可以用**三要素**法来记忆：表达式（expression）、可迭代对象（iterable）、if 条件（可选）。

* 下面重点介绍该语句的执行过程：

    (1)  从**可迭代对象**中拿出一个元素。

    (2)  通过 **if 条件** (如果有的话)，对元素进行筛选：

    ​      若通过筛选，则把元素传递给**表达式**；

    ​      若未通过，则进入 (1) 步骤，进入下一次迭代。

    (3)  将传递给**表达式**的元素，代入**表达式**进行处理，产生一个结果。

    (4)  将 (3) 步产生的结果作为列表的一个元素进行存储。

    (5)  重复 (1) ~ (4) 步骤，直至迭代对象迭代结束，返回新创建的列表。

* 如果觉得上述步骤有点过于繁琐，那么我们可以用以下的代码进行等价：

```python
# 等价代码，不可直接执行，需要对语句进行填充
result = []
for value in iterable:
    if condition:
        result.append(expression)
```

* 举个例子，比如我们要求20以内奇数的平方：

```python
result = []
for i in range(20):
    if i % 2 == 1:
        result.append( i**2 )
result # [1, 9, 25, 49, 81, 121, 169, 225, 289, 361]
```

* 如果用列表解析来表示，可以用如下的语句：

```python
result = [ i**2 for i in range(20) if i % 2 == 1 ]
result # [1, 9, 25, 49, 81, 121, 169, 225, 289, 361]
```

* 这样创建既简洁又容易理解，而且运行起来比一般的代码要更快。

#### 解析语法的其他机制

* value 支持多变量
    * 可以同时用两个变量对两个列表进行打包的遍历方法。

```python
x = [1, 2, 3]
y = [4, 5, 6]

result = [ i*j for i,j in zip(x,y) ] # 多变量以元组的形式传递
result # [4, 10, 18]
```

* 支持循环嵌套
    * 可以在一个循环中嵌套另一个循环，构造所有的排列组合。

```python
colors = ['black', 'white']
sizes = ['S', 'M', 'L']
tshirts = [ '{}{}'.format(color, size) for color in colors for size in sizes ]
tshirts # ['blackS', 'blackM', 'blackL', 'whiteS', 'whiteM', 'whiteL']
```

#### 字典和集合的解析语法

* 解析语法构造字典（字典推导）

```python
squares = {i: i**2 for i in range(10)}
squares # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
```

* 解析语法构造集合（集合推导）

```python
squares = {i**2 for i in range(10)}
squares # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
```

* 突发奇想：如果两端使用小括号"()"会怎么样？

```python
squares = (i**2 for i in range(10))
squares # <generator object <genexpr> at 0x107613db0>
```

* 这个奇奇怪怪的东西叫做"生成器"。可以用它来进行遍历。我们将在下一章提到它。
* 给一个小例子吧：

```python
colors = ['black', 'white']
sizes = ['S', 'M', 'L']
tshirts = ( '{}{}'.format(color, size) for color in colors for size in sizes )
for tshirt in tshirts:
    print(tshirt, end=" ")   
# blackS blackM blackL whiteS whiteM whiteL 
```



### 条件表达

#### 条件表达的基本结构

* 下面是条件表达式的基本结构：

    expression1  if  condition  else  expression2

* 如果还原成等价的代码应该是这样的：

```python
# 等价代码，不可直接执行，需要对语句进行填充
if condition:
    expression1
else:
    expression2
```

* 下面举一个简单的例子：
    * 取n的绝对值给x

```python
# 正常的写法
n = -10
if n >= 0:
    x = n
else:
    x = -n
    
# 条件表达式写法
x = n if n >= 0 else -n
```

* 条件表达式让条件变得更加的简洁，其运行速度也会更快。是一种经常使用的语法格式。



## 生成器、迭代器、装饰器

### 生成器

#### 引例：为什么需要生成器

* 假如我们需要1000000以内的平方项，那么直观的想法是用以下方式实现：

```python
ls = [i**2 for i in range(1000000)]
```

* 这样做会生成一个列表，虽然可以供我们随时取用，但是这个列表过长，占用了太大的内存空间，而且我们也不一定会同时用到所有的结果。
* 当我们需要复杂结果的时候，与其生成一个大列表存储，我们也可以考虑仅在需要时才进行计算。生成器就是为了解决这样的问题。

#### 生成器的特点

* 采用惰性计算的方式；
* 无需一次性存储海量数据；
* 一边执行一边计算，只计算每次需要的值；
* 实际上一直在执行 next() 操作，直到无值可取。

#### 生成器的基本写法

* 生成器表达式：

    * 请看下面的例子：
        * squares 就是一个生成器，它仅仅在后面的 for 循环执行的时候才进行计算

    ```python
    squares = (i**2 for i in range(1000000))
    for i in squares:
        pass # 对结果执行相关操作
    ```

    * 下面是一个求1加到100的和的生成器计算法：

    ```python
    sum( (i for i in range(1, 101)) ) # 5050
    ```

    * 这样我们并不需要把0～100全部写出来就能够直接计算他们的和。

* 生成器函数—— `yield ` 生成斐波那契数列的例子：

    * 假如我们要生成斐波那契数列的一些项，可以这么做：

    ```python
    def fib(n):
        ls = []
        i, a, b = 0, 1, 1
        while i < n:
            ls.append(a)
            a, b = b, a+b
            n += 1
           
    fib(10) 
    ```

    * 但是如果我们不希望用列表把他们存起来，怎么办呢：

    ```python
    def fib(n):
        i, a, b = 0, 1, 1
        while i < n:
            yield a # 生成器函数
            a, b = b, a+b
            n += 1
           
    for a in fib(10):
        print(a, end = ' ')
        
    # 
    ```

    * 通过yield函数会自动生成迭代器。其运行原理是这样的：

        (1)  每次访问生成器的时候都会执行这个函数，当函数执行到 yield 语句时返回，返回值就是 yield 后面的那个变量的值（可以把 yield 看成是 return ）。

        (2)  当下一次访问生成器时，程序就会从上一次 yield 返回的地方开始继续执行，直到再次遇到 yield 返回。如此往复。

        (3)  当函数到达终点正常返回时，生成器也就完成了其使命。

    * 这样，我们就可以做到不使用列表，而在每次需要时才计算的功能。




### 迭代器

#### 可迭代对象 Iterable

* 可直接用于for循环的对象统称为可迭代对象。

    * 如何判断某个对象是否可迭代？ 

    ```python
    from collections import Iterable # 记得导入！
    
    isinstance([1, 2, 3], Iterable) # True
    ```

    * 可以用上面的方法进行判断。导入Iterable，然后通过isinstance来判断。

* 列表、元组、字符串、字典、集合、文件都是可迭代对象。
* 生成器也是可迭代对象。
    * 生成器不仅仅是for循环使用，也可以用 next() 函数进行调用。
    * 当无值可取时，会抛出StopIteration异常。

```python
squares = (i**2 for i in range(10))
isinstance(squares, Iterable) # True

next(squares) # 0
next(squares) # 1
next(squares) # 4
```



#### 迭代器 Iterator

* 可以被 next() 函数调用并不断返回下一个值，直到无值可取时返回StopIteration异常的对象，称为迭代器。
* 显然，生成器也是迭代器。但列表、元组、字符串、集合、字典都不是迭代器。
* 如何判断迭代器？

```python
from collections import Iterator

isinstance([1, 2, 3], Iterator) # False
```

* 可以通过 iter() 函数创建迭代器：

```python
isinstance(iter([1, 2, 3]), Iterator) # True
```

* 因此 for item in Iterable 等价于：
    * 先通过 iter() 函数获取可迭代对象Iterable的迭代器；
    * 然后对获取到的迭代器不断调用 next() 方法来获取下一个值，将其赋值给 item；
    * 当遇到 StopIteration 异常时循环结束。
* zip、enumerate 等 itertools 里的函数都是迭代器：
    * 我们复习一下他们的用法：

```python
x = [1, 2]
y = ['a', 'b']
isinstance(zip(x, y), Iterator) # True

for i in zip(x, y):
    print(i)
    
"""
(1, 'a')
(2, 'b')
"""
```

```python
numbers = [1, 2, 3, 4, 5]
isinstance(enumerate(numbers), Iterator) # True
for i in enumerate(numbers): # 将位置信息和值构成元组输出
    print(i)
    
"""
(0, 1)
(1, 2)
(2, 3)
(3, 4)
(4, 5)
"""
```

* 文件是迭代器：

```python
with open('测试文件.txt', 'r', encoding = 'utf-8') as f: # 要有这个文件
    print(isinstance(f, Iterator)) # True
```

* 迭代器是可耗尽的：

```python
squares = (i**2 for i in range(5))
for square in squares:
    print(square, end = " ") # 0 1 4 9 16 
    
for square in squares:
    print(square, end = " ") # 无输出，迭代器耗尽。
```

* range() 不是迭代器
    * range() 有长度、可索引、可存在运算、不可被 next() 调用、且不会被耗尽。

```python
numbers = range(10)
isinstance(numbers, Iterator) # False
```

* 可以认为range是一个懒序列。
    * 它是一种序列，但并不包含任何内存中的内容，而是通过计算来回答问题。



### 装饰器

#### 需求

* 需要对已开发的上线的程序添加某些功能
* 不能对程序中函数的源代码进行修改
* 不能改变程序中函数的调用方式

* 装饰器就完成了上述功能。

#### 基本概念：函数对象、高阶函数、嵌套函数、闭包

函数是python中的一类对象。

* 可以把函数赋值给变量；
* 可以对该变量进行调用，可实现函数的功能。

```python
def square(x):
    return x**2

print(type(square)) # <class 'function'> square 是function类的一个实例
```

```python
pow_2 = square  # 把函数赋值给变量，可以理解成给函数起了一个别名pow_2
print(pow_2(5)) # 25
print(square(5)) # 25
```

* 高阶函数：接受函数作为参数 或者 返回一个函数 称为高阶函数。
    * 举一个例子：

```python
def square(x):
    return x**2

def pow_2(fun):
    return fun

f = pow_2(square) # pow_2函数返回square 等价于 f = square.
f(8) # 64
```

* 嵌套函数：在函数内部嵌套一个函数

```python
def outer():
    print('outer is running!')
    
    def inner():
        print('inner is running!')
        
    inner()

outer()
"""
outer is running!
inner is running!
"""
```

* 闭包：可以用以下几个定义来理解：
    * 延伸了作用域的函数；
    * 如果一个函数定义在另一个函数的作用域内，并且引用了外层函数的变量，则称该函数为闭包；
    * 闭包是由函数及其相关的引用环境组合而成的实体（即：闭包 = 函数 + 引用环境）
    * 我们用以下的例子来说明：

```python
def outer():
    x = 1
    z = 10
    
    def inner():
        y = x + 100 # 使用了外部函数的x
        return y, z # 使用了外部函数的z
    
    return inner

f = outer() # 包含了内部函数和外层函数的信息
print(f) # <function outer.<locals>.inner at 0x7f9010225a60>

print(f.__closure__) # __closure__ 属性中包含了来自外部函数的信息
for i in f.__closure__: 
    print(i.cell_contents) # 1 10 利用 cell_contents 函数看看存了什么
    
print(f()) # (101, 10)
```

* 一旦在内层函数重新定义了相同名字的变量，则该变量会成为局部变量

```python
def outer():
    x = 1
    
    def inner():
        x = x + 100
        return x
    
    return inner

f = outer()
f() # 报错，inner函数中没有定义x
```

* 使用 nonlocal 允许内嵌函数修改闭包变量：

```python 
def outer():
    x = 1
    
    def inner():
        nonlocal x
        x = x + 100
        return x
    
    return inner

f = outer()
f() # 101
```

#### 装饰器

* 现在给出一个装饰器的小例子：

```python
import time

def f1():
    print('f1 run')
    time.sleep(1)
    
f1() # 需要给 f1() 函数增加计时功能
```

```python
import time

def timer(func):
    
    def inner():
        print("inner run")
        start = time.time()
        func()
        end = time.time()
        print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
       
    return inner

def f1():
    print('f1 run')
    time.sleep(1)
    
f1 = timer(f1) # 给f1添加timer()环境
f1()
"""
inner run
f1 run
f1 函数运行用时 1.00 s
"""
```

* 上述方法用了闭包的原理，在不更改f1函数名的情况下，修改了f1的内容。

* 需要注意的是，这里的 f1 已经不是原来的 f1 了，我们通过这种“偷换概念”的方式实现了装饰器的功能。

    * 但是这样写有点直白，影响了代码的美观，因此python引入了一个语法糖的功能：

    ```python
    import time
    
    def timer(func):
        
        def inner():
            print("inner run")
            start = time.time()
            func()
            end = time.time()
            print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
           
        return inner
    
    @timer   # 相当于实现了 f1 = timer(f1)
    def f1():
        print('f1 run')
        time.sleep(1)
        
    f1()
    ```

* 以上就完成了基本装饰函数的介绍。

#### 深入讨论装饰器

* 继续深入讨论，如何**装饰有参函数**呢？

```python
import time

def timer(func):
    
    def inner(*args, **kwargs): # 设置参数列表
        print("inner run")
        start = time.time()
        func(*args, **kwargs) # 将上述参数传递给func函数进行运行即可
        end = time.time()
        print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
       
    return inner

@timer
def f1(n):
    print('f1 run')
    time.sleep(n)

f1(4)
"""
inner run
f1 run
f1 函数运行用时 4.00 s
"""
```

* 如果被装饰函数**有返回值**怎么办呢？

```python
import time

def timer(func):
    
    def inner(*args, **kwargs):
        print("inner run")
        start = time.time()
        res = func(*args, **kwargs) # 增添返回值
        end = time.time()
        print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
        return res # 返回res
       
    return inner

@timer
def f1(n):
    print('f1 run')
    time.sleep(n)
    return "wake up"
    
f1(4)
"""
inner run
f1 run
f1 函数运行用时 4.00 s
'wake up'
"""
```

* 如何处理**带参数的装饰器**呢？
    * 装饰器本身要传递一些额外参数。
    * 需求：有时需要统计绝对时间，有时需要统计绝对时间的2倍。

```python
def timer(method):
    
    def outer(func):
        
        def inner(*args, **kwargs):
            print("inner run")
            if method == "origin":
                print("origin_inner run")
                start = time.time()
                res = func(*args, **kwargs)
                end = time.time()
                print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
            elif method == "double":
                print("double_inner run")
                start = time.time()
                res = func(*args, **kwargs)
                end = time.time()
                print("{} 函数运行双倍用时 {:.2f} s".format(func.__name__,2*(end - start)))
            return res
        
        return inner
    
    return outer

@timer(method='origin')
def f1():
    print('f1 run')
    time.sleep(1)
    
@timer(method='double')
def f2():
    print('f2 run')
    time.sleep(1)
    
f1()
print()
f2()
"""
inner run
origin_inner run
f1 run
f1 函数运行用时 1.00 s

inner run
double_inner run
f2 run
f2 函数运行双倍用时 2.01 s
"""
```

#### 细节问题

* 装饰器何时执行？
    * 一装饰就执行，无需调用

```python
func_names = []
def find_function(func):
    print('run')
    func_names.append(func)
    return func

@find_function
def f1():
    print("f1 run")
    
@find_function
def f2():
    print("f2 run")
    
@find_function
def f3():
    print("f3 run")
    
"""
此时运行，将会输出：
run
run
run
"""
for fun in func_names:
    print(fun.__name__)
    fun()
    print()
"""
f1
f1 run

f2
f2 run

f3
f3 run

"""
```

* 第一个输出结果可以看出，装饰器被调用了3次。因此才会输出3个run，进而说明装饰器一装饰就会执行。
* 第二个结果可以进一步看出这些函数已经加入了func_name变量中。可以接受调用。

* 回归本源：

    * 原函数的属性被掩盖了

    ```python
    import time
    
    def timer(func):
        
        def inner():
            print("inner run")
            start = time.time()
            func()
            end = time.time()
            print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
           
        return inner
    
    @timer   # 相当于实现了 f1 = timer(f1)
    def f1():
        print('f1 run')
        time.sleep(1)
        
    print(f1.__name__) # inner
    ```

    * 这说明 f1 的函数名还是 inner 并不是f1。这样还是有一些不完美。

* wraps函数

    * wraps函数就很好的解决了这个问题，它骗过了电脑，使得最终显示的名字还是 f1。

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def inner():
        print("inner run")
        start = time.time()
        func()
        end = time.time()
        print("{} 函数运行用时 {:.2f} s".format(func.__name__, (end - start)))
       
    return inner

@timer   # 相当于实现了 f1 = timer(f1)
def f1():
    print('f1 run')
    time.sleep(1)
    
print(f1.__name__) # f1
```

* 至此我们就完美解决了装饰器问题。



## 面向对象的编程简介

### 类与对象

* python的语法中包含了很多面向对象的内容，学过一定编程知识的同学应该清楚面向对象的重要性。所以如果没有学过面向对象的建议先看一些书稍微了解一下。因此本部分内容实在是有点难讲，我可能也不会讲的特别清楚，大家尽量以了解语法为主，面向对象的思想的理解是需要一定的经验和积淀的。

#### 一切皆为对象

对象包括这样几个特点：

* 所有的客观世界的事物都可以看成是一个对象。
* 每个对象都有其内在属性。（属性）
* 每个对象对外会表现出其一定的行为。（方法）

类是对象的载体，把一类对象的公共特征抽象出来，就是一个通用类。

* python可以通过如下的方法创建类：

```python
class Cat():
    def __init__(self, name): # 类的属性初始化
        self.name = name # 属性为名字
        
    def jump(self): # 模拟猫的跳跃
        print(self.name + ' is jumping')
```

* 用类创建实例：

```python
my_cat = Cat('Lucky')
your_cat = Cat('Winner')
```

* 查看属性：

```python
print(my_cat.name) # Lucky
print(your_cat.name) # Winner
```

* 调用方法：

```python
my_cat.jump() # Lucky is jumping
your_cat.jump() # Winner is jumping
```

#### 类的组成

* 命名：
    * 一般首字母大写。
    * 要有实际意义。
    * 一般采用驼峰命名法：CreditCard、ElectricCar
* 格式：

```python
# class 类名
"""类前空两行"""


class Car():
    """对该类的简单介绍"""
    pass

"""类后空两行"""
```

* 属性：

    * 属性可以理解为类内部的变量。仅仅在类的内部使用，用于存储对象的数据和一些特征。
    * 下面是属性的格式。注意传递的参数第一个一定要有 self 。因为初始化参数是要给类的属性赋值，而 self 指代的是该类的对象，如果没有这个参数就无法给属性传参，使用时就会出错。

    ```python
    def __init__(self, 要传递的参数):
    ```

* 方法：

    * 类可以定义一些方法以供使用，需要注意的是类的方法必须要传入self变量。
    * 但是在使用对象调用该方法的时候不需要传入self。

* 下面将刚刚的几个要素应用到下面的例子中。

    * 这个例子我们后面会继续使用。


```python
class Car():
    """模拟汽车"""
    
    def __init__(self, brand, model, year):
        """初始化汽车属性"""
        self.brand = brand # 品牌
        self.model = model # 型号
        self.year = year # 出厂年份
        self.mileage = 0 # 里程初始化
        
    def get_main_information(self):
        """获取汽车主要信息"""
        print("品牌：{}   型号：{}   出厂年份：{}".format(self.brand, self.model,
                                                      self.year))
    def get_mileage(self):
        """获取总公里数"""
        print("行车总里程：{}公里".format(self.mileage))
```



### 实例

#### 创建

* 将实例赋值给对象，实例化的过程需要传入对应的参数（如果有默认参数，也可以使用默认参数）
* 创建的对象名 = 类名（必要的初始化参数）

```python
my_new_car = Car('Audi', 'A6', 2018)
```

#### 访问属性

* 实例名.属性名

```python
print(my_new_car.brand) # Audi
print(my_new_car.model) # A6
print(my_new_car.year)  # 2018
```

#### 调用方法

* 实例名.方法名(必要的参数)

```python
my_new_car.get_main_information() # 品牌：Audi   型号：A6   出厂年份：2018
my_new_car.get_mileage() # 行车总里程：0公里
```

#### 修改属性

* 直接修改

```python
my_old_car = Car("BYD", '宋', 2016)
print(my_old_car.mileage) # 0
my_old_car.mileage = 12000 # 修改属性
print(my_old_car.mileage) # 12000
my_old_car.get_mileage() # 行车总里程：12000公里
```

* 通过方法修改属性
    * 重新定义上述类，增加设置总公里数的方法

```python
class Car():
    """模拟汽车"""
    
    def __init__(self, brand, model, year):
        """初始化汽车属性"""
        self.brand = brand # 品牌
        self.model = model # 型号
        self.year = year # 出厂年份
        self.mileage = 0 # 里程初始化
        
    def get_main_information(self):
        """获取汽车主要信息"""
        print("品牌：{}   型号：{}   出厂年份：{}".format(self.brand, self.model,
                                                      self.year))
    def get_mileage(self):
        """获取总公里数"""
        print("行车总里程：{}公里".format(self.mileage))
        
    def set_mileage(self, distance): # 增加设置总公里数的方法
        """设置总公里数"""
        self.mileage = distance
```

```python
my_old_car = Car('BYD', '宋', 2016)
my_old_car.get_mileage() # 行车总里程：0公里
my_old_car.set_mileage(8000)
my_old_car.get_mileage() # 行车总里程：8000公里
```

#### 拓展功能

* 禁止设置负里程
* 将每次的里程数累加。

```python
class Car():
    """模拟汽车"""
    
    def __init__(self, brand, model, year):
        """初始化汽车属性"""
        self.brand = brand # 品牌
        self.model = model # 型号
        self.year = year # 出厂年份
        self.mileage = 0 # 里程初始化
        
    def get_main_information(self):
        """获取汽车主要信息"""
        print("品牌：{}   型号：{}   出厂年份：{}".format(self.brand, self.model,
                                                      self.year))
    def get_mileage(self):
        """获取总公里数"""
        print("行车总里程：{}公里".format(self.mileage))
        
    def set_mileage(self, distance): # 增加设置总公里数的方法
        """设置总公里数"""
        if distance >= 0:
            self.mileage += distance # 累加
        else:
            print("里程数不能为负！") # 禁止设置负里程
```

```python
my_old_car = Car('BYD', '宋', 2016)
my_old_car.get_mileage() # 行车总里程：0公里
my_old_car.set_mileage(-8000) # 里程数不能为负！
my_old_car.get_mileage() # 行车总里程：0公里
my_old_car.set_mileage(200)
my_old_car.set_mileage(1000)
my_old_car.get_mileage() # 行车总里程：1200公里
```

* 一个类包含很大的信息量，实现高度的拟人（物）化。
    * 原课程中这句话有点笼统，也不太准确。笔者认为，类存在的意义是实现面向对象的设计，以及对物体特性的抽象表达。类在一定程度上沟通了现实世界与抽象代码之间的桥梁，但学习时一定要注意，**并不是所有的类都是为了沟通桥梁而建立的。**所以学习面向对象最重要的是：
        * 充分理解做这一特性的目的。
        * 记忆实现这一特性的语法。



### 继承特性

* 继承是从底层向高层抽象的过程。父类将公共特征抽取出来建立类，各子类建立各自的特征，构建各自的类。也可以修改父类中的方法。

#### 简单继承

* 父类：

```python
class Car():
    """模拟汽车"""
    
    def __init__(self, brand, model, year):
        """初始化汽车属性"""
        self.brand = brand # 品牌
        self.model = model # 型号
        self.year = year # 出厂年份
        self.mileage = 0 # 里程初始化
        
    def get_main_information(self):
        """获取汽车主要信息"""
        print("品牌：{}   型号：{}   出厂年份：{}".format(self.brand, self.model,
                                                      self.year))
    def get_mileage(self):
        """获取总公里数"""
        print("行车总里程：{}公里".format(self.mileage))
        
    def set_mileage(self, distance): # 增加设置总公里数的方法
        """设置总公里数"""
        if distance >= 0:
            self.mileage += distance # 累加
        else:
            print("里程数不能为负！") # 禁止设置负里程
```

* 子类：
    * class 子类名（父类名）：
    * 新建一个电动汽车类：

```python
class ElectricCar(Car):
    """模拟电动汽车"""
    
    def __init__(self, brand, model, year): # 覆写父类的__init__方法
        super().__init__(brand, model, year) # super()表示调用父类方法
```

* 子类对象可以直接使用父类的所有方法：

```python
my_electric_car = ElectricCar('NextWeek', 'FF91', 2046)
my_electric_car.get_main_information() # 品牌：NextWeek   型号：FF91   出厂年份：2046
```

* 现在子类对象和父类是完全一致的。

#### 添加属性和方法

```python
class ElectricCar(Car):
    """模拟电动汽车"""
    
    def __init__(self, brand, model, year, bettery_size):
        """初始化电动汽车属性"""
        super().__init__(brand, model, year)    # 声明继承父类的属性
        self.bettery_size = bettery_size        # 电池容量
        self.electric_quantity = bettery_size   # 电池剩余电量
        self.electric2distance_ratio = 5        # 电量距离换算系数 5公里/kW.h
        self.remainder_range = self.electric_quantity*self.electric2distance_ratio 
        # 剩余可行驶里程
    
    def get_electric_quantity(self):
        """查看当前电池电量"""
        print("当前电池剩余电量：{} kW.h".format(self.electric_quantity))
        
    def set_electric_quantity(self, electric_quantity):
        """设置电池剩余电量，重新计算电量可支撑行驶里程"""
        if electric_quantity >= 0 and electric_quantity <= self.bettery_size:
            self.electric_quantity = electric_quantity
            self.remainder_range = self.electric_quantity*self.electric2distance_ratio
        else:
            print("电量未设置在合理范围！")
    
    def get_remainder_range(self):
        """查看剩余可行驶里程"""
        print("当前电量还可以继续驾驶 {} 公里".format(self.remainder_range))              
```

* 接下来创建一个实例：

```python
my_electric_car = ElectricCar("NextWeek", "FF91", 2046, 70)
my_electric_car.get_electric_quantity()           # 当前电池剩余电量：70 kW.h
my_electric_car.get_remainder_range()             # 当前电量还可以继续驾驶 350 公里
```

* 尝试调整部分参数：

```python
my_electric_car.set_electric_quantity(50)         # 重设电池电量
my_electric_car.get_electric_quantity()           # 当前电池剩余电量：50 kW.h
my_electric_car.get_remainder_range()             # 当前电量还可以继续驾驶 250 公里
```

#### 重写父类的方法

* 原课程中将这一部分与多态特性进行了同时比较，而且代码中并没有使用到面向对象的多态特性。所以这里我们分开进行说明。
* 父类方法的重写（或称覆盖、覆写、重新定义，一个意思，英文是override）是继承特性的一部分，指的是该子类对原父类已定义的方法进行了一些修改，这是为了提升程序的复用性，针对每个子类的不同特性修改适合这个子类的函数。
* 请看下面的例子：

```python
class ElectricCar(Car):
    """模拟电动汽车"""
    
    def __init__(self, brand, model, year, bettery_size):
        """初始化电动汽车属性"""
        super().__init__(brand, model, year)    # 声明继承父类的属性
        self.bettery_size = bettery_size        # 电池容量
        self.electric_quantity = bettery_size   # 电池剩余电量
        self.electric2distance_ratio = 5        # 电量距离换算系数 5公里/kW.h
        self.remainder_range = self.electric_quantity*self.electric2distance_ratio 
        # 剩余可行驶里程
    
    def get_main_information(self):        # 重写父类方法
        """获取汽车主要信息"""
        print("品牌：{}   型号：{}   出厂年份：{}   续航里程：{} 公里"
              .format(self.brand, self.model,
                      self.year,self.bettery_size*self.electric2distance_ratio))
    
    def get_electric_quantit(self):
        """查看当前电池电量，重新计算电量可支撑行驶里程"""
        print("当前电池剩余电量：{} kW.h".format(self.electric_quantity))
        
    def set_electric_quantity(self, electric_quantity):
        """设置电池剩余电量"""
        if electric_quantity >= 0 and electric_quantity <= self.bettery_size:
            self.electric_quantity = electric_quantity
            self.remainder_range = self.electric_quantity*self.electric2distance_ratio
        else:
            print("电量未设置在合理范围！")
    
    def get_remainder_range(self):
        """查看剩余可行驶里程"""
        print("当前电量还可以继续驾驶 {} 公里".format(self.remainder_range))
```

* 这个例子中，函数 get_main_information 重写了父类中的同名方法，补充了续航里程的输出。这样定义以后，我们就可以在子类对象中使用父类方法了。

```python
my_electric_car = ElectricCar("NextWeek", "FF91", 2046, 70)
my_electric_car.get_main_information() 
# 品牌：NextWeek   型号：FF91   出厂年份：2046   续航里程：350 公里
```



### 多态特性

* 多态特性其实说明起来是比较复杂的。那么我们还是使用一个例子来说明。

#### 引子

```python
class Student():
    """学生类"""
    
    def __init__(self, name, school, grade):
        self.name = name
        self.school = school
        self.grade = grade
    def get_profile(self):
        return "我是{}，来自{}，{}年级了".format(self.name, self.school, self.grade)
    
def print_profile(student):
    """打印学生档案"""
    print(student.get_profile())
```

* 假设我们有上述的学生类，并且写好了一个得到自我介绍（get_profile）的方法。此外我们还写了一个全局函数print_profile()，这个函数接受Student类的对象，并且通过调用get_profile方法将自我介绍打印出来。
* 那么就可以写出如下的例子：

```python
print_profile(Student('小红', '第一中学', 1))
# 我是小红，来自第一中学，1年级了
```

* 上述例子中，创建了一个Student对象，并且赋值给print_profile类。这样可以得到理想的结果。

#### 子类

* 如果我们此时需要添加具体的小学生、中学生、大学生三个子类，并且对get_profile函数修改，该怎么办呢？

```python
class PrimarySchoolStudent(Student):
    """初中生类"""
    
    def __init__(self, name, school, grade, hobby):
        super().__init__(name, school, grade)
        self.hobby = hobby
    def get_profile(self):
        return "我是{}，来自{}，{}年级了，我喜欢{}".format(self.name, self.school,
                                                    self.grade, self.hobby)
        
        
class HighSchoolStudent(Student):
    """高中生类"""
    
    def __init__(self, name, school, grade, ability):
        super().__init__(name, school, grade)
        self.ability = ability
    def get_profile(self):
        return "我是{}，来自{}，{}年级了，我擅长{}".format(self.name, self.school,
                                                    self.grade, self.ability)

        
class CollegeStudent(Student):
    """大学生类"""
    
    def __init__(self, name, school, grade, certificate):
        super().__init__(name, school, grade)
        self.certificate = certificate
    def get_profile(self):
        return "我是{}，来自{}，{}年级了，我有{}的技能证书".format(self.name, self.school,
                                                    self.grade, self.certificate)
```

* 现在我们添加了这三个子类。可以看出，每个子类中自我介绍的语句都是不同的。现在我们用相同的方法给子类调用print_profile函数，结果会怎么样呢？

```python
s1 = PrimarySchoolStudent('小明', '第五小学', 5, '读书')
print_profile(s1)
# 我是小明，来自第五小学，5年级了，我喜欢读书

s2 = HighSchoolStudent('小雪', '第二中学', 2, '钢琴')
print_profile(s2)
# 我是小雪，来自第二中学，2年级了，我擅长钢琴

s3 = CollegeStudent('小龙', '清华大学', 3, '六级')
print_profile(s3)
# 我是小龙，来自清华大学，3年级了，我有六级的技能证书
```

* 可以看出，我们这里虽然没有改动原来调用父类的函数 print_profile，却能够直接实现用这一函数打印子类的功能。下面我们做深入讨论。

#### 讨论总结

* 上面的例子展现了如下的功能：当我们写完一个父类方法以及调用该父类对象的函数时，如果我们从父类继承出一个或多个子类，那么调用父类的方法也同样可以接受子类对象作为参数，并且会正确地调用复写过的子类方法来实现相关功能。
* 这其实是几乎所有面向对象语言表现出的特性：多态特性。字面理解，"多种形态"，即一个变量可以接受多种形态的赋值，既可以称为一个父类对象，也可以称为一个子类对象。而且，多态的特性保证了方法的调用是非常**智能化**的，它会根据自身正确的子类类型帮我们去判断需要调用哪个子类的方法，而不需要我们显示地给出。
* 还是以刚刚的例子来说，我们并没有事先告诉print_profile函数，“我要传入一个子类了，请你做好准备，记得调用子类的方法哦，不要调用错了”。多态直接保证了这一步骤的实现。它清晰地判断了子类的类型，然后根据这一特征实现对应的算法。

* 当然你可能会觉得，这很正常啊，毕竟你事先给s1、s2、s3赋值过，系统已经知道他们的数据类型了，自然会产生正确的结果啊。但是，这样的多态特性使得我们的代码复用能力和扩展性大大提高。
* 再举一个简单的例子。比如我们有一个项目的开源代码，该项目本来写好了一个输入数据类，但是我们这时候需要修改成我们所需要的那个输入数据类，此时我们就可以选择定义一个子类，然后复写原代码中的输入方法，这样在系统执行时就会自动调用这一子类的代码进行操作，而不会理会原代码了。从这一角度，多态特性确保了代码的可扩展能力。

* 可能说的有些啰嗦哈，但是希望能听懂。面向对象还有很多特性，我们先就介绍到这里吧。



## 文件

* 绝大多数的数据是通过文件的交互完成的。
* 请注意，本节代码有时无法直接实现，需要进行必要的修改，因为每个电脑上文件的目录不尽相同。

### 文件的读写

#### 打开文件

* 文件打开的通用格式：

```python
with open("文件路径", "打开模式", encoding = "字符编码") as f:
    "对文件进行相应的读写操作"
    
```

* 上述格式被称为 **with 块**。使用with块的好处是执行完毕会自动关闭（close）文件。

* 下面给出一个简单的例子：

```python
with open("E:\ipython\测试文件.txt", 'r', encoding = 'gbk') as f:
    text = f.read()
    print(text)
```

* 上述open函数的第一个参数是文件路径，第二个 **r** 说明是读取（read）操作，编码是gbk编码。
* 接着使用 f 的 read 方法进行读取。
* 下面对 **open** 函数的参数进行详细说明：

##### 文件路径

* python支持两种路径参数：
* 绝对路径：就是使用上例中的完整路径名。
* 相对路径：仅在该代码（程序）文件所在文件夹内搜索。把程序和文件放入一个文件夹可简化路径名。

```python
with open("测试文件.txt", 'r', encoding = 'gbk') as f:
```

* 上面的例子说明路径和代码在同一个文件夹内。

##### 打开模式

* "r"，只读模式，若文件不存在，报错；
* "w"，覆盖写模式，若文件不存在，则创建；若文件存在，则会完全覆盖原文件；
* "x"，创建写模式，若文件不存在，则创建；若文件存在，报错；
* "a"，追加写模式，若文件不存在，则创建；若文件存在，在原文件后面追加内容；
* "b"，二进制文件模式，不能单独使用，需要配合使用，如"rb", "wb", "ab"等，该模式不需要指定encoding；
* "t"，文本文件模式，默认值，需配合使用，如"rt", "wt", "at"，一般省略。
* "+"，与 "r", "w", "x", "a"配合使用，在原功能基础上增加读写功能。具体后面再说。

* 打开模式缺省，默认为只读模式"r"。

##### 字符编码

* 万国码 utf-8
  * 包含全世界所有国家需要用到的字符
* 中文编码 gbk
  * 专门解决中文编码问题
* windows系统下，如果缺省，则默认为gbk（所在区域的编码）
* 为清楚起见，除了处理二进制文件，建议不要缺省encoding参数

#### 读入文件

* 读取整个文件内容——f.read()
  * 如果encoding写错，系统会无法解码，进行报错。

```python
with open('测试文件2.txt', 'r', encoding = 'utf-8') as f: # 'r' 可缺省，但尽量不要省略
    text = f.read()
    print(text)
```

* 逐行进行读取——f.readline()

```python
with open('测试文件2.txt', 'r', encoding = 'gbk') as f:
    for i in range(3):
        text = f.readline()
        print(text)
```

```python
with open('测试文件2.txt', 'r', encoding = 'gbk') as f:
    while True:
        text = f.readline()
        if not text:
            break
        else:
            print(text, end = '') # 去掉了print的默认换行
```

* 几个注意点：
  * 读取空行和读取到最后（结尾）是有区别的。如果原 txt 文件有空行，其 txt 文件中会用 '\n' 表示，并不是没有信息；而读到最后没有内容的时候是一个空字符串''。用这样的方法可以区分空行和文件最后。
  * 由于原 txt 文件中 '\n' 的存在，如果print函数没有去掉默认换行（end = ''），那么每次输出一行都会空一行。这是由于txt文件每行的结尾都会有一个换行符，而print本身每输出一次都会进行换行，因此输出一行都会再空一行。一般我们会把end = '' 参数加上，以保证正确换行，不要产生多余的空行。
* 读取所有行，以每行为元素生成一个列表—— f.readlines()
  * 下面给出两种f.readlines的换行函数的使用方法。

```python
with open('测试文件.txt', 'r', encoding = 'gbk') as f:
    text = readlines() # 注意每个list元素结尾都有换行符
    print(text)
```

```python
with open('测试文件.txt', 'r', encoding = 'gbk') as f:
    for text in f.readlines():
    	print(text, end = '') # 去掉空行
```

* 文件较大时的读取：直接使用 f 可迭代对象
  * 上一章讲到过可迭代对象的概念，那么 f 也是直接可以当做可迭代对象进行使用。下面介绍一下基本用法：
  * 下面的例子，每次读出的也是一行的内容。

```python
with open('测试文件.txt', 'r', encoding = 'gbk') as f:
    for text in f:
    	print(text, end = '') # 去掉空行
```

* 二进制文件：如图片就是二进制文件

```python
with open('test.jpg', 'rb') as f:
    print(len(f.readlines()))
```

#### 写入文件

* 向文件写入一个字符串或字节流（二进制）——f.write()

```python
with open("写入文件测试.txt", 'w', encoding = 'utf-8') as f:
    f.write('这是写入的第一行\n')  # 若文件不存在，则会创建文件
    f.write('这是写入的第二行\n')
    
"""
此时文件的内容是:
这是写入的第一行
这是写入的第二行
"""
with open("写入文件测试.txt", 'w', encoding = 'utf-8') as f:
    f.write('这是写入的第三行\n')  # 重新打开会覆盖原文件！意思是如果重新执行这段代码结果是一样的。
    f.write('这是写入的第四行\n')
    
"""
此时文件的内容是:
这是写入的第三行
这是写入的第四行
"""
```

* 上面的例子说明，每一次新打开一个文件进行写入时，上一次文件保存的数据会被直接覆盖。如果我们有时不需要覆盖，而要继续写，这样我们就必须使用**追加**的方法。

* 追加模式——'a'
  * 追加模式可以保证在原文件之后进行添加新内容。

```python
with open("写入文件测试.txt", 'a', encoding = 'utf-8') as f:
    f.write('这是写入的第三行\n') 
    f.write('这是写入的第四行\n')
"""
文件内容：
[ 原文本 ]
这是写入的第三行
这是写入的第四行
"""
```

* 将元素是字符串的列表整体写入文件——f.writelines()
  * 需要注意列表的每个元素最好加上换行符。

```python
ls = ['第一行', '第二行', '第三行', '第四行']
with open("写入文件测试.txt", 'a', encoding = 'utf-8') as f:
    f.writelines(ls) # 这种写法的结果会堆在一起。
    f.writelines([_ + '\n' for _ in ls]) # 这样就可以换行了
    
"""
第一种方法的文件内容：
第一行第二行第三行第四行
第二种方法的文件内容 ：
第一行
第二行
第三行
第四行
"""
```

#### 文件既读又写

* 'r+' 
  * 如果文件不存在，则报错；
  * 指针在文件开始（而不是结尾）；因此写之前一般要把文件指针移到文件末尾，否则写入时会对原内容进行覆盖！

```python
with open('test.txt', 'r+', encoding = 'gbk') as f:
    f.seek(0, 2) # 文件指针移动函数。第一个参数是偏移量，第二个参数是位置（0：文件开头；1：文件当前位置；2：文件结尾）
    			 # 如果不把文件指针定位到文件结尾，写入时文件会对原内容进行覆盖。
    text = ['增添第一行\n', '增添第二行\n']
    f.writelines(text)

"""
文件内容是：
增添第一行
增添第二行
[ 原文本 ]
"""
```

* 'w+'
  * 如果文件不存在，则创建文件；
  * 若文件存在，则会立刻清空原文件内容！

```python
with open('test.txt', 'w+', encoding = 'gbk') as f:
    pass # 这样也会清空原文件内容
```

* 'a+'
  * 如果文件不存在，则创建文件；
  * 指针在末尾，添加新内容不清空原内容。因此读之前一般要把指针移到文件开头，否则可能读不到内容。
  * 如果一开始就把指针放到开头再写，也会覆盖掉原内容。

```python
with open('test.txt', 'r+', encoding = 'gbk') as f:
    text = ['增添第一行\n', '增添第二行\n']
    f.writelines(text)
    f.seek(0, 0) # 将指针移到文件开头
    print(f.read()) # 打印所有行
    
"""
文件内容是：
[ 原文本 ]
增添第一行
增添第二行
"""
```



### 数据文件的存储与读取

* 本节主要介绍两种数据存储文件格式，csv和json

#### csv 格式

* csv 是一种通过**逗号**将数据分开的字符序列，可以通过 excel 打开
* 读取

```python
with open('成绩.csv', 'r', encoding='gbk') as f:
    ls = []
    for line in f: # 逐行读取法
        ls.append(line.strip('\n').split(',')) # strip函数可以去掉每行两侧的换行符；split函数将结果按逗号进行分割 
        
for res in ls:
    print(res) # 按行输出结果
```

* 写入

```python
ls = [['编号', '数学成绩', '语文成绩'], ['1', '100', '98'], 
      ['2', '96', '99'], ['3', '97', '95']]
with open('成绩.csv', 'w', encoding='gbk') as f: # 使用utf-8中文容易出现乱码
    for row in ls:
        f.write(','.join(row) + '\n') # 逗号组合成字符串，后加换行符
```

* 在模块部分将介绍csv模块的读取操作

#### json 格式

* json 格式常被用于存储字典类型数据。
* 写入——dump()

```python
import json

scores = {'Petter':{'math':96, 'physics':98},
          'Paul':{'math':92, 'physics':99},
          'Mary':{'math':98, 'physics':97} }
with open('score.json', 'w', encoding = 'utf-8') as f:
    json.dump(scores, f, indent=4, ensure_ascii=False)
    # indent 是一个换行缩进设置，在存储时设置缩进可以使预览更美观；
    # ensure_ascii 在字典中没有中文时设置为False，如果有中文则设置为True
```

* 读取——load()

```python
with open('score.json', 'r', encoding='utf-8') as f:
    scores = json.load(f) # 读取字典
    for k, v in scores.items(): # 遍历键值
        print(k, v)
```



## 异常处理

### 常见异常

* 下面列举了一些常见异常。但是实际应用中的异常还远不止这些。

#### 除0运算——ZeroDivisionError

```python
1/0 # 运行后会产生一个除零异常
```

#### 找不到可读文件——FileNotFoundError

```python
with open('nobody.csv') as f: # 如果没有该文件，就会报错
    pass
```

#### 值错误——ValueError

* 传入一个调用者不期望的值，即使这个值是合法的

```python
s = '1.3'
n = int(s) # 字符串上表示的数字是浮点型，无法调用int函数
```

#### 索引错误——IndexError

* 通常是下标超出序列边界

```python
ls = [1, 2, 3]
ls[5] # 索引错误，索引最多到2
```

#### 类型错误——TypeError

* 传入对象类型与要求不符

```python
1 + '3' # int 和 str 类型如何相加呢？
```

#### 其他异常

* 还有很多其他的异常，如**NameError**，使用一个未被定义的变量；**KeyError**，试图访问字典里不存在的键……等。

```python
print(a) # 如果根本没有定义变量a，则会产生NameError错误
d = {}
d['1'] # 试图访问一个根本不存在的键，会产生KeyError错误
```

* 当异常发生的时候，通常会将程序中断，使得程序无法正常运行。但是这是我们常常不希望的。因此python提供了一种对异常加以处理的方法。



### 异常处理

* 对异常进行必要的处理，可以提高程序的稳定性和可靠性。

#### 基本语句 try ... except ...

* 如果 try 内的代码块顺利执行，则 except 代码块不会被触发。
* 如果 try 内出现异常，则会触发 except 对应的代码块（如果异常类型与except上的一致）

```python
x = 10
y = 0
try:
    z = x / y
except ZeroDivisionError: # 预判一下会发生什么错误
    print('0不可以被除！')

# 输出：0不可以被除！ 程序不会异常中断
```

```python
x = 10
y = 0
try:
    z = x / y
except NameError:
    print('无该变量！')
    
# 虽然触发了ZeroDivisionError异常，但是没有触发NameError异常，所以程序还是会直接报错并中断。
```

* 需要注意的是这里预判的异常类型必须与产生的异常类型相一致才会报错。

#### 多分支结构 try ... except ... except ...

* 可以通过多个except语句形成多个预判和多个处理代码块。

```python
ls = []
d = {'name': 'Tom'}
try:
    # 这里把其他的注释掉，每个都试一次，看看效果
    y = m
    ls[3]
    d['age']
except NameError:
    print('变量名不存在！')
except IndexError:
    print('索引超出上限！')
except KeyError:
    print('键不存在')
```

#### 奖赏机制 try ... except ... else

* 如果try语句后面的代码块没有报错，那么就会执行else语句，如果报错，执行except代码块。

```python
try:
    with open('测试文件.txt') as f:
        text = f.read()
except FileNotFoundError:
    print('找不到该文件')
else:
    print(text)
```

#### 必执行语句 try ... except ... finally

* 无论try语句后面的代码块是否出错，finally都必须执行

```python
try:
    with open('测试文件.txt') as f:
        text = f.read()
except FileNotFoundError:
    print('找不到该文件')
else:
    print(text)
finally:
    print('读取完毕')
```

#### 万能异常（Exception）与异常捕获（as）

* 如果我们事先不知道会产生哪种类型的异常，可以使用万能异常 **Exception**：

```python
x = 10
y = 0
try:
    z = x / y
except Exception:
    print('发生未知错误') 

# 这样ZeroDivisionError依然会检测出来，走except代码块。
```

* 如果我们需要输出程序产生了什么异常，那么我们可以使用异常捕获的格式：

```python
x = 10
y = 0
try:
    z = x / y
except Exception as e:
    print('发生错误：' + str(e)) 

# 发生错误：division by zero   这样就可以输出错误原因。
```

* 异常处理的语句基本上就是这些。



* 至此，python语言从基础编程部分的基本语法到高级编程部分的相关的编程思路、底层实现、面向对象、文件、异常等特性都已经介绍完毕。可以说，我们向python编程又迈进了一大步。然而，要成为一个真正的python程序员，还有一个非常重要的内容需要熟悉，就是python的模块调用。我们将在下一篇笔记中加以探讨。



* Written by：Sirius. Lu
* Reference：深度之眼  python基础训练营
* 2020.6.19