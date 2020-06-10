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

```python
print(my_new_car.brand) # Audi
print(my_new_car.model) # A6
print(my_new_car.year)  # 2018
```

#### 调用方法

```python
my_new_car.get_main_information() # 品牌：Audi   型号：A6   出厂年份：2018
my_new_car.get_mileage() # 行车总里程：0公里
```

