# Python 学习笔记（一）基础编程

* 本笔记 # 后为该语句的输出结果，或该变量的值。若 # 后接 ! 号，意思是该语句不能这样写。
* 对于多行的输出结果，我会用""" """进行注释。
* 对于一些输出结果，笔记中为方便理解会在一个代码块写出所有的输出语句，实际调试中应该仅保留一个输出语句（格式化输出print除外），否则前面的输出会被最后一个输出语句覆盖。



**目录**

[toc]

## 基本语法

### 数据类型概览

#### 基本数据类型

* 数字类型
  * 整数类型 3
  * 浮点数类型 2.5 、复数类型 3 + 4j
* 布尔类型 True  False（注意第一个字母大写）
* 字符串类型 'python'   "Yes"   "中国"   不可变类型！



#### 组合数据类型

* 列表类型：list
  * [1, 2, 3, 4]
  * 有位置和顺序，支持索引、修改等操作
* 元组类型：tuple
  * (1, 2, 3, 4)
  * 有位置和顺序，不支持修改，只可读
* 字典类型：dict
  * {1: 'a', 2: 'b', 3:'c'}
  * 键 key —— 值 value 各个元素都是键值对的形式
  * 无序，不能索引，但可以通过键获得对应的值。
  * 各个键之间不能相同，即使赋值时有相同键，也只会保留最后一个键值对（而不会报错）。
* 集合类型：set
  * {'a', 'b', 'c'}
  * 各个元素不能相同。



### 变量

* 变量在python中等同于对象。

* 如何命名：大写字母、小写字母、数字、下划线、汉字及其组合。

  * 首字母不能为数字

  * 变量名中间不能有空格

  * 不能与Python保留字相同

  * python保留字的查看方法：

    ```python
    help("keywords")
    ```

* 变量名定义的几个小技巧：

  * 变量名尽可能有实际意义，尽量使用英文单词，避免使用中文和中文拼音

  * 下划线定义法（通常用于变量和函数名）用下划线区分各个单词

    ```python
    # 下划线定义法示例：
    # 函数名：show_all_images、get_results
    # 变量名：count_numbers
    ```

  * 驼峰表示法（通常用于类名）单词首字母大写；用于变量和函数名时从第二个单词开始首字母大写。

    ```python
    # 驼峰表示法示例：
    # 类名：PlaceSetting、DessertInformationSystem
    # 函数名：showAllImages、endOfTheLine
    # 变量名：countAllNumbers
    ```

  * 对于常量，可以用所有字母都大写的方式来定义，但是它实际还是一个变量，通过这种方法可以告诉别人不要轻易的去改它。

    ```python
    MAX_ITERATION = 1000
    PI = 3.14
    ```

* 赋值方法

  ```python
  a = 1 + 2  # a = 3   直接赋值法
  a += 1     # a = 4   增量赋值法
  # ! a++、++a 不可以这样赋值
  a, b = 2, 3  # a = 2, b = 3  打包赋值法
  a, b = b, a  # 打包赋值法可以实现变量的直接交换。
  ```



### 控制流程

* for 循环

    ```python
    res = 0
    for i in [1, 2, 3, 4, 5]:
        res += i
    res   # 15
    ```

* while 循环

    ```python
    i = 1
    res = 0
    while i <= 5:
      res += i
      i += 1
    res   # 15
    ```

* if 判断条件

    ```python
    age = 18
    if age < 18:
        print("未成年")
    elif age == 18:
        print("刚刚成年")
    else:
        print("已成年")
    ```



### 数据输入与输出

* 动态交互输入 input

    ```python
    x = input("请输入一个数字") # 交互框：请输入一个数字：
    x  # 输出你输入的数字 
    # ! x += 10 字符串不能与数字相加
    ```

    * 注意这里的x是字符串类型。
    * 用eval()函数可以转化为整型。

    ```python
    x = eval(input("请输入一个数字"))
    x += 10
    ```

* 直接显示数据

    ```python
    x = 10
    x       # 输出x
    ```

* 打印输出 print

    ```python
    x = 10
    print(x) # 输出10
    ```

    * 每调用一次print函数，都会对结果进行换行，如果不想换行应该添加end参数。

    ```python
    print(123, end = "")
    print(456)       # 输出123456
    print(123, end = " ")
    print(456)       # 输出123 456
    ```

    * 组合输出

    ```python
    a, b = 1, 2
    print("a = ", a, " b = ", b)  # a = 1  b = 2   逗号分隔输出法
    print("a = {0}, b = {1}".format(a, b)) # a = 1, b = 2   格式化输出法
    print("b = {1}, a = {0}".format(a, b)) # b = 2, a = 1   顺序可以变化
    print("a = {}, b = {}".format(a, b)) # a = 1, b = 2   可以不写顺序
    ```

    * 格式化输出还有很多方法，这里就不详细介绍了。



### 书写格式

* 行最大长度：79字符
* 缩进：
    * python中的缩进非常重要，表明代码和前句之间的从属关系。
    * while、if、for等语句必须合理使用缩进，否则很可能产生错误的运行结果。
    * 缩进量：4个字符。

* 使用空格：

    * 二元运算符两端常加一个空格：

    ```python
    a > 3    # 一般不写成 a>3
    b += 10  # 一般不写成 b+=10
    ```

    * 在逗号后添加空格：

    ```python
    x, y = 1, 2
    ls = [1, 2, 3]
    ```

    * 不要使用一个以上的空格：

    ```python
    # ! x      = 2
    ```

    * 在制定关键字参数的时候不要加空格：

    ```python
    def fun(n=1, m=2):
        print(n, m)
    ```

    

## 基本数据类型

### 数据类型查看与转换

#### 查看数据的类型

* type函数可查看数据或变量的类型：

  ```python
  a, b, c, d, e = 3, 2.5, 2+3j, True, 'abcde'
  type(a) # int 整数
  type(b) # float 浮点数
  type(c) # complex 复数
  type(d) # bool 布尔
  type(e) # str 字符串
  f = [1, 2, 3]
  g = (1, 2, 3)
  h = {2: '小明', 3: '小红', 4: '小张'}
  i = {'a', 'b', 'c'}
  type(f) # list 列表
  type(g) # tuple 元组
  type(h) # dict 字典
  type(i) # set 集合
  ```

* isinstance函数（变量，比较的类型）

  * 该函数能判断某变量是否属于一个类型。
  * 该函数能识别继承而来的对象，即承认继承。

  ```python
  age = 20
  name = "Ada"
  isinstance(age, int)     # True
  isinstance(age, object)  # True   所有变量都继承自object类
  isinstance(name, object) # True   所有变量都继承自object类
  ```



#### 数据类型转换

* 数字转字符串  str()

  ```python
  age = 20
  print("My age is " + str(age)) # My age is 20
  ```

* 仅有数字组成的字符串转数字  int()   float()   eval()

  ```python
  s1 = '20'
  s2 = "10.1"
  int(s1)     # 20
  float(s1)   # 20.0
  eval(s1)    # 20
  
  # ! int(s2) 将s2转换成int会直接报错
  float(s2)   # 10.1
  eval(s2)    # 10.1
  ```



### 数字类型

#### 进制转换

* 各种进制的表示：

   * 十进制decimal：16
    * 二进制binary：0b10000 （开头为0b）
    * 八进制octal：0o20（开头为0o）
    * 十六进制hex：0x10（开头为0x）

* 十进制进制转换为其他进制：（例子）

   ```python
   a = bin(16) # 0b1000
   b = oct(16) # 0o20
   c = hex(16) # 0x10
   ```

   注：上述输出结果为字符串类型，并非整数！（尝试 a == b == c）

* 将其他进制数转化成十进制：（接上例）

   ```python
   d = int(a, 2) # 16
   e = int(b, 8) # 16
   f = int(c, 16) # 16
   ```



#### 浮点数的不确定性

* 尝试：0.1 + 0.2 == 0.3   --> False

* 原因：计算机采用二进制小数表示浮点数的小数部分，而部分小数不能用二进制等值表示。

   * 例子：0.1 + 0.7 = 0.79999999

* 影响：通常情况下不会影响计算精度。

* 处理/优化：四舍五入获得精确解：

   ```python
   a = 3 * 0.1
   print(a) # 0.3000000000004
   ```

   ```python
   b = round(a, 1)
   print(b) # 0.3
   b == 0.3 # True
   ```

   round函数的用法：第一个参数为原数，第二个参数为保留的小数位数。属于内置函数，不需要调用任何模块。



#### 复数的表示

* 复数符号用 **j** 和 **J** 均可。

   ```python
   1+2j
   3+4J
   ```

* 当虚部为1时，复数符号j不可省略。

   ```python
   2+1j  # ! 2+j
   ```



#### 操作符

* 取反  `-`

  ```python
  x = 1
  -x # -1
  ```

* 乘方  `**`

  ```python
  2**3 # 8
  ```

* 整除 `//` 和 取模  `%`

  ```python
  13 // 5  # 2
  13 % 5   # 3
  ```

* 注意：

  * 整数与浮点数的运算结果是浮点数。

    ```python
    2 + 1.5 # 3.5 --> 浮点数
    ```

  * 除法运算 `/`  的结果是浮点数。 

    ```python
    8 / 4 # 2.0 --> 浮点数
    ```



#### 数学函数

* 绝对值（对复数是求模）abs(x)

  ```python
  abs(-5) # 5
  abs(3+4j) # 5.0
  ```

* 幂次方 pow(x, n)  pow(x, n, m)

  ```python
  pow(2, 5) # 2的5次方，与2**5等价
  pow(2, 5, 3) # 2的5次方的结果求余3。用该方法计算比使用运算符更加快捷。
  ```

* 四舍五入 round(x) round(x, n)

  ```python
  a = 1.618
  round(a) # 2 默认四舍五入为整数
  round(a, 2) # 1.62 保留2位小数
  round(a, 5) # 1.618 超过位数的不会进行补齐，直接使用原数返回。
  ```

* 同时求整除和取模 divmod(x, y)

  ```python
  divmod(13, 5) # (2, 3) 返回的第一个值是整除结果，第二个值是取模结果。
  ```

  注意：

  * divmod返回值是一个元组。
  * divmod计算等价于(x // y, x % y)，但是前者更快，因为仅仅做了一次除法。

* 序列最大最小值 max() min()

  ```python
  max(1, 2, 3, 4, 5) # 5 直接把数字作为参数
  a = [1, 2, 3, 4, 5]
  max(a) # 5 将列表作为参数
  ```

* 求和 sum()

  ```python
  sum([1, 2, 3, 4, 5]) # 15
  # ! sum(1, 2, 3, 4, 5) 不能直接把数字作为参数！
  ```



#### math 模块与 numpy 模块简介（仅仅展示一部分）

* math 库

  ```python
  import math
  math.exp(1) # 2.71828.... e指数运算
  math.log2(2) # 1.0 对2取对数 对其他底数取对数的函数是math.log()
  math.sqrt(4) # 2.0 开平方
  ```

* numpy 库

  ```python
  import numpy as np
  a = [1, 2, 3, 4, 5]
  np.mean(a) # 3.0 求均值
  np.median(a) # 3.0 求中位数
  np.std(a) # 1.414... 求标准差
  ```



### 字符串类型

#### 字符串表达

* 双引号或单引号

  ```python
  'python'
  "python"
  ```

* 双中有单：字符串中有单引号的情况

  ```python
  "'python' is a useful language."
  ```

* 单中有双：字符串中有双引号的情况

  ```python
  '"python" is a useful language.'
  ```

* 转义字符的使用 `\`

  ```python
  '\"python\" is a useful language.'
  ```

  特殊用途：用于字符串的换行

  ```python
  'py\
  thon' # 输出 'python'
  ```



#### 字符串索引

* 正向索引与反向索引：（索引的符号是`[]`）

  ```python
  s = 'My name is Peppa Pig.'
  
  # 字符串长度
  len(s) # 21 返回字符串的长度
  
  # 正向索引
  s[0] # 'M'  正向索引从0开始，因此s[0]找到第一个字符。
  s[2] # ' '  空格也是字符
       # ! s[21]   超过字符长度会报错
  
  # 反向索引
  s[-1] # '.' 反向索引从-1开始，因此s[-1]找到最后一个字符。
  s[-3] # 'i'
        # 同样反向索引页不能超过长度
  ```

* 字符串切片 [开始位置：结束位置：切片间隔]

  ```python
  s = 'Python3'
  s[0:3:1]  # 'Pyt' 从位置0('P')到位置3('h')切片。但是Python中的区间都使用的是左闭右开
  					#（即不包含结束位置），因此（'h'）不在结果中。
  s[0:6:2]  # 'Pto'
  s[0:3]    # 'Pyt' 切片可以不写切片间隔，此时默认间隔为1，即全部保留。
  s[1:]     # 'ython3' 切片可以不写结束位置，默认取到最后
  s[:3]     # 'Pyt' 切片可以不写开始位置，默认从头开始
  s[:]			# 'Python3' 切片可以都不写，那么表示取全部字符
  ```

  * 难点1：带反向索引的切片

  ```python
  # 该例子接上例
  s[-3:-1:1] # 'on' 从倒数第3个位置('o')到倒数第一个位置('3')，但是结束位置不取，因此'3'不在结果中
  s[-5:]     # 'thon3'
  s[:-1]     # 'Python'
  ```

  * 难点2：反向切片

  ```python
  s = "123456789"
  s[-1:-10:-1] # '987654321' 由于间隔是-1所以向左取。从最后一个位置开始取到倒数第10个位置。
  			 # 要注意的是这里只有9个字符，虽然倒数第10个位置不存在，但是由于是开区间，所以
    			 # 实际只会取到倒数第9个字符。这样写是符合语法规范的，不会报错。
  s[:-8:-1]    # '9876543'
  s[::-1]      # '987654321'
  ```

* 注意：字符串是不可变类型，因此不能直接用索引进行修改！

  ```python
  s = 'Python'
  # ! s[3] = 'a'
  s = s[3:5]       # 这样做是可以的
  s = 'Tensorflow' # 对整个字符串进行重新赋值是可以的
  ```

  

#### 字符串操作符

* 字符串拼接 `+`

  ```python
  s1 = 'pyth'
  s2 = 'on3'
  s1 + s2    # 'python3'
  ```

* 字符串重复 `*`

  ```python
  s = 'abc'
  s1 = s * 3 # 'abcabcabc'
  s2 = 3 * s # 'abcabcabc'
  ```

* 判断某字符串是否在子字符串中 `in`

  ```python
  names = "Mike, Tom and Jack"
  "Peter" in names    # False
  "Jack" in names     # True
  ```

  * 应用：利用循环遍历字符串

    ```python
    for s in "Python":
      print(s, end = " ") 
    # P y t h o n
    ```



#### 常用字符串操作函数

* 求字符串的长度 len(str)

  ```python
  s = 'Python'
  len(s)   # 6
  ```

* 字符编码转换

  * Python默认字符使用Unicode编码，有时我们需要将其转化成它的唯一编码。

  * 将字符转换成Unicode码 ord(转换的字符)

    ```python
    ord("1")  # 49
    ord('a')  # 47
    ord("中")  # 20013 
    ```

  * 将Unicode码转换成字符 chr(Unicode码)

    ```python
    chr(23456) # 宠
    ```

* 字符串的分割   str.split("分割字符")

  ```python
  s = "I love Python very much"
  s.split(" ") # ['I', 'love', 'Python', 'very', 'much']
               # split方法需要给出一个参数，这个参数是分割符；
    			 # 该方法返回一个列表；
  s    		 # 'I love Python very much' 该方法不会修改原字符串。
  ```

* 字符串的聚合  "聚合字符".join(可迭代数据类型)

  * 可迭代数据类型包括：字符串、列表、元组等
  * 只有字符可以进行聚合！（数字不可以）

  ```python
  s = "12345"
  ",".join(s)  # '1,2,3,4,5'
  l = ["1", "2", "3"]
  '*'.join(l)  # '1*2*3'
  # ! l = [1,2,3]  只有字符可以聚合！
  ```

* 删除两端字符   str.strip(待删除的字符)

  * lstrip和rstrip也是类似的用法

  ```python
  s = '    many blanks       '
  s.strip(" ")     # 'many blanks' 删除两端的空格
  s.lstrip(" ")    # 'many blanks       '  只删除左端空格
  s.rstrip(" ")    # '    many blanks' 删除右端空格
  ```

* 字符串的替换  str.replace(被替换的str, 替换成的str)

  ```python
  s = 'Python is very good'
  s.replace("Python", "py") # 'py is very good'
  ```

* 字符串的统计 str.count(待统计的str)

  ```python
  s = "Python is very good"
  s.count("o") # 3
  ```

* 字符串的大小写 str.upper()  str.lower()  str.title()

  ```python
  s = "Python is good."
  s.upper()   # 'PYTHON IS GOOD.'  全部大写
  s.lower()   # 'python is good.'  全部小写
  s.title()   # 'Python Is Good.'  每个单词首字母大写
  ```

* 判断字符串是否只由数字组成 str.isdigit()

  ```python
  s1 = "Python"
  s2 = "123456"
  s3 = "Python123456"
  s1.isdigit()  # False
  s2.isdigit()  # True
  s3.isdigit()  # False
  ```

* 判断字符串是否只由字母组成 str.isalpha()

  ```python
  s1 = "Python"
  s2 = "123456"
  s3 = "Python123456"
  s1.isalpha()  # True
  s2.isalpha()  # False
  s3.isalpha()  # False
  ```

* 判断字符串是否只由数字和字母组成 str.isalnum()

  * 该方法可以用来判断用户名是否合法

  ```python
  s1 = "Python123"
  s2 = "Python123__"
  s1.isalnum()       # False
  ```



### 布尔类型

#### 基本用法

* 布尔类型用于表示逻辑运算的结果

  ```python
  a = 10
  a > 8   # True
  a == 12 # False
  a < 9   # False
  ```

* any() 函数与 all() 函数

  * any() 是否有元素为True
  * all() 是否所有元素都为True

  ```python
  any([False, 1, 0, None])  # False    注意，0, False, None 都可以表示“无”的意思
  all([False, 1, 0, None])  # True     注意，非0元素也看作为True
  ```



#### 布尔类型的应用

* 利用布尔类型进行条件判断：

  * 猜数字小例子：

```python
  n = 200
  while True:
  m = eval(input("请输入一个整数"))
  if m == n:
      print("猜对啦")
      break
  elif m > n:
      print("太大了")
  else:
      print("太小了")
```

* 利用布尔类型进行批量筛选

  ```python
  import numpy as np
  a = np.array([1, 3, 5, 7, 9])
  a > 4     # [False False True True True]
  a[a > 4]  # array([5, 7, 9])
  ```



## 组合数据类型

### 列表 list

* 列表是Python中用到的最多的组合数据类型。



#### 列表的表达

* 列表的特点：
  * 列表属于序列类型，内部的元素之间有位置关系，因此能通过位置索引访问。
  * 列表不同于C语言的数组，可以在一个列表中同时使用多种不同类型的元素。
  * 列表属于可变类型，支持元素的增删改查等操作。

  ```python
  ls = ["Python", 20, True, {"Version": 3.7}] # 这样定义是正确的
  ```

* 用list产生列表：list(可迭代对象)

  * 字符串转列表：

    ```python
    list("编程语言") # ['编', '程', '语', '言']
    ```

  * 元组转列表：

    ```python
    list((1, 2, 3, 4)) # [1, 2, 3, 4]
    ```

  * 集合转列表：

    ```python
    list({"Jack", "Mike", "Tom"}) # ['Jack', 'Tom', 'Mike']
    ```

* range函数获得序列 range( 起始数字, 中止数字, 数字间隔 )

    * 起始数字可以省略，默认为0；
    * 数字间隔可以省略，默认为1；
    * 中止数字不可以省略，且获得的序列不包含中止数字

    ```python
    for i in range(6):
        print(i, end = " ")
    # 0 1 2 3 4 5       
    ```

* 将range与list配合转化成列表：

    ```python
    list(range(6)) # [0, 1, 2, 3, 4, 5]
    ```

    

#### 列表的性质和基本函数

* 求长度  len()   ——同字符串类型

* 索引、正向切片、反向切片  ——同字符串类型

* 列表拼接 `+`   ——同字符串类型

    ```python
    a = [1, 2]
    b = [3, 4]
    a + b      # [1, 2, 3, 4]
    ```

* 重复复制 `*` ——同字符串类型

    ```python
    [0] * 5   # [0, 0, 0, 0, 0]
    ```

    * 可用于初始化列表



#### 列表的常用操作

* 增加元素

    * 在列表的末尾增加元素 list.append(增加元素)

    ```python
    languages = ["Python", 'R', 'C++']
    languages.append("Java")
    languages  # ['Python', 'R', 'C++', 'Java']
    ```

    * 在列表的任意位置增加元素 list.insert(位置索引，待增元素)
        * 在列表相应位置之前插入元素

    ```python
    # 接上例
    languages.insert(1, "C")
    languages  # ['Python', 'C', 'R', 'C++', 'Java']
    ```

    * 在列表的末尾加入另一个列表  list1.extend(list2)
        * 注意该操作与append的区别。

    ```python
    # 接上例
    languages.extend(["PHP", "Go", "C#"])
    languages  # ['Python', 'C', 'R', 'C++', 'Java', 'PHP', 'Go', 'C#']
    languages.append(["PHP", "Go", "C#"]) # 与extend操作的对比
    languages  
    # ['Python', 'C', 'R', 'C++', 'Java', 'PHP', 'Go', 'C#', ['PHP', 'Go', 'C#']]
    ```

* 删除元素

    * 删除列表某位置的元素  list.pop(索引)
        * 若省略索引参数，则默认删除最后一个元素

    ```python
    languages = ['Python', 'C', 'C++', 'R', 'Java']
    languages.pop(1)
    languages   # ['Python', 'C++', 'R', 'Java']
    languages.pop()
    languages   # ['Python', 'C++', 'R']
    ```

    * 删除列表中第一次出现的某元素  list.remove(待删元素)

    ```python
    languages = ['Python', 'C', 'R', 'C', 'Java']
    languages.remove("C")
    languages   # ['Python', 'R', 'C', 'Java']
    ```

    * 如何删除列表中所有出现的该元素呢？

    ```python
    languages = ['C', 'Python', 'R', 'C', 'Java', 'C', 'PHP']
    while "C" in languages:
        languages.remove("C")
    languages   # ['Python', 'R', 'Java', 'PHP']
    ```

* 查找元素

    * 查找列表中第一次出现待查元素的位置  list.index(待查元素)

    ```python
    languages = ['Python', 'C', 'R', 'Java']
    index = languages.index("R")
    index  # 2
    ```

* 修改元素

    * 索引赋值法  list[索引] = 新值

    ```python
    languages = ['Python', 'C', 'R', 'Java']
    languages[1] = 'C++'
    languages  # ['Python', 'C++', 'R', 'Java']
    ```

* 列表的复制

    * 错误的方式——浅拷贝

    ```python
    languages = ['Python', 'C', 'R', 'Java']
    languages_2 = languages
    languages_2  # ['Python', 'C', 'R', 'Java'] 看起来是正确的复制
    
    languages.pop() # 对languages删除一个元素
    languages_2  # ['Python', 'C', 'R']   languages_2也发生了变化
    ```

    * 正确的方式——深拷贝
        * 方法1：list.copy()
        * 方法2：list[:]

    ```python
    languages = ['Python', 'C', 'R', 'Java']
    languages_2 = languages.copy()  # 运用copy方法进行深拷贝
    languages_3 = languages[:]      # 运用切片[:]进行深拷贝
    ```

* 列表的排序  list.sort()

    * 通过reverse参数来设置递增或递减排序

    ```python
    ls = [2, 5, 2, 8, 19, 3, 7]
    ls.sort()
    ls  # [2, 2, 3, 5, 7, 8, 19]
    ls.sort(reverse = True)
    ls  # [19, 8, 7, 5, 3, 2, 2]
    ```

    * sorted(list) 函数对列表进行临时排序，返回排序后的列表，原列表不变

    ```python
    ls = [2, 5, 2, 8, 19, 3, 7]
    ls_2 = sorted(ls)
    ls_2  # [2, 2, 3, 5, 7, 8, 19]
    ls    # [2, 5, 2, 8, 19, 3, 7]
    ```

* 列表的翻转   list.reverse()

    * list.reverse() 做的是永久性翻转
    * 可以用切片方式对列表进行临时翻转（原列表不变）

    ```python
    ls = [1, 2, 3, 4, 5]
    ls[::-1]  # [5, 4, 3, 2, 1]
    ls.reverse()  # [5, 4, 3, 2, 1]
    ```

* 使用for循环对列表进行遍历：

    ```python
    # 接上例
    for i in ls:
        print(i, end = " ")
    # 5 4 3 2 1 
    ```



### 元组 tuple

#### 元组的表达

* 元组可以使用多种类型元素，一旦定义，内部元素不支持增、删、修改操作。
* 可以将元组视为"不可变列表"。

```python
names = ("Peter", "Jack", "Mary")
```



#### 元组的操作

* 不支持元素增加、删除、修改操作
* 其他操作与列表完全一致



#### 元组的常见用处

* 打包与解包

    * 函数的打包与解包

    ```python
    def fun(x):
        return x**2, x**3
    
    fun(3) # (9, 27) 打包返回
    
    a, b = fun(3) # a = 9, b = 27 解包赋值
    ```

    * zip函数的打包和解包

    ```python
    numbers = [201, 202, 203]
    name = ["Mike", "Linda", "Mary"]
    list(zip(numbers, name)) # [(201, 'Mike'), (202, 'Linda'), (203, 'Mary')]  对应元素打包
    
    for number,name in zip(numbers, name):
        print(number, name)    
    """
    解包输出
    201 Mike
    202 Linda
    203 Mary
    """
    ```



### 字典 dict

#### 字典的表达

* 映射类型：通过 键-值 映射实现数据的存储和查找
* 常规的字典是无序的

```python
students = {201901: '小明', 201902:'小红', 201903:'小强'}
```

* 字典键的要求：

    * 字典的键不能重复

    ```python
    students = {201901: '小明', 201901:'小红', 201903:'小强'} 
    students  # {201901: '小红', 201903: '小强'}
    ```

    * 字典的键必须是不可变类型，如果键可变，就找不到对应存储的值了。
        * 总结一下可变与不可变类型：
            * 不可变类型：数字、字符串、元组。
            * 可变类型：列表、字典、集合。
        * 举一些例子：

    ```python
    d1 = {1: 3}
    d2 = {"s": 3}
    d3 = {(1,2,3): 3}
    # ! d = {[1,2]: 3}  不合法，列表作为字典键
    # ! d = {{1:2}: 3}  不合法，字典作为字典键
    # ! d = {{1,2}: 3}  不合法，集合作为字典键
    ```



#### 字典的性质

* 字典的长度——键值对的个数

```python
students = {201901: '小明', 201902:'小红', 201903:'小强'} 
len(students) # 3
```

* 字典的索引——通过字典键获得值

```python
students = {201901: '小明', 201902:'小红', 201903:'小强'} 
students[201902] # '小红'
```



#### 字典的常用操作

* 增加键值对（索引法）  dict[new_key] = new_value

```python
students = {201901: '小明', 201902:'小红', 201903:'小强'} 
students[201904] = '小雪'
students # {201901: '小明', 201902: '小红', 201903: '小强', 201904: '小雪'}
```

* 删除键值对   

    * del 删除法  del dict[key]

    ```python
    students = {201901: '小明', 201902:'小红', 201903:'小强'} 
    del students[201903]
    students # {201901: '小明', 201902: '小红'}
    ```

    * dict.pop(key)
        * 该函数的返回值为该键对应的值

    ```python
    students = {201901: '小明', 201902:'小红', 201903:'小强'} 
    value = students.pop(201903) 
    students # {201901: '小明', 201902: '小红'}
    value    # '小强'
    ```

    * 随机删除 dict.popitem()
        * 随机删除一个键值对，并以元组返回删除的键值对

    ```python
    students = {201901: '小明', 201902:'小红', 201903:'小强'} 
    key, value = students.popitem()
    key, value  # (201903, '小强')
    students    # {201901: '小明', 201902: '小红'}
    ```

* 修改值（索引法） dict[key] = new_value

```python
students = {201901: '小明', 201902:'小红', 201903:'小强'} 
students[201902] = '小雪'
students  # {201901: '小明', 201902: '小雪', 201903: '小强'}
```

* dict.get() 函数

    * 该函数有两个参数，即dict.get(key, default)
    * 该函数的意义是从dict中找key对应的值，若找到了则返回这个值，否则返回default
    * 该函数可以用于统计字符出现的频率：

    ```python
    s = "牛奶奶找刘奶奶买牛奶"
    d = {}
    for i in s:
        d[i] = d.get(i, 0) + 1
    d  # {'牛': 2, '奶': 5, '找': 1, '刘': 1, '买': 1}
    ```

* dict.keys()  dict.values()  方法

    * 这两个函数用于得到字典的所有键（或所有值），可将其转化成列表进行进一步处理

    ```python
    students = {201901: '小明', 201902:'小红', 201903:'小强'} 
    list(students.keys())    # [201901, 201902, 201903]
    list(students.values())  # ['小明', '小红', '小强']
    ```

* dict.items() 方法与字典的遍历

```python
# 接上例
list(students.items()) # [(201901, '小明'), (201902, '小红'), (201903, '小强')]
for k,v in students.items():
    print(k, v)
"""
遍历输出：
201901 小明
201902 小红
201903 小强
"""
```



### 集合 set

#### 集合的表达

* 一系列互不相等的元素的无序集合
* 元素必须都是不可变类型：数字、字符串、元组，与字典的键类似
* 可以看作没有值、或值为None的字典

```python
students = {"小明", "小红", "小强", "小明"}
students # {'小强', '小明', '小红'}
```



#### 集合的运算

* 交集 `&` 与并集 `|`

```python
Chinese_A = {'Jack', 'Mary', 'Linda', 'Bob', 'Mike'}
Math_A = {'Mary', 'Bob', 'Sheldon', 'Jessy'}
Chinese_A & Math_A  # {'Bob', 'Mary'}  两门都是A的同学
Chinese_A | Math_A  # {'Bob', 'Jack', 'Jessy', 'Linda', 'Mary', 'Mike', 'Sheldon'}
					# 至少有一门为A的同学
```

* 非共同元素（交集的补集）`^`

```python
# 接上例
Chinese_A ^ Math_A  # {'Jack', 'Jessy', 'Linda', 'Mike', 'Sheldon'}  只有1门是A的同学
```

* 差集  S`-`T   在集合S中而不在T中元素的集合

```python
# 接上例
Chinese_A - Math_A # {'Jack', 'Linda', 'Mike'} 只有语文得A的同学
```



#### 集合的常用操作

* 增加元素 set.add(增加的元素)

```python
students = {'Jack', 'Mary', 'Linda', 'Bob'}
students.add('Mike')
students # {'Bob', 'Jack', 'Linda', 'Mary', 'Mike'}
```

* 移除元素 set.remove(移除的元素)

```python
students.remove('Jack')
students # {'Bob', 'Linda', 'Mary', 'Mike'}
```

* 集合的长度 len(set)

```python
len(students) # 4
```

* 集合的遍历

```python
for stu in students:
    print(stu)
    
"""
Mary
Linda
Mike
Bob
"""
```



## 程序控制结构

* （缩减版，与C类似的省略）

### 条件测试

* 非顺序性的程序控制，往往需要根据一定的条件，决定程序运行的路线。

#### 比较运算

* 数据关系比较：< 	> 	<= 	>= 	== 	!=

* 非空比较：

    * 当变量=0、变量=False、变量=None、变量为空的容器时，我们认为该变量为空。
    * 判断容器为空的代码如下：

    ```python
    ls = []
    if ls:   # 判断其是否非空
        print("not empty")
    else:
        print("is empty")
    ```

#### 逻辑运算

* 与 `and` 或 `or` 非 `not`
    * 用法和c语言相同
    * 优先级：not > and > or

```python
a = 10
b = 8
c = 12
a > b and b > c  # False
a > b or b > c   # True
not(a > b)       # False

True or True and False  # True
(True or True) and False  # False
```

#### 存在运算

* 元素 in 可迭代类型（列表/字符串/元组等）

```python
car = ['BYD', 'BMW', 'AUDI', 'TOYOTA']
'BMW' in car  # True
'BENZ' in car  # False
```

* 元素 not in 可迭代类型

```python
# 接上例
'BMW' not in car # False
'BENZ' not in car # True
```



### 分支结构

#### 单分支

```python
age = 8
if age > 7:
    print('上学')
```

#### 二分支

```python
age = 6
if age > 7:
    print('school')
else:
    print('play')
```

#### 多分支

* 不管有多少分支，最终只执行一个分支

```python
score = 75
if score >= 95:
    print('A+')
elif score >= 90:
    print('A')
elif score >= 80:
    print('B')
elif score >= 70:
    print('C')
else:
    print('D')
```

#### if 的嵌套

```python
scoreA = 85
scoreB = 90
if scoreB > 90:
    if scoreA > 90:
        print('excellent')
    else:
        print('good')
else:
    print('not so good')
```



### 循环结构

#### for 循环

* 主要形式：

    * for 元素 in 可迭代对象:

        ​      执行语句

    * 依次取出可迭代对象的每一个元素，并进行相应的操作

* 直接迭代——列表[]、元组()、集合{}、字符串" "

```python
students = ('Jack', 'Mary', 'Jim')
for stu in students:
    print('Congratulations! ' + stu)
"""
Congratulations! Jack
Congratulations! Mary
Congratulations! Jim
"""
```

* 变换迭代——字典
    * 使用dict.items()方法获取每个键值；
    * 使用dict.keys()方法获得键。

```python
students = {201901: 'Jack', 201902: 'Mary', 201903: 'Jim'}
for k, v in students.items():
    print(k, v)
"""
201901 Jack
201902 Mary
201903 Jim
"""    
for stu in students: # !这样写迭代的是keys，虽然不会报错，但是我们一般不这么写（可读性差）
    print(stu)
"""
201901
201902
201903
"""
# 正确的写法：
for stu in students.keys():
    print(stu)

```

* range() 对象

```python
res = []
for i in range(10):
    res.append(i**2)
res # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

* 循环控制：`break` `continue` 和C语言用法完全一致。

```python
scores = [97, 28, 30, 47, 86, 80]
for i in range(len(scores)):  # 使用range(len())结构实现用位置信息对列表进行遍历
    if scores[i] > 30 and scores[i] < 50:
        print(i)
        break
# 3
```

* 重点：for循环与else的配合
    * else可以用在for循环之后，表示：
        * 当for循环没有做break全部正常执行完毕后，执行else语句；
        * 当for循环中做了break跳出时，不执行else语句。

```python
scores = [60, 79, 65, 90, 95, 88]
for score in scores:
    if score < 60:
        print("有不及格的同学")
        break
else:
    print("全部及格")
# 全部及格
```

#### while 循环

* while 条件:

    ​      语句

* 除形式和C语言有所不同外，其他都基本一致。

* 也可以使用break continue

* 也可以与else进行配合。



### 控制语句注意问题

* 尽量少使用多层嵌套。可读性差。
* 避免死循环。
* 不要使用过于复杂的判断条件。



## 函数

* 模块化设计思想



### 函数的基本使用

#### 函数的定义与调用

##### 定义

def  函数名( 参数 ) :

​        函数体

​        return  返回值

```python
def area_of_square(length_of_side):  # 求正方形的面积
    square_area = pow(length_of_side, 2)
    return square_area
```

##### 调用

函数名( 参数 )

```python
area = area_of_square(5) # 25
```

#### 函数的参数传递

* python函数也存在实参和形参的问题。和C语言一致。
* 位置参数：严格按照位置顺序进行赋值，一般用在参数比较少的时候。
    * 调用时给出的参数过少，会报错

```python
def fun(x, y, z):
    print(x, y, z)
    
fun(1, 2, 3)  # 参数一个也不能多，一个也不能少
```

* 关键字参数：打破位置限制，直呼其名。
    * 但是依然要求实参数量 = 形参数量
    * 用在参数比较多的场合

```python
fun(y=2, x=1, z=3)
```

* 位置参数可以和关键字参数混合使用，但是位置参数必须放在关键字参数前面。

```python
fun(1, z=3, y=2)
```

* 不能为一个形参重复传值

```python
# ! fun(1, z=3, x=1) 相当于x变量被传了两次，这是不允许的
```

#### 函数参数的其他机制

* 默认参数

    * 在定义函数时就给其赋值——使用形参的常用值
    * 机器学习库中非常常见
    * 调用函数时，可以不对该形参传值，也可以正常传值

    ```python
    def register(name, age, sex='male'):
        print(name, age, sex)
        
    register('Jack', 16) # Jack 16 male 没有对sex参数传值，默认为male
    register('Mary', 20, 'female') # Mary 20 female 也可以对sex传值，进行修改
    ```

    * 注意1：位置参数必须放在默认参数的前面（否则会报错）

    ```python
    # ! def fun(name = 'Bob', age, sex):
    ```

    * 注意2：默认参数应该设置为**不可变类型**（数字、字符串、元组），
        * 否则有可能造成同样的参数不同结果，好像函数具有记忆功能。下面是一个这样的例子：

    ```python
    def func(ls=[]):
        ls.append(1)
        print(ls)
        
    func() # [1]
    func() # [1, 1]
    func() # [1, 1, 1]
    
    # 比较一下：
    def func2(s='Python'):
        s += '3.7'
        print(s)
        
    func2() # 'Python3.7'
    func2() # 'Python3.7'
    ```

    * 上述的问题出现原因是：如果使用可变类型，其一旦建立，其地址就会固定下来，因此每次操作都对原来地址上的数据修改；而如果是不可变类型，每次操作都会修改数据，其地址也会发生变化，因此不会产生问题。

* 可选参数

    * 有的参数可要可不要，通过设置默认值为`None`来实现：

    ```python
    def get_name(first_name, last_name, middle_name=None):
        if middle_name:
            return first_name + middle_name + last_name
        else:
            return first_name + last_name
        
    get_name('Twist', 'Lu') # 'TwistLu'
    get_name('Twist', 'Lu', 'Sirius') # 'TwistSiriusLu'
    ```

* 可变长参数写法1： *args

    * 有时我们不知道要传递过来多少参数，就可以将这些参数用`*args`来代替。
    * 该形参必须放在参数列表的最后
    * 多出来的参数会以元组的形式打包传入函数。不能给其他没有的形参。

    ```python
    def foo(x, y, z, *args):
        print(x, y, z)
        print(args)
        
    foo(1, 2, 3, 4, 5, 6) 
    """
    1 2 3
    (4, 5, 6) # 打包传入
    """
    # ！foo(1, 2, 3, b=4) 不能给没有的形参b
    ```

    * 实参打散
        * 有时我们会把多出来的参数合并成一个列表，我们也希望传入的参数大致也是这样。
        * 此时我们就会在调用时增加一个 `*` 符号来解决。

    ```python
    # 接上例
    foo(1, 2, 3, [4, 5, 6])
    """
    1 2 3
    ([4, 5, 6],) # 这不是我们希望的结果
    """
    foo(1, 2, 3, *[4, 5, 6])
    """
    1 2 3
    (4, 5, 6)
    """
    ```

* 可变长参数写法2：**kwargs
    * 将多余的参数以字典的形式传递给kwargs

    ```python
    def foo2(x, y, z, **kwargs):
        print(x, y, z)
        print(kwargs)
        
    # ! foo2(1, 2, 3, 4, 5, 6) # 必须以字典的形式给出
    foo2(1, 2, 3, a=4, b=5, c=6) 
    """
    1 2 3
    {'a': 4, 'b': 5, 'c': 6}
    """
    ```

    * 同上面的参数，字典的实参也可以打散，但是要加`**`符号：

    ```python
    # 接上例
    # ! foo2(1, 2, 3, {"a":4, "b":5, "c":6}) 没有加**符号
    foo2(1, 2, 3, **{"a":4, "b":5, "c":6})
    """
    1 2 3
    {'a': 4, 'b': 5, 'c': 6}
    """
    ```

* 可变长参数可以组合使用：

    ```python
    def foo3(*args, **kwargs):
        print(args)
        print(kwargs)
        
    foo3(1, 2, 3, a=4, b=5, c=6)
    """
    (1, 2, 3)
    {'a': 4, 'b': 5, 'c': 6}
    """
    ```



### 函数体与变量作用域

* 函数体仅在调用函数时执行相关代码，而定义时不执行任何操作。

#### 局部变量与全局变量

* 仅在函数体内定义和发挥作用的称为**局部变量**。

```python
def multiply(x, y):
    z = x * y
    return z

multiply(2, 9)  # 18
# ! print(z) z是局部变量，函数体之外并不存在！
```

* 外部定义的都是**全局变量**

    * 全局变量可以在函数体内部被使用：

    ```python
    ls = [0]
    def multiply(x, y):
        z = x * y
        ls.append(z)  # 使用了全局的列表ls
        return z
    
    multiply(2, 9) # 18 
    ls # [0, 18]
    ```

    * 通过 **global** 关键字可在函数体内定义全局变量：

    ```python
    def multiply(x, y):
        global z
        z = x * y
        return z
    
    multiply(2, 9) # 18
    print(z) # 18
    ```

#### 返回值

* 单个返回值：通常的返回值仅有一个，前面的例子都是这样。
* 多个返回值：可以在函数体内连续返回多个值。它们是以元组的形式返回的。

```python
def fun(x):
    return x, x**2, x**3 # 以元组打包返回

x1, x2, x3 = fun(2) # 解包赋值
```

* 当有多个return语句时，一旦执行其中一个，该函数就返回。
* 当没有 return 语句时，返回值为None



### 匿名函数

#### 基本形式与用法

* 基本形式：lambda 变量: 函数体

* 常见用法：在参数列表中使用匿名函数（比如与key = 搭配）
    * 在 list.sort 和 sorted 函数中的使用：

```python
ls = [(93, 88), (79, 100), (86, 71), (85, 85), (76, 94)]
ls.sort()
ls # [(76, 94), (79, 100), (85, 85), (86, 71), (93, 88)] 默认ls按照元组的第一个元素排序

ls.sort(key = lambda x: x[1]) # 按x[1]大小进行排序
ls # [(86, 71), (85, 85), (93, 88), (76, 94), (79, 100)] 

# 同理，可以对总和进行排序
ls.sort(key = lambda x: x[0]+x[1])
ls # [(86, 71), (85, 85), (76, 94), (79, 100), (93, 88)]

# 复习一下，可以对列表进行降序排序
ls.sort(key = lambda x: x[0]+x[1], reverse = True)
ls # [(93, 88), (79, 100), (85, 85), (76, 94), (86, 71)]
```

* 在max()、min()函数中也可以这么用：

```python
ls = [(93, 88), (79, 100), (86, 71), (85, 85), (76, 94)]
n = max(ls, key = lambda x: x[1])
n  # (79, 100)

n = min(ls, key = lambda x: x[1])
n  # (86, 71)
```



### 几点建议

* 函数和参数的命名要有实际意义
* 应包含简要阐述函数功能的注释

```python
def fun():
    # 这里应该解释一下这个函数的作用
    pass  # 空语句，用于保持语法的完整性
```

* 函数定义前后应该空两行
* 默认参数赋值等号两侧不加空格

```python
def f1():
    pass

				# 空行以示清白
def f2():
    pass


def f3(x=3):  # 默认参数赋值等号两侧不加空格
    pass
```



### 附：单元测试抛出异常

* 运用 assert 产生异常：
    * 当assert 后面的表达式为真，则什么也不做；
    * 当assert 后面的表达式为假，则触发异常。

```python
def division(x, y):
    assert y != 0, '分母不应为0'
    return x / y

division(5, 3) # 1.6667
division(4, 0) # AssertionError: 分母不应为0
```



* 至此，基本的python编程方法已经介绍完毕。后面将进入python的高级编程方法。



* Written by：Sirius. Lu
* Reference：深度之眼  python基础训练营
* 2020.6.3