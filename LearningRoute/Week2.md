# 第2周学习报告

## 学习内容

- shell的变量（接上周）
- shell script
- python

## 学习收获

### 接上周的shell变量

shell script中可以用`read`命令进行变量值的输入。

```shell
read -p "pleas input the value:" -t 30 var
```

上述命令在屏幕显示提示信息，等待输入30秒。

`declare [-axirp] var`用于声明变量属性（默认字符串类型）：

- -a array
- -i integer
- -x 相当于export
- -r readonly（不能unset）
- -p 显示变量属性

数值计算默认为整型，将"-"改为"+"可以取消操作。

更改文件系统和程序的限制关系：`ulimit`（user limit）限制打开文件数量，CPU时间，运存等。

`alias`和`unalias`定义和取消命令别名。

### shell script

通配符及其含义

|符号|含义|
|---|---|
|*|0个及以上任意字符|
|?|至少一个任意字符|
|[]|至少一个括号内的字符|
|[-]|-表示编码顺序的所有字符，如0-9|
|[^]|反向选择，至少一个不是括号里的字符|

特殊符号

|符号|含义|
|---|---|
|#|shell script行注释|
|\\|跳脱字符，将通配符或特殊符号转化为一般字符|
|\||分隔管道命令|
|;|分隔连续命令|
|>,>>|输出导向，分别是替换和叠加|
|<,<<|输入导向|

#### 数据流重定向

- 标准输入（stdin）：代码为0，使用<或<<
- 标准输出（stdout）：代码为1，使用>或>>
- 标准错误输出（stderr）：代码为2，使用2>或2>>

不需要的数据重定向到`/dev/null`（黑洞）。

将输出定向到同一文件：`2>&1`

`<<`表示结束输入的字符，如`<<"eof"`

`;`用于连接无关的连续命令

相关的连续命令（可参考C/C++中的逻辑表达式截断）：

- cmd1 && cmd2：cmd1正确运行结束后才运行cmd2
- cmd1 || cmd2：cmd1运行错误才运行cmd2

#### 管线(pipe)命令

用|连接，前一个的stdout作为后一个的stdin（仅读取stdout），后一个命令必须能接受stdin。

信息筛选：cut，grep（按行处理）

```shell
cut -d'分隔字符' -f fields #依据分隔字符分成几段，用-f取出第几段
cut -c 字符区间 #以字符的单位取出固定字符区间
```

grep：选取含有关键词的行并输出

`grep [-acinv] [--color=auto] '搜索字符串' filename`

- -a 将bin以txt的方式搜索
- -c 计算匹配次数
- -i 忽略大小写
- -n 输出行号
- -v 反向选择，输出不匹配的行
- --color=auto 关键词高亮

双向重定向 ：用tee同时保留文件和stdout

分割命令split

```shell
split -b size(with unit) file PREFIX
split -l line
```

用输出重定向合并文件

```shell
cat file* >> filename
```

减号可以在管线命令中代替文件使用，避免了另外生成文件。

#### 正则表达式

考虑到正则表达式的种类和类型比较多，现在学了如果不是总是用的话很容易忘，所以现在只是粗略地看一下，等之后需要用的时候再根据情况具体地看。

#### 文件对比

diff（以行为单位）

- -b 忽略空白的差异（blank）
- -B 忽略空白行的差异
- -i 忽略大小写

```shell
diff [-bBi] from-file to-file
```

diff也可用于比较目录的内容。

`cmp [-s] file1 file2`按位比较文件，用-s输出所有不同，默认只输出第一次不同。

shell script中的判断式：

test

- -e exist
- -f 是否是file
- -d 是否是directory
- -r readonly
- -w writeable
- -x 是否能运行（execute）

两文件的判断

`test file1 [-options] file2`

- -nt newer than
- -ot older than
- -ef 是否是同一文件

整数判断

`test n1 [-op] n2`

- -eq equal
- -ne not equal
- -gt greater than
- -lt less than
- -ge >=
- -le <=

字符串判断

- -z 是否为空
- -n 是否非空（-n可省略）
- str1 = str2
- str1 != str2

多重条件判断

- -a and
- -o or
- ! 反相

判断符号[]：

中括号中的每个成分必须用空白隔开。

可以用`chmod +x *.sh`来为.sh文件添加可执行权限。

shell script也可有带参数的形式，参数分别是$1, $2, $3...，可以用shift进行参数的前移，即去掉前面的参数。

条件判断的使用

```shell
if [ 逻辑表达式 ]; then
    语句1
    语句2
    ...
fi #通过倒着写if表示这个if语句的结束

if [ 逻辑表达式 ]; then
    语句1
    语句2
    ...
elif [ 逻辑表达式 ]; then
    语句1
    语句2
    ...
else
    语句1
    语句2
    ...
fi
```

条件判断分支的使用

```shell
case $变量名 in
    "第一个变量内容")
        程序段
        ;; #用两个连续的分号表示程序段的结束
    "第二个变量内容")
        程序段
        ;;
    ...
    *) #最后一个变量用*来表示所有其他值
        exit 1
        ;;
esac #同样是倒着写case表示case的结束
```

函数

```shell
function FunName() #放在shell script的最前面
{
    #function里面也有与shell script独立的内建变量$1, $2, $3...即可以带参数
    程序段
}

FunName arg1 arg2
```

循环语句

```shell
while [ condition ]
do
    程序段
done

until [ condition ]
do
    程序段
done
```

```shell
for var in con1 con2 con3... #每次循环时i的状态
do
    程序段
done

for ((i=1; i<=$n; i++))
do
    程序段
done

for var in $(seq 1 $n)
do
    程序段
done
```

sh程序本身也可以检查shell script：

`sh [-nvx] *.sh`

- -n 不运行，仅检查语法错误
- -v 运行前显示sh文件内容
- -x 显示sh的语句执行过程

### python

python跟C/C++有很多重合的地方，所以只学习了python的语法和特有的性质。

list

```python
l = [1, 2, 'haha']
print l[0]
print l[1]
print l[2]
print l[-1]
print l[-2]
print l[-3]
l.append('ok')
l.insert(1, 'aa')
print l
```

tuple（类似C/C++的常量数组）

```python
t = (1, 2, 'aa')
t = (1,)
```

if语句

```python
if <con1>:
    <op1>
elif <con2>:
    <op2>
elif <con3>:
    <op3>
else:
    <op4>
```

循环语句

```python
for var in range(101): #0, 1, 2, ..., 100
    代码段

while <逻辑表达式>:
    代码段

```

dict（类似C/C++的map，映射）

```python
d = {1: 233, 'aa': 'bb'}
d[1]
d['aa']
d['cc'] = 2
d.get(key) #会判断d中是否存在key，不存在默认返回none（即不返回），第二参数是缺省的返回值
key in d #返回bool类型，类似shell script的判断
d.pop(key)

#key必须是不可变对象
```

set

```python
s = set([1, 2, 3]) #initialize a set with list
print s #set is not list
s.add(key)
s.remove(key)
s1 & s2
s1 | s2
```

可以把函数名赋值给一个变量，相当于用了函数的引用。

函数定义

```python
def MyFun1():
    pass #pass可当作占位符，使得这部分还没写的时候程序也能运行

def MyFun2(x):
    if not isinstance(x, (int, float)): #类型判断
        raise TypeError('wrong type')
    if x >= 0:
        return (1, x) #可以用tuple返回多个值，函数外获取返回值的方式类似matlab
    else:
        return (-1, x)
```

函数的声明可以带缺省值（默认参数），而默认参数必须指向不变的对象。

可变参数

```python
def MyFun(*num)
    for i in num:
        代码段

MyFun(1, 2, 3)
MyFun(*[1, 2, 3])
#函数外可以用*号加list或tuple名将list或tuple转化为可变参数，函数内部可变参数会转化为tuple
```

关键字参数

```python
def MyFun(**kw): #跟可变参数类似，关键字参数传进来的是dict，用双*号将dict或set转化为关键字参数
    print kw

MyFun(key1='aa', key2='bb')
KeyWord = {'key1': 'aa', 'key2': 'bb'}
MyFun(**KeyWord)
```

参数组合

参数定义顺序：

1. 必选参数
2. 默认参数
3. 可变参数
4. 关键字参数

切片操作的语法类似matlab。

迭代（仅对于iterable的对象）

```python
for var in p:
    代码段

#判断是否为可迭代对象

from collections import Iterable
isinstance(obj, Iterable)

#可以使用enumerate函数将list转化为索引-元素对，使得在循环中可以同时迭代索引和元素本身
```

列表生成可以把元素放在第一位，for循环跟在后面，可以加if条件判断。

生成器(generator)

```python
#相当于在使用列表生成器的时候将中括号换成小括号
g = (x*x for x in range(1, 11))
for n in g:
    print n

#用函数的方法构建generator

def gen(n)
    while n < 10:
        yield n #每次遇到yield返回，下次调用则从这个yield开始运行
```

## 疑问和困难

1. 现在应该学习python2.7还是较新的python3.x
2. 目前的前沿研究中所用到的模块主要支持的是哪个版本的python
