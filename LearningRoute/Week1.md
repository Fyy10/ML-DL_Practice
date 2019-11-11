# 第1周学习报告

## 学习内容

- Linux操作系统
- vim相关操作
- shell中的变量（一部分）

## 学习收获

由于之前就是在Linux系统里用vim进行编程的，所以对于Linux系统和vim也算是有所了解，也有现成的编程环境。不过之前并没有过多地使用vim的快捷键，所以这周对于Linux系统的介绍部分进行了选择性跳过，主要学习了vim的各种常用快捷键，以及shell中的变量。

下面列出了以前没怎么用过但是以后可能会常用的快捷键（备忘）。

|常用快捷键|功能|
|-----|----|
|ctrl+f|下一页（forward）|
|ctrl+b|上一页（backward）|
|ctrl+d|向下半页（down）|
|ctrl+u|向上半页（up）|
|0|数字0，移动到本行最前面|
|$|移动到本行最后面|
|G|移动到最后一行|
|nG|n为数字，移动到第n行|
|gg|移动到第一行|
|nEnter|n为数字，向下n行|
|/word|向下寻找word|
|?word|向前寻找word|
|n|继续查找（next）|
|N|与n相反，反向查找|
|:n1, n2 s/word1/word2/g|将n1行到n2行间的word1替换为word2|
|:n1, $ s/word1/word2/gc|从n1行到末尾替换，每次confirm|
|x, X|删除，x向后（delete），X向前（backspace）|
|dd|删除光标所在那一行|
|ndd|删除下面n行|
|yy|复制当前行|
|nyy|复制下面n行|
|p|粘贴（paste）|
|u|撤销（undo）|
|ctrl+r|重做（redo）|
|.|小数点，重复上次操作|
|:! 命令|暂时退出到命令行执行命令|

vim区块（visual block）选择

|快捷键|功能|
|-----|----|
|v|字符选择，选择光标经过的字符|
|V|行选择，选择光标经过的行|
|ctrl+v|区块选择（矩形）|
|y|复制选择区|
|d|删除选择区|

vim多文件编辑

|命令|功能|
|----|----|
|:n|编辑下一个文件|
|:N|编辑上一个文件|
|:files|列出所有正在编辑的文件|

vim多窗口编辑

|命令|功能|
|----|----|
|:sp 文件名|开启（separate）新窗口，文件名可缺省|
|ctrl+w+上下键|切换窗口（window）|
|ctrl+w+q|关闭当前窗口（quit）|

### 关于bash中的变量

在变量名的前面加上$符号来获取变量的内容，变量的设置可以用“变量名=变量值”的形式，但是等号两边不能有多余的空格。设置变量的内容也可以引用已有的变量，或者在已有的变量后加入新内容，如：

```shell
var=haha
echo $var   #haha
var="$var"haha
echo $var   #hahahaha
```

关于变量设置的命令

|命令|功能|
|---|---|
|echo|用来显示变量的值（其实跟变量没有什么关系）|
|export|将变量设置为环境变量（使得子程序也可以使用）|
|env|environment，显示所有环境变量|
|set|显示所有变量|
|unset|删除某变量|

一些常见（常用）变量及其含义

|变量名|含义|
|-----|----|
|$|$本身也是变量，表示当前这个bash的PID|
|?|前一个程序运行的返回值|
|HOME|家目录|
|SHELL|使用的shell，一般是/bin/bash|
|RANDOM|生成0~32767之间的随机数|

## 疑问和困难

1. 新学的vim快捷键比较多而且杂，容易忘，可能需要一段时间的使用和练习才能比较熟练地使用。
2. 想知道在shell script中，环境变量和自定义变量分别有哪些用法
3. shell script主要用来做什么？（编译，测试，对拍，批处理还是其它）