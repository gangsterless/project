# project
this is a project about recommend system
整体思路是参考
https://blog.csdn.net/u014374284/article/details/49933487
的提取方法
某些地方写的有点儿啰嗦，一方面是想写的清楚一些，没有用很多的pandas的自定义函数，另一方面对此库不太熟悉。
1 项目结构
①data/raw:

原始数据 tianchi_fresh_comp_train_item.csv和tianchi_fresh_comp_train_user.csv 由于tianchi_fresh_comp_train_user.csv 太大，难以用常规方法一次性读取，就分批读取n行，分开保存，分割文件的函数见Utils/myutils/CutFile 目前tianchi_fresh_comp_train_user0.csv是提取的前100万行，后续可以考虑用Pickle二进制读取，或者用内存更大的服务器。
②data/dealeddata:

这是经过处理的数据，由于我们只提取了user的前100万行所以很多信息不完整。具体每个文件的介绍见后面的函数模块。
③DataAnalysis/PLOT:

画图模块，目前只有画条形图，后面考虑各个变量之间的相关性，会增加更多的函数
④Utils/PrintInfo:

就是简单的输出星号，使得各个输出部分更清晰
⑤Utils/myutils:

各种工具函数，主要是split_train_and_test用来分割测试集，训练集，验证集，分割的结果存在前面的data/dealeddata中，SortByTime是按时间排序返回类型是list
⑥RawAnalysis

原始数据处理模块，DataDes描述原始信息，LoadData，用pandas加载数据，最主要的类 Feature_Extractor 用来提取各种特征。


另外，每次都提交一份最新的merge后的特征向量列表,为了上传方便，不是太大。


运行指南：

开始的时候在data/raw/ 里面应该有这两个文件
!["初始文件示意图"](https://github.com/gangsterless/project/blob/master/Sketch%20Map/sketch1.png)
