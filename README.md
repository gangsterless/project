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




之后运行myutils 里面的cutfile()


!["切割文件函数"](https://github.com/gangsterless/project/blob/master/Sketch%20Map/sketch3.png)



之后应该会多出好多文件保留第0个就成。

!["第0个"](https://github.com/gangsterless/project/blob/master/Sketch%20Map/sketch2.png)



下一步运行myutils 里面的split_train_and_test(),产生好多文件



!["运行示意"](https://github.com/gangsterless/project/blob/master/Sketch%20Map/sketch4.png)



!["产生训练测试验证等等"](https://github.com/gangsterless/project/blob/master/Sketch%20Map/sketch5.png)



下一步运行RawAnalysis就会产生融合后的特征
!["融合特征"](https://github.com/gangsterless/project/blob/master/Sketch%20Map/sketch6.png)



主函数入口在RawAnalysis
如果调用  F_extractor.Feature_merge()会产生融合模型的csv
最后如果调用 F_extractor.Relevance_Analysis()会产生相关性图片



目前已产生特征向量：
目前特征介绍
user_id：用户唯一标识

item_id：商品唯一标识

item_category：商品所属类别，出现歧义时按照用户表处理

isbuy:最后是否买了某商品，对于训练集就是12,16的结果，如果买了就对应买的商品id否则，填充0

0：对应用户浏览商品量

1：对应用户收藏商品量

2：对应用户加入购物车商品量

3：对应用户购买量

each_user_visit_count_1212：对应用户双十二浏览商品量

each_user_cart_count_1212：对应用户双十二加入购物车商品量

each_user_colletion_count_1212：对应用户双十二收藏商品量

each_user_buy_count_1212：对应用户双十二购买量

each_user_visit_count_latest3：对应用户最近三天浏览次数

each_user_colletion_count_latest3：对应用户最近三天收藏次数

each_user_cart_count_latest3：对应用户最近三天加入购物车次数

each_user_buy_count_latest3：对应用户最近三天购买次数

max_buy_sum_diff：对应用户最近购买最大量距预测日时长单位是小时

add_buy_div_buy_times：对应用户加购量占总购买量比值

buy_div_visit_times：对应用户购买量与访问量比值

buy_div_colletion_times：对应用户购买量与收藏量比值

buy_div_cart_times：对应用户购买量与加入购物车比值

double12_visit_div_visit：对应用户双12访问量与访问量比值

double12_buy_div_buy：对应用户双12购买量与购买量比值

Add_buy：用户加购量

buy_item_count：用户购买商品的种类数

max_buy_item_id：这是个元组，两个元素对应，最大购买量对应商品的id,该商品的购买量，可以为空（没有买）

active_grade：活跃度，怎么定义这个函数还要商讨

last_buy_diff：上次购买距预测日时长

以下是商品的特征

buy_times：某商品被用户购买次数（由于用户只选取了一小部分，故该信息十分不准确）

visit_times：用户访问次数：某商品被用户访问次数（由于用户只选取了一小部分，故该信息十分不准确）

collection_times：某商品被用户收藏次数（由于用户只选取了一小部分，故该信息十分不准确）

cart_times：某商品被加入购物车次数（由于用户只选取了一小部分，故该信息十分不准确）

item_buy_div_visit：该商品被购买量与被访问量之比

item_colletion_div_visit：该商品被收藏量与被访问量之比

catorgory_visit_times：该类被浏览次数

catorgory_buy_times：该类被购买次数

catorgory_cart_times：该类被加入购物车次数

catorgory_collection_times：该类被收藏次数



