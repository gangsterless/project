import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from Utils.PrintInfo import curline
from  Utils.ConstValues import *
import seaborn as sns
from sklearn import linear_model
from sklearn.ensemble import  RandomForestRegressor

absdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))#获取当前父文件夹路径
print(absdir)
userdir = absdir+'/'+'data/'
itemdir = absdir+'/'+'data/'
def DataDes(user,item):
    from DataAnalysis.PLOT import PlotBar
    curdir = os.path.abspath(os.path.dirname(__file__))
    #首先整体描述数据
    if not os.path.exists(curdir + '/TotalDescribe.txt'):
        f = open(curdir + '/TotalDescribe.txt', 'w')
        user_des = str(user_csv.describe())
        item_des = str(item_csv.describe())
        res = str(user_des)
        res = "userdes:\n" + user_des + "\n" + "itemdes:\n" + item_des + "\n" + "user_csv.head:\n" + str(
            user_csv.head()) + \
              "\n" + "item_csv.head:\n" + str(item_csv.head())
        f.write(res)
        f.close()
        print(res)
    else:
        print('描述文件已存在')
        with open(curdir + '/TotalDescribe.txt', 'r')as f:
            res = f.read()
            print(res)
    #分析用户id有多少重复
    #多少个用户
    usercount = len(user['user_id'].unique())
    print('有%d个用户信息'%(usercount))
    #这些用户买了多少商品
    itemcount = len(user['item_id'].unique())
    print('有%d个商品信息' % (itemcount))
    #之后分析用户四种行为各自占比 浏览、收藏、加购物车、购买，对应取值分别是1、2、3、4
    behaviorDistribution = user['behavior_type'].value_counts()
    #不难发现多部分都是白嫖，真正买的微乎其微
    list_user_behavior_type = behaviorDistribution.tolist()
    print(list_user_behavior_type)
    Y = list_user_behavior_type
    X = ["浏览","收藏","加购物车","购买"]
    # X = [i for i in range(1,5)]
    PlotBar("用户行为分布情况",X,Y)
    #时间分布
    year  = []
    for i in range(len(user['time'])):
        # print(user['time'][i])
        eachtime = str(user['time'][i])
        eachyear = eachtime.split()[0]
        year.append(eachyear)
    dfyear = DataFrame(year)
    user['year'] = dfyear
    year_count = user['year'].value_counts()
    list_year_and_day = year_count.tolist()
    print(type(year_count))
    X = year_count.index
    # print(X)
    Y = list_year_and_day
    # PlotBar("购买日期情况",X,Y)
def LoadData():
    #当有行号时，必须加上最后一个参数
    user_train_csv = pd.read_csv(userdir+r'dealeddata/train_feature.csv', index_col=0)
    #item先不处理吧
    item_train_csv = pd.read_csv(itemdir+r'raw/tianchi_fresh_comp_train_item.csv')
    #我在想是不是应该在这里就merge一下？？？或许应该是这样
    #对滴
    #这里应该怎么merge是个问题 train_feature里面的item_id 跟 tianchi_fresh_comp_train_item好多是不对应的，即前者
    #有的后者没有，后者有的前者也不一定有，我们还是要主要提取用户的信息所以，以前者为基准
    train_label =  pd.read_csv(itemdir+r'dealeddata/userdf_train_label.csv', index_col=0)
    mergeres = user_train_csv
    #这个数据好迷，同一个item_id在用户信息中和商品信息中对应的类别竟然是不同的
    # 所以删除item_train_csv中的类别信息，否则merge的时候会出现冲突
    deledcategory_iteminfo =item_train_csv.drop('item_category',axis=1,inplace=False)
    mergeres = pd.merge(mergeres, deledcategory_iteminfo, on='item_id',how='left')
    mergeres = pd.merge(mergeres, train_label, on='user_id')
    #两个都有catorgory
    mergeres = mergeres[['user_id', 'item_id','item_category','isbuy']]
    # mergeres.to_csv('mergeres.csv',index=False)
    # print(item_train_csv['item_id'])

    return user_train_csv,item_train_csv,mergeres
class Feature_Extractor:

    def __init__(self,user,item,mergeres):
        self.user = user
        self.item = item
        self.mergeres = mergeres
        self.user_id_set = user['user_id'].unique()
    def extract_geo_info(self):

        '''
        分析地理信息，可能用不上，先写一下
        '''
        curline(info='开始分析用户地理位置', isstart=True)
        geo_null_count = self.user['user_geohash'].isnull().sum()
        print('缺失值有%d个' % (geo_null_count))
        drropped_geo = self.user['user_geohash'].dropna().tolist()
        drropped_geo_set = set(drropped_geo)
        print('一共有%d个地方' % (len(drropped_geo_set)))

    def extract_timeinfo(self):

        from Utils.myutils import SortByTime
        import time
        import datetime
        curline(info='开始分析日期情况个数', isstart=True)
        # 按时间排序的
        orderedtimedict = {}
        # 将用户的浏览时间按日期排序
        i = 1
        for each in self.user_id_set:
            print(i)
            try:
                each_user_visit = self.user.loc[(self.user['user_id'] == each)]
                each_user_visit_list = each_user_visit['time'].tolist()
                sorteddres = SortByTime(each_user_visit_list)
                orderedtimedict[each] = sorteddres
            except:
                print('length' + len(each_user_visit_list))
                print(each)
            i += 1
        #由于时间序列的个数不相同，所以值为一个列表
        k = list(orderedtimedict.keys())
        v = list(orderedtimedict.values())
        user_orderedtime_df = pd.DataFrame(list(zip(k, v)), columns=['k', 'v'])
        print(user_orderedtime_df.head())
        user_orderedtime_df_dir = absdir + '/' + 'data/dealeddata/'
        if not os.path.exists(user_orderedtime_df_dir+'user_orderedtime.csv'):
            user_orderedtime_df.to_csv(user_orderedtime_df_dir+'user_orderedtime.csv')
    def extract_behavior_info(self):
        from Utils.PrintInfo import curline
        from collections import Counter
        from Utils.myutils import SortByTime

        import time
        import datetime
        curline(info= '开始分析每个用户每种行为的个数',isstart=True)
        totaldict  ={}
        #加购量
        Add_purchase_list = []
        #购买商品种类数
        purchase_item_count_list = []
        #最多购买物品id
        max_buy_id_list = []
        #活跃天数
        active_days_list = []
        #x,y,z,q分别代表 1,2,3,4四种行为
        # 怎么定义这个活跃度还需要研究一下，在这里改计算公式
        F_active_grade = lambda x,y,z,q: x+2*y+3*z+4*q
        #上次购买距预测日时长
        last_buy_diff_list = []
        ##最大购买量距预测日时长
        max_buy_sum_diff_list = []
        #用户发生二次购买商品数占总购买商品数的比值
        Add_buy_div_buy_times_list = []
        #双十二期间的四个量
        each_user_visit_count_1212_list = []
        each_user_colletion_count_1212_list = []
        each_user_cart_count_1212_list = []
        each_user_buy_count_1212_list = []
        # 近三天期间的四个量
        each_user_visit_count_latest3_list = []
        each_user_colletion_count_latest3_list = []
        each_user_cart_count_latest3_list = []
        each_user_buy_count_latest3_list = []

        for index,value in enumerate(self.user_id_set):
            each_user_info = self.user.loc[(self.user['user_id'] == value)]
            # 浏览
            each_user_read_count = each_user_info.loc[each_user_info['behavior_type'] == 1].shape[0]
            each_user_colection_count = each_user_info.loc[each_user_info['behavior_type'] == 2].shape[0]
            # 加购物车
            each_user_cart_count = each_user_info.loc[each_user_info['behavior_type'] == 3].shape[0]
            # 购买
            each_user_buy = each_user_info.loc[each_user_info['behavior_type'] == 4]
            each_user_buy_count = each_user_buy.shape[0]

            if each_user_read_count > MAX_HUMAN_VISIT_TIMES:
                # print(str(each)+'  '+str(each_user_read_count))
                # print('这个不要了')
                continue
                # 收藏

            this_user_info = self.user.loc[self.user['user_id'] == value]
            double12_user_info = this_user_info.loc[this_user_info['time'].apply(lambda x: x[0:10])=='2014-12-12']
            if  len(double12_user_info)==0:
                each_user_visit_count_1212_list.append(0)
                each_user_colletion_count_1212_list.append(0)
                each_user_cart_count_1212_list.append(0)
                each_user_buy_count_1212_list.append(0)
            else:
                each_user_visit_count_1212_list.append(double12_user_info.loc[double12_user_info['behavior_type']==1].shape[0])
                each_user_colletion_count_1212_list.append( double12_user_info.loc[double12_user_info['behavior_type'] == 2].shape[0])
                each_user_cart_count_1212_list.append(double12_user_info.loc[double12_user_info['behavior_type'] == 3].shape[0])
                each_user_buy_count_1212_list.append(double12_user_info.loc[double12_user_info['behavior_type'] == 4].shape[0])

            #近三天信息提取
            latest3_user_info = this_user_info.loc[this_user_info['time'].apply(lambda x: STARTTIME <=time.strptime(x,"%Y-%m-%d %H")<=ENDTIME)]

            if len(latest3_user_info)==0:
                each_user_visit_count_latest3_list.append(0)
                each_user_colletion_count_latest3_list.append(0)
                each_user_cart_count_latest3_list.append(0)
                each_user_buy_count_latest3_list.append(0)
            else:
                each_user_visit_count_latest3_list.append(latest3_user_info.loc[latest3_user_info['behavior_type']==1].shape[0])
                each_user_colletion_count_latest3_list.append(latest3_user_info.loc[latest3_user_info['behavior_type']==2].shape[0])
                each_user_cart_count_latest3_list.append(latest3_user_info.loc[latest3_user_info['behavior_type']==3].shape[0])
                each_user_buy_count_latest3_list.append(latest3_user_info.loc[latest3_user_info['behavior_type']==4].shape[0])


            activegrade = F_active_grade(each_user_read_count, each_user_colection_count, each_user_cart_count,
                                         each_user_buy_count)
            active_days_list.append(activegrade)
            tmplist = []

            #浏览太多可能会有问题,估计是爬虫

            #加购量
            if each_user_buy_count>0:
                #挑选出最近一次购买时间,精确到小时
                each_user_buy_list = each_user_buy['time'].tolist()
                sortedlast = SortByTime(each_user_buy_list)[-1]
                last_buy_time = time.strptime(sortedlast, "%Y-%m-%d %H")
                time1 = datetime.datetime(last_buy_time[0], last_buy_time[1], last_buy_time[2],last_buy_time[3])
                time2 = datetime.datetime(PREDICTDAY[0], PREDICTDAY[1], PREDICTDAY[2], PREDICTDAY[3])
                #最后一次买的时间距预测时间的时长，单位是小时
                last_buy_diff_list.append((time2-time1).days*24+(time2-time1).seconds//3600)
                #买了几种商品

                c1 = each_user_buy['item_id'].count()
                c2  =  len(each_user_buy['item_id'].unique())
                #获取买的最多商品的id
                each_user_buy_list =  each_user_buy['item_id'].tolist()
                #二次购买的量占总购买量的比例
                add_buy_count_list =  [i-1 for i in Counter(each_user_buy_list).values() if i>1]
                add_buy_times = sum(add_buy_count_list)
                Add_buy_div_buy_times_list.append(add_buy_times/sum( Counter(each_user_buy_list).values()))
                max_buy_id =  Counter(each_user_buy_list).most_common(1)[0]
                #获取最大购买量发生的时间，如果有多个选取离预测日期最近的一个
                max_buy_sum_time = list(each_user_buy.loc[each_user_buy['item_id']==max_buy_id[0]]['time'])
                max_buy_sum_time = SortByTime(max_buy_sum_time)[-1]
                max_buy_sum_time = time.strptime(str(max_buy_sum_time), "%Y-%m-%d %H")
                time3 = datetime.datetime(max_buy_sum_time[0], max_buy_sum_time[1], max_buy_sum_time[2], max_buy_sum_time[3])
                max_buy_sum_diff_list.append((time2 - time3).days * 24 + (time2 - time1).seconds // 3600)
                Add_purchase_list.append(c1-c2)
                purchase_item_count_list.append(c2)
                max_buy_id_list.append(max_buy_id)


            else:
                Add_buy_div_buy_times_list.append(0)
                Add_purchase_list.append(0)
                max_buy_sum_diff_list.append(np.nan)
                purchase_item_count_list.append(0)
                max_buy_id_list.append('')
                #如果没有买过就填充非数字
                last_buy_diff_list.append(np.nan)
            tmplist.extend([each_user_read_count,each_user_colection_count,each_user_cart_count,each_user_buy_count])
            totaldict[value] = tmplist
        user_behavior_count_df = (pd.DataFrame(totaldict)).T
        user_behavior_count_df['user_id'] = totaldict.keys()
        user_id_col = user_behavior_count_df['user_id']
        user_behavior_count_df.drop(labels=['user_id'], axis=1, inplace=True)
        user_behavior_count_df.insert(0, 'user_id', user_id_col)
        # 双十二用户浏览次数
        user_behavior_count_df['each_user_visit_count_1212'] = each_user_visit_count_1212_list
        user_behavior_count_df['each_user_colletion_count_1212'] = each_user_colletion_count_1212_list
        user_behavior_count_df['each_user_cart_count_1212'] = each_user_cart_count_1212_list
        user_behavior_count_df['each_user_buy_count_1212'] = each_user_buy_count_1212_list
        #近三天的相关信息
        user_behavior_count_df['each_user_visit_count_latest3'] = each_user_visit_count_latest3_list
        user_behavior_count_df['each_user_colletion_count_latest3'] = each_user_colletion_count_latest3_list
        user_behavior_count_df['each_user_cart_count_latest3'] = each_user_cart_count_latest3_list
        user_behavior_count_df['each_user_buy_count_latest3'] = each_user_buy_count_latest3_list
        # 购买量与浏览量比值
        buy_div_visit_times  = list(map(lambda a, b:0 if b==0 else a / b,user_behavior_count_df[3],user_behavior_count_df[0]))
        # 购买量与收藏量比值
        buy_div_colletion_times = list(map(lambda a, b: 0 if b == 0 else a / b, user_behavior_count_df[3], user_behavior_count_df[1]))
        # 购买量与购物车量比值
        buy_div_cart_times = list(map(lambda a, b: 0 if b == 0 else a / b, user_behavior_count_df[3], user_behavior_count_df[2]))
        #双十二浏览量与总浏览量比值
        double12_visit_div_visit = list(map(lambda a, b: 0 if b == 0 else a / b,user_behavior_count_df['each_user_visit_count_1212'],
                                            user_behavior_count_df[0]))
        # 双十二购买量与总购买量比值
        double12_buy_div_buy = list(map(lambda a, b: 0 if b == 0 else a / b, user_behavior_count_df['each_user_buy_count_1212']
                                        , user_behavior_count_df[3]))
        #最大购买量距预测日时长
        user_behavior_count_df['max_buy_sum_diff'] = max_buy_sum_diff_list
        # 用户发生二次购买商品数占总购买商品数的比值
        user_behavior_count_df['add_buy_div_buy_times'] = Add_buy_div_buy_times_list

        # print(buy_div_visit_times)
        user_behavior_count_df['buy_div_visit_times'] = buy_div_visit_times
        user_behavior_count_df['buy_div_colletion_times'] = buy_div_colletion_times
        user_behavior_count_df['buy_div_cart_times'] = buy_div_cart_times
        user_behavior_count_df['double12_visit_div_visit'] = double12_visit_div_visit
        user_behavior_count_df['double12_buy_div_buy'] = double12_buy_div_buy
        user_behavior_count_df['Add_buy'] = Add_purchase_list
        user_behavior_count_df['buy_item_count'] = purchase_item_count_list

        #先尝试把
        user_behavior_count_df['max_buy_item_id'] = max_buy_id_list
        user_behavior_count_df['active_grade'] = active_days_list
        user_behavior_count_df['last_buy_diff'] = last_buy_diff_list
        self.user_behavior_count = user_behavior_count_df
        curline(info='',isstart=False)
    def extract_item_info(self):
        from Utils.PrintInfo import curline
        from collections import Counter
        from Utils.myutils import SortByTime
        import time
        import datetime
        curline(isstart=True,info='开始提取商品信息')
        item = self.item
        #商品信息的dataframe
        item_df_info = pd.DataFrame()

        #有些商品的id出现了好多好多次，但他们的地理信息不同，说明在不同地方均有出售
        # dupclicatetimes = item.groupby('item_id').apply(lambda d: len(d.index) if len(d.index) > 1 else None).dropna()
        # print(dupclicatetimes)
        # print(item_df_info['item_id'].duplicated())


        item_df_info['item_id'] = item['item_id'].unique()
        #被购买次数
        buy_times_list = [0]*item_df_info['item_id'].shape[0]
        #被浏览次数
        visit_times_list = [0]*item_df_info['item_id'].shape[0]
        #被收藏次数
        collection_times_list = [0]*item_df_info['item_id'].shape[0]
        #加入购物车
        cart_times_list = [0]*item_df_info['item_id'].shape[0]
        item_df_info['buy_times'] = buy_times_list
        item_df_info['visit_times'] = visit_times_list
        item_df_info['collection_times'] = collection_times_list
        item_df_info['cart_times'] = cart_times_list
        # print(item_df_info)
        buy_times_dict =  Counter(self.user.loc[self.user['behavior_type'] == 4]['item_id'])
        visit_times_dict = Counter(self.user.loc[self.user['behavior_type'] == 1]['item_id'])
        colletion_times_dict = Counter(self.user.loc[self.user['behavior_type'] == 2]['item_id'])
        cart_times_dict = Counter(self.user.loc[self.user['behavior_type'] == 3]['item_id'])
        for rindex in item_df_info['item_id'].index:
            key = item_df_info['item_id'][rindex]
            if key in buy_times_dict.keys():
                item_df_info['buy_times'][rindex] = buy_times_dict[key]
                # print('还是找得到的'+str(key))
            if key in visit_times_dict.keys():
                item_df_info['visit_times'][rindex] = visit_times_dict[key]
            if key in colletion_times_dict.keys():
                item_df_info['collection_times'][rindex] = colletion_times_dict[key]
            if key in cart_times_dict.keys():
                item_df_info['cart_times'][rindex] = cart_times_dict[key]

        '''
        这样看来大多数商品压根没人买过，其实原因是因为我们的用户数据量截取的太少
        总共也没几个用户当然买的少
        '''
        # print(item_df_info.where(item_df_info['buy_times']>0).dropna())
        # print(item_df_info.where(item_df_info['visit_times']>0).dropna())
        # print(item_df_info.where(item_df_info['collection_times'] > 0).dropna())
        # print(item_df_info.where(item_df_info['cart_times'] > 0).dropna())
        #
        # print(item_df_info)
        self.item_info = item_df_info
        # item_df_info.to_csv(absdir+r'\data\dealeddata\itemifo.csv')
    #从用户数据出发提取商品种类信息
    def extract_catorgory_info(self):
        from Utils.PrintInfo import curline
        from collections import Counter
        catorgory_info = pd.DataFrame()
        catorgory_set = self.user['item_category'].unique()
        catorgory_info['item_category'] = catorgory_set
        # print(catorgory_set)
        catorgory_buy_times_list = [0]*catorgory_set.shape[0]
        catorgory_cart_times_list = [0] * catorgory_set.shape[0]
        catorgory_visit_times_list = [0] * catorgory_set.shape[0]
        catorgory_collection_times_list = [0] * catorgory_set.shape[0]
        catorgory_buy_dict = Counter(self.user.loc[self.user['behavior_type'] == 4]['item_category'])
        catorgory_cart_dict = Counter(self.user.loc[self.user['behavior_type'] == 3]['item_category'])
        catorgory_visit_dict = Counter(self.user.loc[self.user['behavior_type'] == 1]['item_category'])
        catorgory_collection_dict = Counter(self.user.loc[self.user['behavior_type'] == 2]['item_category'])
        catorgory_info['catorgory_buy_times'] = catorgory_buy_times_list
        catorgory_info['catorgory_cart_times'] = catorgory_cart_times_list
        catorgory_info['catorgory_visit_times'] = catorgory_visit_times_list
        catorgory_info['catorgory_collection_times'] = catorgory_collection_times_list
        for rindex in range(len(catorgory_set)):
            key = catorgory_set[rindex]
            if key in catorgory_buy_dict.keys():
                catorgory_info['catorgory_buy_times'][rindex] = catorgory_buy_dict[key]
            if key in catorgory_cart_dict.keys():
                catorgory_info['catorgory_cart_times'][rindex] = catorgory_cart_dict[key]
            if key in catorgory_visit_dict.keys():
                catorgory_info['catorgory_visit_times'][rindex] = catorgory_visit_dict[key]
            if key in catorgory_collection_dict.keys():
                catorgory_info['catorgory_collection_times'][rindex] = catorgory_collection_dict[key]
        # print(catorgory_info)
        self.catorgory_info  = catorgory_info
        pass
    #数据相关性分析
    def Relevance_Analysis(self):
        #self.user
        user_behavior_count = self.user_behavior_count
        cov_between_visit_and_buy = user_behavior_count[0].corr(user_behavior_count['buy_item_count'])
        userid = user_behavior_count.iloc[:,0].index
        visit_times = user_behavior_count[0]
        buy_times = user_behavior_count[3]
        # print(visit_times)
        # print(userid)
        # plt.scatter([i for i in range(len(userid))],visit_times,alpha=0.5,s = 16)
        # plt.scatter([i for i in range(len(userid))], buy_times*50,alpha=0.5,s = 16)
        #画分布图
        for i in range(4):
            x = user_behavior_count[i]
            sns.set()  # 切换到seaborn的默认运行配置
            sns.distplot(x)# 默认
            plt.xlabel(BEHAVIOR_MAP[i]+" times")
            plt.ylabel('frequency')
            plt.title('the distribution of '+BEHAVIOR_MAP[i])
            plt.xlim(0,max(x))
            plt.savefig('../graph/'+BEHAVIOR_MAP[i]+'.jpg')
            plt.close()
            # 画散点图
        for i in range(4):
            x = user_behavior_count[i]
            plt.ylabel(BEHAVIOR_MAP[i] + " times")
            plt.xlabel('id')
            plt.title('the scatter of ' + BEHAVIOR_MAP[i])
            plt.xlim(0, max(x))
            plt.scatter([i for i in range(len(userid))], x, alpha=0.5, s=16)
            plt.savefig('../graph/scatter of' + BEHAVIOR_MAP[i] + '.jpg')
            plt.close()
        #相关性分析
        hitmapTemp =  user_behavior_count.iloc[:,0:4]
        hitmapData = hitmapTemp.corr()
        f,ax = plt.subplots(figsize=(12,12))
        sns.heatmap(hitmapData,vmax = 1,square=True)
        plt.savefig('../graph/correlation.png')
        hitmap_dict = hitmapData[3].to_dict()
        del hitmap_dict[3]
        print("List the numerical features decendingly by their correlation with Sale Price:\n")
        for ele in sorted(hitmap_dict.items(), key=lambda x: -abs(x[1])):
            print(ele)
        # for each in item['item_id']:
        #     print(each)
    def Feature_merge(self):
        self.extract_behavior_info()
        self.extract_item_info()
        self.extract_catorgory_info()
        self.mergeres = pd.merge(self.mergeres, self.user_behavior_count, on='user_id')
        self.mergeres = pd.merge(self.mergeres,self.item_info,on = 'item_id')
        self.mergeres = pd.merge(self.mergeres,self.catorgory_info,on = 'item_category')

        self.mergeres = self.mergeres.drop_duplicates().reset_index(drop=True)
        # if not os.path.exists('../data/dealeddata/mergeddata.csv'):
        self.mergeres.to_csv('../data/dealeddata/mergeddata.csv',index=False)
#       #先来merge一下
#         self.mergeres = pd.merge(self.mergeres,self.user_behavior_count,on = 'user_id',how='outer')
#         self.mergeres = pd.merge(self.mergeres, self.user_behavior_count, on='user_id')
#         self.mergeres = pd.merge(self.mergeres, self.item_info, on='item_id')
#         self.mergeres.to_csv('mergeres.csv',index=False)


if __name__=='__main__':
    #直接运行就可以在data/dealeddata/下 生成一个merge后的csv，目前特征向量数目较少，但框架基本都有了
    #还有待进一步完善尤其是时间信息，
    from Utils.myutils import split_train_and_test

    user_csv,item_csv,mergeres = LoadData()
    F_extractor =  Feature_Extractor(user_csv,item_csv,mergeres)
    F_extractor.Feature_merge()
