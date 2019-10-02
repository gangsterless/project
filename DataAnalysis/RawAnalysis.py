import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from Utils.PrintInfo import curline
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
    user_csv = pd.read_csv(userdir+r'dealeddata/train_feature.csv')
    #item先不处理吧
    item_csv = pd.read_csv(itemdir+r'raw/tianchi_fresh_comp_train_item.csv')
    return user_csv,item_csv
class Feature_Extractor:

    def __init__(self,user,item):
        self.user = user
        self.item = item
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
        for each in self.user_id_set:
            tmpdict = {}
            tmplist = []
            each_user_visit = self.user.loc[(self.user['user_id'] == each )]
            #浏览
            each_user_read_count = each_user_visit.loc[ each_user_visit['behavior_type']==1].shape[0]
            #收藏
            each_user_colection_count = each_user_visit.loc[each_user_visit['behavior_type'] == 2].shape[0]
            #加购物车
            each_user_cart_count = each_user_visit.loc[each_user_visit['behavior_type'] == 3].shape[0]
            #购买
            each_user_buy = each_user_visit.loc[ each_user_visit['behavior_type']==4]
            each_user_buy_count = each_user_buy.shape[0]
            activegrade =  F_active_grade(each_user_read_count,each_user_colection_count,each_user_cart_count,each_user_buy_count)
            active_days_list.append(activegrade)
            #加购量
            if each_user_buy_count>0:
                #挑选出最近一次购买时间,精确到小时
                each_user_buy_list = each_user_buy['time'].tolist()
                sortedlast = SortByTime(each_user_buy_list)[-1]
                last_buy_time = time.strptime(sortedlast, "%Y-%m-%d %H")
                predictday = time.strptime('2014-12-18 00', "%Y-%m-%d %H")
                time1 = datetime.datetime(last_buy_time[0], last_buy_time[1], last_buy_time[2],last_buy_time[3])
                time2 = datetime.datetime(predictday[0], predictday[1], predictday[2],predictday[3])
                #最后一次买的时间距预测时间的时长，单位是小时
                last_buy_diff_list.append((time2-time1).days*24+(time2-time1).seconds//3600)
                #买了几种商品

                c1 = each_user_buy['item_id'].count()
                c2  =  len(each_user_buy['item_id'].unique())
                #获取买的最多商品的id
                each_user_buy_list =  each_user_buy['item_id'].tolist()
                max_buy_id =  Counter(each_user_buy_list).most_common(1)

                Add_purchase_list.append(c1-c2)
                purchase_item_count_list.append(c2)
                max_buy_id_list.append(max_buy_id)

            else:
                Add_purchase_list.append(0)
                purchase_item_count_list.append(0)
                max_buy_id_list.append('')
                #如果没有买过就填充非数字
                last_buy_diff_list.append(np.nan)
            tmplist.extend([each_user_read_count,each_user_colection_count,each_user_cart_count,each_user_buy_count])
            totaldict[each] = tmplist
        user_behavior_count_df  = (pd.DataFrame(totaldict)).T
        user_behavior_count_df['Add_buy'] = Add_purchase_list
        user_behavior_count_df['buy_item_count'] = purchase_item_count_list
        user_behavior_count_df['max_buy_id'] = max_buy_id_list
        user_behavior_count_df['active_grade'] = active_days_list
        user_behavior_count_df['last_buy_diff'] = last_buy_diff_list
        print(user_behavior_count_df)

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
        print(item_df_info.where(item_df_info['buy_times']>0).dropna())
        print(item_df_info.where(item_df_info['visit_times']>0).dropna())
        print(item_df_info.where(item_df_info['collection_times'] > 0).dropna())
        print(item_df_info.where(item_df_info['cart_times'] > 0).dropna())

        print(item_df_info)
        # item_df_info.to_csv(absdir+r'\data\dealeddata\itemifo.csv')
    #从用户数据出发提取商品种类信息
    def extract_catorgory_info(self):
        from Utils.PrintInfo import curline
        from collections import Counter
        catorgory_info = pd.DataFrame()
        catorgory_set = self.user['item_category'].unique()
        catorgory_info['item_category_id'] = catorgory_set
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
        print(catorgory_info)
        pass
        # for each in item['item_id']:
        #     print(each)





if __name__=='__main__':
    from Utils.myutils import split_train_and_test

    user_csv,item_csv = LoadData()
    # split_train_and_test(user_csv,item_csv)
    # DataDes(user_csv,item_csv)

    # extract_feature(user_csv,item_csv)
    F_extractor =  Feature_Extractor(user_csv,item_csv)
    # F_extractor.extract_behavior_info()
    # F_extractor.extract_geo_info()
    # test = pd.read_csv(r'D:\Data\big3data\recommendsys\project\data\dealeddata' + r'\user_orderedtime.csv')
    # print(eval(test['v'][0])[0])
    # print(test.head())
    # split_train_and_test(user_csv,item_csv)
    # DataDes(user_csv,item_csv)
    # F_extractor.extract_item_info()
    F_extractor.extract_catorgory_info()