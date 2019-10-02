import os
import datetime
import sys
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(dir)
#行数 23291028,太大了也
def Getlines(name):#获得行数
    count = 0
    for index, line in enumerate(open(name, 'r')):
        count += 1
    print(count)

def CutFile(name, size):#路径名和每个文件的大小，单位是行数
    i = 0
    findex = 0
    with open(name,'r')as f:
        subdir = dir+'/'+'data/raw/tianchi_fresh_comp_train_user'+str(findex)+'.csv'
        if not os.path.exists(subdir):
            file = open(subdir,'w')
            wr  =True
        else:
            wr = False
        for line in f:
            # print(line)
            if wr:
                file.write(line)
            i+=1
            if i>=size:
                if wr:
                    file.close()
                i = 0
                findex+=1
                subdir = dir + '/' + 'data/raw/tianchi_fresh_comp_train_user' + str(findex) + '.csv'
                if not os.path.exists(subdir):
                    file = open(subdir, 'w')
                    wr  = True
def SortByTime(timeseries):
    import operator
    import functools
    def cmp_datetime(a, b):
        a_datetime = datetime.datetime.strptime(a, '%Y-%m-%d %H')
        b_datetime = datetime.datetime.strptime(b, '%Y-%m-%d %H')

        if a_datetime > b_datetime:
            return -1
        elif a_datetime < b_datetime:
            return 1
        else:
            return 0

    res = sorted(timeseries, key=functools.cmp_to_key(cmp_datetime),reverse=True)
    return res
#用来分割测试集，训练集，和验证集
def split_train_and_test(userdf,itemdf):
    def standardlizedict(d):
        klist = []
        vlist = []
        for k,v in d.items():
            klist.append(k)
            vlist.append(v)
        resdict = {'user_id':klist,'isbuy':vlist}
        return resdict
    import time
    import pandas as pd
    '''

    Train Feature Span :11.18~12.15
    Train Target Span: 12.16
    Train Feature Span :11.18~12.15 
Train Target Span: 12.16 
测试：12.17 —> 从11.18~12.16提取特征 
验证：12.18 —> 从11.18~12.17提取特征
    :return:
    '''
    # date_range = pd.date_range(start='2014-11-18 00'.format("%Y-%m-%d %H"),end='2014-12-16 24'.format("%Y-%m-%d %H"))
    user_id_set = pd.unique(userdf['user_id']).tolist()
    user_isbuy_value_train = [0]*len(user_id_set)
    user_isbuy_value_test = [0]*len(user_id_set)
    user_isbuy_value_validatin = [0]*len(user_id_set)
    userdf_isbuy_train_dict = dict(zip(user_id_set,user_isbuy_value_train))
    userdf_isbuy_test_dict = dict(zip(user_id_set, user_isbuy_value_test))
    user_isbuy_validation_dict = dict(zip(user_id_set,user_isbuy_value_validatin))
    #窗口提取
    #训练集特征提取的开始时间，结束时间
    train_feature_starttime = time.strptime("2014-11-18 00","%Y-%m-%d %H")
    train_feature_endtime = time.strptime("2014-12-15 23","%Y-%m-%d %H")
    train_label_starttime = time.strptime("2014-12-16 00","%Y-%m-%d %H")
    train_label_endtime = time.strptime("2014-12-16 23", "%Y-%m-%d %H")

    # 测试集特征提取的开始时间，结束时间
    test_feature_starttime = time.strptime("2014-11-18 00", "%Y-%m-%d %H")
    test_feature_endtime = time.strptime("2014-12-16 23", "%Y-%m-%d %H")
    test_feature_label_starttime = time.strptime("2014-12-17 00", "%Y-%m-%d %H")
    test_feature_label_endtime = time.strptime("2014-12-17 23", "%Y-%m-%d %H")

    # 验证集特征提取的开始时间，结束时间
    validation_feature_starttime = time.strptime("2014-11-18 00","%Y-%m-%d %H")
    validation_feature_endtime =  time.strptime("2014-12-17 23", "%Y-%m-%d %H")
    validation_label_starttime =  time.strptime("2014-12-18 00","%Y-%m-%d %H")
    validation_label_endtime = time.strptime("2014-12-18 23", "%Y-%m-%d %H")


    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    validation_feature = []
    validation_label = []
    print(userdf.index)
    for rindex in userdf.index:
        print(rindex)
        thistime =  userdf.loc[rindex]['time']
        thistime = time.strptime(thistime, "%Y-%m-%d %H")
        tmpdf = userdf.iloc[rindex]
        user_id = tmpdf['user_id']
        tmpdf.tolist()


        if thistime>train_feature_starttime and thistime<train_feature_endtime:
            train_feature.append(tmpdf)
        if thistime>train_label_starttime and thistime<train_label_endtime:
            if tmpdf['behavior_type']==4:
                userdf_isbuy_train_dict[tmpdf['user_id']]=1

        if thistime>test_feature_starttime and thistime<test_feature_endtime:
            test_feature.append( tmpdf)
        if thistime>test_feature_label_starttime and thistime<test_feature_label_endtime:
            if tmpdf['behavior_type']==4:
                userdf_isbuy_test_dict[tmpdf['user_id']]=1
        if thistime>validation_feature_starttime and thistime<validation_feature_endtime:
            validation_feature.append(tmpdf)
        if thistime>validation_label_starttime and thistime<validation_label_endtime:
            if tmpdf['behavior_type']==4:
                user_isbuy_validation_dict[tmpdf['user_id']]=1

    # train_user_df =  userdf.loc[userdf['time'] in date_range]

    train_feature_df = pd.DataFrame(train_feature)
    test_feature_df = pd.DataFrame(test_feature)
    validation_feature_df = pd.DataFrame(validation_feature)
    #必须要符合转化为dataframe的格式
    userdf_train_label = pd.DataFrame(standardlizedict(userdf_isbuy_train_dict))
    userdf_test_label = pd.DataFrame(standardlizedict(userdf_isbuy_test_dict))
    userdf_validation_label  = pd.DataFrame(standardlizedict(user_isbuy_validation_dict))

    print(userdf_train_label.describe())
    print(userdf_test_label.describe())
    print(userdf_validation_label.describe())
    print(train_feature_df.describe())
    print(test_feature_df.describe())
    print(validation_feature_df.describe())
    userdf_train_label.to_csv(dir+r'\data\dealeddata\userdf_train_label.csv')
    userdf_test_label.to_csv(dir+r'\data\dealeddata\userdf_test_label.csv')
    userdf_validation_label.to_csv(dir+r"\data\dealeddata\userdf_validation_label.csv")
    train_feature_df.to_csv(dir+r"\data\dealeddata\train_feature.csv")
    test_feature_df.to_csv(dir+r"\data\dealeddata\test_feature.csv")
    validation_feature_df.to_csv(dir+r"\data\dealeddata\validation_feature.csv")

#计算召回率
def computeRecall(P, F1):
    R = F1*P/(2*P-F1)
    print ("P: %f"%P)
    print ("R: %f"%R)
    print ("F1: %f"%F1)
#评估结果
def evaluate(prediction,result):

    print('Prediction set size: %d' % len(prediction))
    print ('Result set size: %d' % len(result))
    prediction = set(prediction)
    result = set(result)

    intersection = prediction & result

    precision = float(len(intersection))/len(prediction)*100
    recall = float(len(intersection))/len(result)*100

    F1 = 2 * precision * recall / (precision + recall)

    print ('P : %2f' % precision)
    print ('R : %2f' % recall)
    print ('F1: %2f' % F1)
    return precision, recall, F1
#先弄100万行，后续用pickle搞
# CutFile(dir+'/'+'data/raw/tianchi_fresh_comp_train_user.csv',1000000)
# Getlines(dir+'/'+'data/raw/tianchi_fresh_comp_train_user.csv')
# SortByTime(['2017-09-21 02', '2017-09-15 23', '2017-09-18 04'])