import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import log_loss
import lightgbm as lgb
# from sklearn.model_selection import train_test_split,GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def timestamp_convert(value):
    format = '%Y-%m-%d %H:%M:%S'  # 要转化的格式
    value = time.localtime(value)
    # print(value)
    newdata = time.strftime(format, value)
    return newdata


def covert_seconds_to_days(train_data):
    # frame['days'] = (pd.to_datetime(frame['date']) - parse('2015-01-01')).dt.days#转化为时间
    train_data['date'] = train_data['context_timestamp'].apply(timestamp_convert)  # 将这一列的每个数据进行一一转换
    train_data['day'] = train_data.date.apply(lambda x: int(x[8:10]))  # 将这一列的每个数据进行一一切片
    train_data['hour'] = train_data.date.apply(lambda x: int(x[11:13]))
    return train_data
	

def features_choose(train_data):
    train_data = covert_seconds_to_days(train_data)   
    
#==============================================================================
#                           user_features + day hour
#==============================================================================
    train_data['user_age0'] = train_data['user_age_level'].apply(lambda x: 1 if x>=1003 else 2)
    train_data['user_gender_id0'] = train_data.user_gender_id.apply(lambda x: 1 if x==-1 else 2)
    train_data['user_occupation0'] = train_data.user_occupation_id.apply(lambda x: 1 if x==-1 | x==2003 else 2)
    
    def map_user_star(x):
        if (x==-1)&(x==3000):
            return 1
        elif (x>=3001)&(x<=3008):
            return 2    
        else:
            return 3
    train_data['user_star0'] = train_data.user_star_level.map(map_user_star)#apply(lambda x: 1 if x==-1 | x==3000 else 2)
    
    train_data['context_page0'] = train_data.context_page_id.apply(lambda x: 1 if x <= 4007 else 2)
    
    def map_hour(x):
        if (x>=7)&(x<=12):
            return 1
        elif (x>=13)&(x<=20):
            return 2
        elif (x>=0)&(x<=6):
            return 3            
        else:
            return 4
    def map_day(x):
        if (x>=18)&(x<=19):
            return 1
        elif (x>=20)&(x<=22):
            return 2             
        else:
            return 3        
    train_data['hour_class'] = train_data.hour.apply(map_hour)
    train_data['day_class'] = train_data.hour.apply(map_day)
#==============================================================================
#             将user_gender_id属性分割成多个(适合比较少类目的属性)
#    gender_id_class = pd.get_dummies(train_data.user_gender_id)
#    gender_class = gender_id_class.rename(columns=lambda x: 'gender_class_' + str(x))
#    train_data = pd.concat([train_data, gender_class], axis=1)
#    train_data.drop(['user_gender_id'], axis=1, inplace=True)

#==============================================================================
#                             item内构建新特征
#==============================================================================
    train_data['item_pv_sales'] = np.sqrt(train_data['item_pv_level'] * train_data['item_sales_level'])
    train_data['item_price_level0'] = train_data.item_price_level.apply(lambda x: 1 if x <= 4 else 2)
    train_data['item_collected0'] = train_data.item_collected_level.apply(lambda x: 1 if x>=11 and x <= 15 else 2)
    train_data['item_pv_collected'] = np.sqrt(train_data['item_pv_level'] * train_data['item_collected_level'])
    def map_sales(x):
        if x <= 6:
            return 1
        elif (x >= 7)&(x<=11):
            return 2             
        elif (x >= 12)&(x<=16):
            return 3
        else:
            return 4  
    train_data['item_sales0'] = train_data.item_sales_level.map(map_sales)            
    train_data['item_total_sales'] = np.sqrt(train_data['item_price_level'] * train_data['item_sales_level']) 
    
#                              Label_encoder
    train_data['len_item_category_list'] = train_data.item_category_list.apply(lambda x: len(str(x).split(";"))) 
    train_data['len_item_property_list'] = train_data.item_property_list.apply(lambda x: len(str(x).split(";"))) 
    train_data['len_predict_category_property'] = train_data.predict_category_property.apply(lambda x: len(str(x).split(";"))) 
    lbl = preprocessing.LabelEncoder()
    for i in ['user_id','item_id','item_brand_id','item_city_id']:
        train_data[i] = lbl.fit_transform(train_data[i])    
    for i in range(1,3):
        train_data['item_category_list' + str(i)] = lbl.fit_transform(train_data['item_category_list'].map
                   (lambda x : str(str(x).split(";"))[i] if len(str(x).split(";")) > i else ""))
    count_vec = TfidfVectorizer()
    train_data00 = count_vec.fit_transform(train_data['item_property_list'])
    print(train_data00)
#    for i in range(1,4):
#        train_data['predict_category_property' + str(i)] = lbl.fit_transform(train_data['predict_category_property'].map
#           (lambda x : str(str(x).split(";"))[i] if len(str(x).split(";"))>i else ""))
#    train_data.drop(['item_category_list', 'item_property_list', 'predict_category_property'], axis=1, inplace=True)

#==============================================================================
#                               shop内构建特征
#==============================================================================
    #print(train_data.columns)
    # train_data['shop_score_service0'] = train_data.shop_score_service.apply(lambda x: 1 if x>=0.96 and x<=0.98 else 2)
    # train_data['shop_score_delivery0'] = train_data.shop_score_delivery.apply(lambda x: 1 if x>=0.96 and x<=0.98 else 2)
    # train_data['shop_score_description0'] = train_data.shop_score_description.apply(lambda x: 1 if x>=0.96 and x<=0.98 else 2)
    #
    train_data['shop_star0'] = train_data['shop_star_level'].apply(lambda x: 1 if ((x>=5012) & (x<=5015)) else 2)
    
    train_data['shop_mean_score'] = (train_data.shop_score_service + train_data.shop_score_delivery + train_data.shop_score_description) / 3

    train_data['shop_view'] = train_data.shop_review_positive_rate / train_data.shop_star_level

#    train_data.drop(['shop_score_service', 'shop_score_delivery', 'shop_score_description'],
#                    axis=1, inplace=True)
#    train_data.drop(['shop_review_positive_rate','shop_star_level'],
#                    axis=1,inplace=True)

#==============================================================================
#                               剩余的缺失值-1替换为中位数
#==============================================================================
    nan_list = ['item_brand_id', 'item_city_id', 'item_sales_level', 'user_age_level', 'user_occupation_id','user_star_level',
                 'user_gender_id', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
    for label in nan_list:
        train_data[label] = train_data[label].replace(-1,train_data[label].median())#mean()
        train_data[label] = train_data[label].replace(-1,train_data[label].median())
    train_data.drop(['item_pv_level','item_sales_level','item_collected_level','item_price_level'],
                    axis=1, inplace=True)
    if online ==False:
        train_data.drop_duplicates(inplace=True)  # 默认为所有行相同去重，也可以指定列下去重;  测试集不去重否则可能提交有问题
    
#==============================================================================
#                               要删除的特征
#==============================================================================
    train_data.drop(['date', 'context_timestamp'], axis=1, inplace=True)
    train_data.drop(['item_category_list', 'item_property_list', 'predict_category_property'], axis=1, inplace=True)
    
    return train_data

#==============================================================================
#                                  MODEL
#==============================================================================
def lgb_baselinemodel():
    path = 'E:/tensorflow_work/bishe/'
    print('loading data ...')
    train_data = pd.read_csv(path + "round1_ijcai_18_train_20180301.txt", sep=" ")
    print("processing features...")
    train_data = features_choose(train_data)
    features = train_data.columns.tolist()
    features.remove('is_trade')#[i for i in train_data.columns.tolist() if i!='is_trade']
    print(features)
    label = 'is_trade'
    print("starting training")
#==============================================================================
#                                 线下得分
#==============================================================================
    if online == False:
        training_data = train_data[(train_data['day'] <= 23) & (train_data['day'] >= 18)]
        train_test_data = train_data[train_data['day'] == 24]
        
#==============================================================================
#         clf = lgb.LGBMClassifier(
#         objective='binary',
#         num_leaves=30,
#         depth=5,
#         learning_rate=0.08,
#         seed=2018,
#         colsample_bytree=0.3,
#         subsample=0.8,
#         n_jobs=10,
#         n_estimators=1000)
#         lgb0.fit(training_data, training_data.loc[0:len(training_data)-1,'is_trade'], eval_set=[(train_test_data, train_test_data.loc[0:len(train_test_data)-1,'is_trade'])],
#                         early_stopping_rounds=10)
#==============================================================================
        clf = lgb.LGBMClassifier(objective='binary',num_leaves=64, max_depth=4, learning_rate=0.06, 
                                 n_estimators=1000, seed=2018, n_jobs=20, subsample=0.9)        
        clf.fit(training_data[features], training_data[label], feature_name=features)
        train_test_data['pred'] = clf.predict_proba(train_test_data[features], )[:, 1]
        logloss = log_loss(train_test_data['is_trade'], train_test_data['pred'])
        print("########################################")
        print('logloss value：{}'.format(logloss))
        print("########################################")	
        
    test_data = pd.read_csv(path + "round1_ijcai_18_test_a_20180301.txt", sep=" ")
    test_data = features_choose(test_data)
    if online == True:
        train_data = train_data.copy()
        model = lgb.LGBMClassifier(num_leaves=64, max_depth=7, learning_rate=0.06, n_estimators=400, seed=2018, n_jobs=20, subsample=0.9)
        model.fit(train_data[features], train_data[label], feature_name=features)
        test_data['proba'] = model.predict_proba(test_data[features],)[:, 1]
        test_data[['instance_id','proba']].to_csv('lgbmodel_result.csv', index=False, sep=" ")

if __name__ == '__main__':
	start = time.time()
	online = False
	lgb_baselinemodel()
	print("运行耗时：", (time.time() - start))


	