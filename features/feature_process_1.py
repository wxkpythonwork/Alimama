import numpy as np
import pandas as pd
import time

# from sklearn.model_selection import train_test_split,GridSearchCV


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
    #     user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0:'user_query_day'})
    #      #默认inner。left、right分别保留左边、右边部分数据（缺失补NAN）,inner公共部分,outer俩边所有
    #     data = pd.merge(data, user_query_day, how='inner', on=['user_id','user_query_day'])
    #     user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={1:"user_query_day_hour"})
    #     data = pd.merge(data, user_query_day_hour, how='inner', on=['user_id', 'day', 'hour']
    return train_data
	

def features_choose():
    path = 'E:/tensorflow_work/bishe/'
    print('loading data ...')
    train_data = pd.read_csv(path + "round1_ijcai_18_train_20180301.txt", sep=" ")
    print("loading success ...")
    print("dealing with NAN")
    nan_list = ['item_brand_id', 'item_city_id', 'item_sales_level', 'user_age_level', 'user_occupation_id',
                'user_star_level',
                'user_gender_id', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description']
    # 缺失值-1替换为中位数
    for label in nan_list:
        train_data[label] = train_data[label].replace(-1,train_data[label].median())#mean()
        train_data[label] = train_data[label].replace(-1,train_data[label].median())

    train_data.drop_duplicates(inplace=True)  # 默认为所有行相同去重，也可以指定列下去重
    train_data = covert_seconds_to_days(train_data)
    train_data.drop(['date', 'context_timestamp'], axis=1, inplace=True)
    
#==============================================================================
#    先不分析属性从属关系,只做统计
#==============================================================================
#    train_data.drop(['item_category_list', 'item_property_list', 'predict_category_property'], axis=1, inplace=True)
    train_data['len_item_category_list'] = train_data.item_category_list.apply(lambda x: len(str(x).split(";"))) 
    train_data['len_item_property_list'] = train_data.item_property_list.apply(lambda x: len(str(x).split(";"))) 
    train_data['len_predict_category_property'] = train_data.predict_category_property.apply(lambda x: len(str(x).split(";"))) 
    train_data.drop(['item_category_list', 'item_property_list', 'predict_category_property'], axis=1, inplace=True)

#==============================================================================
#    归一化
#==============================================================================
    # train_data.context_page_id = train_data.context_page_id.values - 4000
    # train_data.user_age_level = train_data.user_age_level.values - 1000
    train_data.context_page_id = (train_data['context_page_id'] - train_data['context_page_id'].min()) / (train_data['context_page_id'].max() - train_data['context_page_id'].min())                               
    train_data.user_age_level = (train_data['user_age_level'] - train_data['user_age_level'].min()) / (train_data['user_age_level'].max() - train_data[    'user_age_level'].min())                                     
    
#==============================================================================
#    将user_gender_id属性分割成多个(适合比较少类目的属性)
#==============================================================================
    gender_id_class = pd.get_dummies(train_data.user_gender_id)
    gender_class = gender_id_class.rename(columns=lambda x: 'gender_class_' + str(x))
    train_data = pd.concat([train_data, gender_class], axis=1)
    train_data.drop(['user_gender_id'], axis=1, inplace=True)

#==============================================================================
#    item内构建新特征
#==============================================================================
    train_data['item_pv_sales'] = np.sqrt(train_data['item_pv_level'] * train_data['item_sales_level']) 
    train_data['item_pv_collected'] = np.sqrt(train_data['item_pv_level'] * train_data['item_collected_level'])
    train_data['item_total_sales'] = np.sqrt(train_data['item_price_level'] * train_data['item_sales_level']) 
    train_data.drop(['item_pv_level','item_sales_level','item_collected_level','item_price_level'],
                    axis=1, inplace=True)

#==============================================================================
#    shop内构建特征
#==============================================================================
    train_data['shop_mean_score'] = (train_data.shop_score_service + train_data.shop_score_delivery + train_data.shop_score_description) / 3
    train_data['shop_view'] = train_data.shop_review_positive_rate * train_data.shop_star_level
    train_data.drop(['shop_score_service', 'shop_score_delivery', 'shop_score_description'],
                    axis=1, inplace=True)
    train_data.drop(['shop_review_positive_rate','shop_star_level'],
                    axis=1,inplace=True)
	#处理后数据储存的路径
    
    train_data.to_csv(path + "train_data_process.txt",index=True,sep=" ")

if __name__ == '__main__':
	start = time.time()	
	features_choose()	
	end = time.time()
	print("耗时：{}".format(end - start))
	