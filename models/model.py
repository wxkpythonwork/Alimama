import time
import pandas as pd
from sklearn.metrics import log_loss
import lightgbm as lgb

def lgb_baselinemodel(train_data):
    
    features = [i for i in train_data.columns.tolist() if i!='is_trade']#train_data.columns.tolist().remove('is_trade')
    label = ['is_trade']
#   线下得分
    if online == False:
        training_data = train_data[(train_data['day'] < 24) & (train_data['day'] >= 18)]
        train_test_data = train_data[train_data['day'] == 24]
        clf = lgb.LGBMClassifier(objective='binary',num_leaves=40, max_depth=7, learning_rate=0.05, n_estimators=400, seed=2018, n_jobs=20, subsample=0.9)
        clf.fit(training_data[features], training_data[label], feature_name=features)
        train_test_data['pred'] = clf.predict_proba(train_test_data[features], )[:, 1]
        logloss = log_loss(train_test_data['is_trade'], train_test_data['pred'])
        print('logloss value：{}'.format(logloss))

#    if online == True:
#        train_data = train_data.copy()
#        model = lgb.LGBMClassifier(num_leaves=64, max_depth=7, n_estimators=80, n_jobs=20)
#        model.fit(train_data[features], train_data[label], feature_name=features)
#        test_data['proba'] = model.predict_proba(test_data[features],)[:, 1]
#        test_data[['instance_id','proba']].to_csv('baseline_lgb.csv', index=False, sep=" ")

if __name__ == '__main__':
	start = time.time()
	online = False
	path = "E:/tensorflow_work/bishe/"
	train_data = pd.read_csv(path + "train_data_process.txt", sep=" ")
	#test_data = pd.read_csv(path + "test_data_process.txt", sep=" ")
	lgb_baselinemodel(train_data)
	print("运行耗时：", (time.time() - start))