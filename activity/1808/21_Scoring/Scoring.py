# coding: utf-8
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG
from LoadData import load_train_data, load_test_data, load_submission
import pandas as pd 
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ログの出力名を設定
logger = getLogger(__name__)

# ログのフォーマットを設定（詳細は次のマークダウンテーブル参考）
fmt = Formatter('%(asctime)s %(name)s %(lineno)d %(levelname)s %(message)s')

# ログのコンソール出力の設定
shandler = StreamHandler()
shandler.setLevel('INFO')
shandler.setFormatter(fmt)

# ログのファイル出力先の設定
fhandler = FileHandler('result_tmp/Scoring.log')
fhandler.setLevel(DEBUG)
fhandler.setFormatter(fmt)

# ログレベルの設定
logger.setLevel(DEBUG)
logger.addHandler(shandler)
logger.addHandler(fhandler)
logger.propagate = False


def RandomForest():
	logger.info('RandomForestRegressor start')
	logger.debug('make_train_data start')
	train = pd.read_csv('./result_tmp/scaled_train.csv')
	
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month'], axis=1).values
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	logger.debug('make_train_dat end')

	logger.info('Fitting start')
	forest = RandomForestRegressor(n_estimators=50, random_state=1)
	forest.fit(X, y)
	logger.debug('Fitting end')
	
	logger.info('Scoring start')
	#logger.info('Accuracy on test set: {:.3f}'.format(forest.score(X_test, y_test)))
	test_data = load_test_data()
	test = test_data.drop(['ID'], axis=1).values
	
	submission = load_submission()
	submission['item_cnt_month'] = forest.predict(test).astype(np.float16).clip(0., 20.)
	submission.to_csv('./result_tmp/submit_180826_1st.csv', encoding='utf-8-sig', index=False)
	logger.info('submission:\n{}'.format(submission.head()))
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')


if __name__ == '__main__':
	RandomForest()
