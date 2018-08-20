from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG
import pandas as pd
from sklearn.model_selection import train_test_split
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
fhandler = FileHandler('result_tmp/RandomForestRegressor.log')
fhandler.setLevel(DEBUG)
fhandler.setFormatter(fmt)

# ログレベルの設定
logger.setLevel(DEBUG)
logger.addHandler(shandler)
logger.addHandler(fhandler)
logger.propagate = False

logger.info('read_csv start')
train = pd.read_csv('../../input/sales_train.csv')
test = pd.read_csv('../../input/test.csv')
submission = pd.read_csv('../../input/sample_submission.csv')
logger.debug('read_csv end')

logger.info('data processing start')
y = train['item_cnt_day']
X = train.drop(['item_cnt_day'], axis=1)

X['date'] = pd.to_datetime(X['date'], format='%d.%m.%Y')
X = X.set_index('date')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logger.debug('data processing end')

logger.info('RandomForestRegressor start')
logger.info('Fitting start')
forest = RandomForestRegressor(n_estimators=50, random_state=1)
forest.fit(X_train, y_train)
logger.debug('Fitting end')
logger.info('Scoring start')
y_predicted = forest.predict(X_test)
logger.info('Accuracy on test set: {:.3f}'.format(forest.score(X_test, y_test)))
logger.debug('RandomForestRegressor end')
logger.debug('====================')



# coding: utf-8
# DataProcess.py
from LoadData import load_train_data, load_test_data, load_suppliment
import pandas as pd 
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG

# ログの出力名を設定
logger = getLogger(__name__)

# ログのフォーマットを設定（詳細は次のマークダウンテーブル参考）
fmt = Formatter('%(asctime)s %(name)s %(lineno)d %(levelname)s %(message)s')

# ログのコンソール出力の設定
shandler = StreamHandler()
shandler.setLevel('INFO')
shandler.setFormatter(fmt)

# ログのファイル出力先の設定
fhandler = FileHandler('result_tmp/loaddata.log')
fhandler.setLevel(DEBUG)
fhandler.setFormatter(fmt)

# ログレベルの設定
logger.setLevel(DEBUG)
logger.addHandler(shandler)
logger.addHandler(fhandler)
logger.propagate = False

def make_item_cnt_month():
	logger.info('MakeItemCntMonth starts')
	train = load_train_data()
	item_cnt_month = train['item_cnt_day'].groupby( \
		[train['date_block_num'], train['shop_id'], train['item_id']]).sum()
	item_cnt_month.name = 'item_cnt_month'
	item_cnt_month_df = pd.DataFrame(item_cnt_month)
	logger.debug(item_cnt_month_df.shape)
	logger.debug(train.shape)
	adjusted_train  = pd.merge(train, item_cnt_month_df, on=['date_block_num', 'shop_id', 'item_id'])
	logger.debug(adjusted_train.shape)
	adjusted_train.to_csv('./result_tmp/adjusted_train', encoding='utf-8-sig')
	logger.debug('MakeItemCntMonth ends')
	return adjusted_train


def clean_train(adjusted_train):
	logger.info('CleanTrain starts')
	adjusted_train.drop(['date_block_num', 'item_price', 'item_cnt_day'], axis=1, inplace=True)
	adjusted_train['date'] = pd.to_datetime(adjusted_train['date'])
	adjusted_train = adjusted_train.set_index('date')
	logger.debug(adjusted_train.shape)
	logger.debug('CleanTrain ends')
	return adjusted_train


if __name__ == '__main__':
	adjusted_train = make_item_cnt_month()
	adjusted_train = clean_train(adjusted_train)
