# coding: utf-8
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG
from LoadData import load_train_data, load_test_data, load_submission
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit, GroupKFold, GridSearchCV
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


def forest_submit():
	logger.info('RandomForestRegressor start')
	logger.debug('make_train_data start')
	#train = pd.read_csv('./result_tmp/scaled_train.csv')
	train = pd.read_csv('./result_tmp/scaled_train_DateBlockNum.csv')
	#train = train[train['date_block_num']==33]  #直近1ヶ月
	train = train.loc[(30<train['date_block_num'])&(train['date_block_num']<=33)]  #直近3m
	
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month', 'date_block_num'], axis=1).values
	#X = train.drop(['item_cnt_month'], axis=1).values
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	logger.debug('make_train_data end')

	logger.info('Fitting start')
	forest = RandomForestRegressor(n_estimators=50, random_state=1)
	forest.fit(X, y)
	logger.debug('Fitting end')

	#EDAしたいとき
	#fti = forest.feature_importances_
	#print('Feature Importances:')
	#for i, feature in enumerate(train.colunms):
	#	print('\t{0:10s}:{1:>.6f}'.format(feature, fti[i]))

	logger.info('Scoring start')
	#logger.info('Accuracy on test set: {:.3f}'.format(.score(X_test, y_test)))
	test_data = load_test_data()
	test = test_data.drop(['ID'], axis=1).values
	
	submission = load_submission()
	submission['item_cnt_month'] = forest.predict(test).astype(np.float16).clip(0., 20.)
	#submission.to_csv('./result_tmp/submit_180826_1st.csv', encoding='utf-8-sig', index=False)
	submission.to_csv('./result_tmp/submit_180827_31-33.csv', encoding='utf-8-sig', index=False)
	logger.info('submission:\n{}'.format(submission.head()))
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')


def forest_cv():
	logger.info('RandomForestRegressor start')

	logger.debug('make_train_data start')
	train = pd.read_csv('./result_tmp/scaled_train.csv')
	#train = pd.read_csv('./result_tmp/scaled_train_DateBlockNum.csv')
	#train = train[train['date_block_num']==33]  #直近1ヶ月
	#train = train.loc[(30<train['date_block_num'])&(train['date_block_num']<=33)]  #直近3m
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month'], axis=1).values
	#X = train.drop(['item_cnt_month', 'date_block_num'], axis=1).values
	logger.debug('make_train_data end')

	logger.info('Cross-validation start')
	forest = RandomForestRegressor(n_estimators=50, random_state=1)
	
	#cvはKFoldのshuffle引数をTrue/Falseから選べる。
	#ただのkfoldしたいときは上記引数をFalseにすればよい。その際、インデックス順にk分割する。
	#shuffle_cvしたいときは上記引数をTrueにすればよい。毎回分割が変わる。
	kfold = KFold(n_splits=3, shuffle=True, random_state=0)
	#skf = StratifiedKFold(n_splits=3)  #skfはkfoldと同じ引数をもつ。
	#tscv = TimeSeriesSplit(n_splits=3)
	scores = cross_val_score(forest, X, y, cv=kfold)
	#以下、GroupKFoldを使うときの書き方
	#groups = list(train['date_block_num'])
	#scores = cross_val_score(forest, X, y, groups, cv=GroupKFold(n_splits=3))
	logger.info('Cross-validation scores_forest: {}'.format(scores))
	logger.info('Average Cross-validation score_forest: {}'.format(scores.mean()))
	logger.debug('Cross-validation end')
	
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')
	

def forest_gscv():
	logger.info('RandomForestRegressor start')

	logger.debug('make_train_data start')
	train = pd.read_csv('./result_tmp/scaled_train.csv')
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month'], axis=1).values
	logger.debug('make_train_data end')

	logger.info('GridSearchCV start')
	param_grid = {'n_estimators':[10, 30, 50],
				 'random_state':[1, 2, 3]}
	logger.debug('Parameter grid:\n{}'.format(param_grid))
	grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=4)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	grid_search.fit(X_train, y_train)
	logger.info('Best GridSearchCV parameters_forest: {}'.format(grid_search.best_params_))
	logger.info('Best GridSearchCV score_forest: {}'.format(grid_search.best_score_))
	logger.info('Test set score_forest: {:.2f}'.format(grid_search.score(X_test, y_test)))
	results = pd.DataFrame(grid_search.cv_results_)
	results.to_csv('./result_tmp/GridSearch.csv', encoding='utf-8-sig', index=False)
	logger.debug('GridSearchCV end')
	
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')


def forest_gscv_test():
	logger.info('RandomForestRegressor start')

	logger.debug('make_train_data start')
	train = pd.read_csv('./result_tmp/scaled_train.csv')
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month'], axis=1).values
	logger.debug('make_train_data end')

	logger.info('GridSearchCV start')
	param_grid = {'n_estimators':[5, 100, 500],
				  'random_state':[1, 2, 3]}
	logger.debug('Parameter grid:\n{}'.format(param_grid))
	grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	grid_search.fit(X_train, y_train)
	logger.info('Best GridSearchCV parameters_forest: {}'.format(grid_search.best_params_))
	logger.info('Best GridSearchCV score_forest: {}'.format(grid_search.best_score_))
	logger.info('Test set score_forest: {:.2f}'.format(grid_search.score(X_test, y_test)))
	results = pd.DataFrame(grid_search.cv_results_)
	results.to_csv('./result_tmp/GridSearch_test2.csv', encoding='utf-8-sig', index=False)
	logger.debug('GridSearchCV end')
	
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')


if __name__ == '__main__':
	#forest_submit()
	#forest_cv()
	#forest_gscv()
	forest_gscv_test()
