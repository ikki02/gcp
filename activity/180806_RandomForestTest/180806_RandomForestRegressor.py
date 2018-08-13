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
