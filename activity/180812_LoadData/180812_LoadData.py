from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG
import pandas as pd
from IPython.display import display

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


def load_train_data():
    logger.info('read_train start')
    train = pd.read_csv('../../input/sales_train.csv')
    logger.debug('read_train end')
    return train


def load_test_data():
    logger.info('read_test start')
    test = pd.read_csv('../../input/test.csv')
    logger.debug('read_test end')
    return test


def load_submission():
    logger.info('read_submission start')
    submit = pd.read_csv('../../input/sample_submission.csv')
    logger.debug('read_submission end')
    return submit


def load_suppliment():
    logger.info('read_csv start')
    item_cat = pd.read_csv('../../input/item_categories.csv')
    item = pd.read_csv('../../input/items.csv')
    shop = pd.read_csv('../../input/shop.csv')
    logger.debug('read_csv end')
    display(item_cat.iloc[:5, :])
    display(item.iloc[:5, :])
    display(shop.iloc[:5, :])
    return item_cat, item, shop


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
    print(load_submission().head())
