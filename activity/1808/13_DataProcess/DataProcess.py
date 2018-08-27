# coding: utf-8
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
fhandler = FileHandler('result_tmp/Input.log')
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
		[train['date_block_num'], train['shop_id'], train['item_id']]).apply(sum)
	item_cnt_month.name = 'item_cnt_month'
	item_cnt_month_df = pd.DataFrame(item_cnt_month)
	logger.debug(item_cnt_month_df.shape)
	item_cnt_month.to_csv('./result_tmp/scaled_train.csv', encoding='utf-8-sig')
	logger.debug('MakeItemCntMonth ends')
	return item_cnt_month


if __name__ == '__main__':
	scaled_train = make_item_cnt_month()
