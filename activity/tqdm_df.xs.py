# coding: utf-8
get_ipython().run_line_magic('run', 'LoadData.py')
train = load_train_data()
item_cnt_month = train['item_cnt_day'].groupby([train['date_block_num'], train['shop_id'], train['item_id']]).apply(sum)
item_cnt_month
item_cnt_month.head()
item_cnt_month.to_csv('./result_tmp/grouped.csv')
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas(desc='hoge process:')
tqdm_test = train['item_cnt_day'].groupby([train['date_block_num'], train['shop_id'], train['item_id']]).progress_apply(sum)
%save tqdm 1-18


# coding: utf-8
# multiIndexのdfに対するxsの使い方。
item_cnt_month.xs(0, level=1)
item_cnt_month.xs([0, 98], level=['shop_id','item_id'])
item_cnt_month.xs([0, 98], level=['shop_id','item_id'], drop_level=False)
