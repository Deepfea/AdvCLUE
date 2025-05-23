import numpy as np
import os
import pandas as pd
import jieba

def seg_data(data_df):
    seg_list = []
    for data_num in range(len(data_df)):
        segs = []
        temp_string = data_df.loc[data_num, 'premise']
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        cut_result = list(set(cut_result))
        segs.append(cut_result)
        temp_string = data_df.loc[data_num, 'hypothese']
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        cut_result = list(set(cut_result))
        segs.append(cut_result)
        seg_list.append(segs)

    seg_arr = np.array(seg_list)
    return seg_arr


if __name__ == '__main__':

    pass
