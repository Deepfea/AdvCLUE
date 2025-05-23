import numpy as np
import os
import pandas as pd
import jieba
import re

def seg_data(data_df):
    seg_list = []
    for data_num in range(len(data_df)):
        left_part, right_part = re.split(data_df.loc[data_num, 'tag'], data_df.loc[data_num, 'text'])
        segs = []
        temp_string = left_part
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        segs.extend(cut_result)
        temp_string = right_part
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        segs.extend(cut_result)
        segs = list(set(segs))
        # segs = set(segs)
        seg_list.append(segs)
    seg_arr = np.array(seg_list)
    return seg_arr


if __name__ == '__main__':
    pass

