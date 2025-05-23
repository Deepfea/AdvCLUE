import numpy as np
import os
import pandas as pd
import jieba
import re

def seg_data(data_df):
    seg_list = []
    for data_num in range(len(data_df)):
        # print(data_df.loc[data_num, 'answer'])
        # print(data_df.loc[data_num, 'text'])
        results = re.split(data_df.loc[data_num, 'answer'], data_df.loc[data_num, 'text'])
        # print(results)
        cut_result = []
        for cut_num in range(len(results)):
            if len(results[cut_num]) == 0:
                continue
            temp_string = results[cut_num]
            temp_result = jieba.cut(temp_string, cut_all=False)
            temp_result = list(temp_result)
            cut_result.extend(temp_result)

        segs = list(set(cut_result))
        seg_list.append(segs)

    seg_arr = np.array(seg_list)
    return seg_arr


if __name__ == '__main__':
    pass
