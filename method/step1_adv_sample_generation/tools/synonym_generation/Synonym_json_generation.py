import codecs
import json
from collections import defaultdict

def get_kw2similar_words(fin, fout):
    kw2similar_words = defaultdict(set)  # 去重
    with codecs.open(fin, "r", "GB18030") as fr:
        for idx, line in enumerate(fr):
            try:
                row = line.strip().split(" ")
                print(row)
                if row[0][-1] == u"@": continue
                for kw in row[1:]:
                    row.remove(kw)
                    kw_and_type = kw + row[0][-1]
                    kw2similar_words[kw_and_type].update(row[1:])
                    row.insert(-1, kw)
            except Exception as error:
                print("Error line", idx, line, error)
            if idx % 1000 == 0: print(idx)

    for kw, similar_words in kw2similar_words.items():
        kw2similar_words[kw] = list(similar_words)  # kw2similar_words = defaultdict(list)
    json.dump(kw2similar_words, open(fout, "w"), ensure_ascii=False)


if __name__ == '__main__':
    pass