import codecs
import json
from collections import defaultdict

def find_synonyms(kw):
    kw2similar_words = json.load(open("/media/usr/external/home/usr/project/project3_data/tools/similar_words.json", "r"))
    keys_set = set(kw2similar_words.keys())
    tongyici = []
    for kw_and_type in [kw + u"=", kw + u"#"]:
        if kw_and_type in keys_set:
            # print(kw_and_type, "/".join(kw2similar_words[kw_and_type]))
            return kw2similar_words[kw_and_type]
    return tongyici

if __name__ == '__main__':
    pass