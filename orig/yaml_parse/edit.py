#coding:utf-8
import yaml
import random

random.seed(0)

newline_indicator = u'\u257c'  # 改行文字の置換
period_indicator = u'\u2596'   # 文中のピリオドを置換
maru_indicator = u'\u259f'     # 文中の読点を置換

train_en = open("train.en", "w")
dev_en = open("dev.en", "w")
test_en = open("test.en", "w")
train_ja = open("train.ja", "w")
dev_ja = open("dev.ja", "w")
test_ja = open("test.ja", "w")

fp = "/opt/data/translation/rails_tutorial/translation_memory.yml"

data_dict = {}
with open(fp) as f:
    data = yaml.load(f)

for num, d in enumerate(data):
    tmp = []
    jap = d[":ja"].replace("\n", newline_indicator)
    jap = jap.replace(u"。", maru_indicator)
    jap = jap.replace(".", period_indicator)
    if jap.endswith(maru_indicator):
        jap = jap[:-1] + u"。"

    eng = d[":en"].replace("\n", newline_indicator)
    eng = eng.replace(".", period_indicator)
    eng = eng.replace(u"。", maru_indicator)
    if eng.endswith(period_indicator):
        eng = eng[:-1] + "."

    tmp.append(jap)
    tmp.append(eng)

    data_dict[num] = tmp

keys = data_dict.keys()
random.shuffle(keys)

for key in keys:
    if key < 300:  #train
        train_ja.write(data_dict[key][0].encode("utf-8") + "\n")
        train_en.write(data_dict[key][1].encode("utf-8") + "\n")
    elif key < 600:  # dev
        dev_ja.write(data_dict[key][0].encode("utf-8") + "\n")
        dev_en.write(data_dict[key][1].encode("utf-8") + "\n")
    else:  # test
        test_ja.write(data_dict[key][0].encode("utf-8") + "\n")
        test_en.write(data_dict[key][1].encode("utf-8") + "\n")

train_ja.close()
train_en.close()
dev_ja.close()
dev_en.close()
test_ja.close()
test_en.close()
