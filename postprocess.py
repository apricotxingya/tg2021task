# -*- coding: UTF-8 -*-
# @Time    : 2021/3/18
# @Author  : xiangyuejia@qq.com
# Apache License
# Copyright©2020-2021 xiangyuejia@qq.com All Rights Reserved
import os
import json
import pandas as pd
from tqdm import tqdm


def read_data_file(data_file, has_ans):
    upper = {}
    id2q = {}
    id2a = {}
    id2e = {}
    e2id = {}
    qid2eid_relevance = {}
    qid2eid_isGoldWT21 = {}
    test_jason_file = data_file
    with open(test_jason_file) as fin:
        data = json.load(fin)
    data = data['rankingProblems']
    for qa in data:
        qid = qa['qid']
        qtext = qa['questionText']
        atext = qa['answerText']
        id2q[qid] = qtext
        id2a[qid] = atext
        qid_l = qid.lower()
        upper[qid_l] = qid
        if has_ans:
            documents = qa['documents']
            for ev in documents:
                eid = ev['uuid']
                qid2eid_relevance[(qid, eid)] = ev['relevance']
                qid2eid_isGoldWT21[(qid, eid)] = ev['isGoldWT21']

    table_index_file = 'data/tableindex.txt'
    with open(table_index_file) as fin:
        table_index = [l.strip() for l in fin.readlines()]
    table_dir_path = 'data/tables'
    for table in table_index:
        table_file = os.path.join(table_dir_path, table)
        with open(table_file) as fin:
            datas = [l.strip().split('\t') for l in fin.readlines()]
            for data in datas[1:]:
                if data[-2] != '':
                    continue
                nt = []
                for k in data[:-3]:
                    if k != '':
                        nt.append(k)
                etext = ' '.join(nt)
                eid = data[-1]
                id2e[eid] = etext
                if etext not in e2id:
                    e2id[etext] = [[eid, table]]
                else:
                    e2id[etext].append([eid, table])
    return id2q, id2a, id2e, upper, qid2eid_relevance, qid2eid_isGoldWT21


def generate_human_read_file(
        predict_file,
        human_read_file,
        count_limit=100,
        out_txt_file='new.txt',
        data_file='data/wt-expert-ratings.test.json',
        has_ans=False,
):
    datas = []
    qids = set()
    id2q, id2a, id2e, upper, qid2eid_relevance, qid2eid_isGoldWT21 = read_data_file(data_file, has_ans)
    with open(predict_file, 'r', encoding='utf8') as fin:
        line = fin.readline()
        line_count = 0
        while line is not None and line != '':
            line_count += 1
            if line_count % 1000 == 0:
                pass
                # print('line_count ', line_count)
            qid, eid = line.strip().split('\t')
            if qid in upper:
                qid = upper[qid]
            if qid not in qids:
                count = 0
                qids.add(qid)
            if count >= count_limit:
                line = fin.readline()
                continue
            if qid not in id2q:
                print('not in id2q',qid)
                count = 100000
                line = fin.readline()
                continue
            if eid not in id2e:
                print('not in id2e',eid)
                line = fin.readline()
                continue
            question = id2q[qid]
            ans = id2a[qid]
            evidence = id2e[eid]
            if has_ans:
                relevance = qid2eid_relevance[(qid, eid)] if (qid, eid) in qid2eid_relevance else -1
                isGoldWT21 = qid2eid_isGoldWT21[(qid, eid)] if (qid, eid) in qid2eid_isGoldWT21 else -1
                datas.append([qid, eid, question, ans, relevance, isGoldWT21, evidence])
            else:
                datas.append([qid, eid, question, ans, -1, -1, evidence])
            count += 1
            line = fin.readline()
    print('len of qids ', len(qids))
    df = pd.DataFrame(datas)
    df.to_excel(human_read_file, index=False, header=False)
    with open(out_txt_file, 'w') as fout:
        for data in datas:
            qid = data[0]
            eid = data[1]
            print('{}\t{}'.format(qid, eid), file=fout)
    with open('tmp.txt', 'w') as fout:
        for qid in qids:
            print('', file=fout)
            print(qid, file=fout)
            print(id2q[qid], file=fout)
            print(id2a[qid], file=fout)
            for qid2eid in qid2eid_isGoldWT21.keys():
                if qid2eid[0] == qid and qid2eid_isGoldWT21[qid2eid]=='1':
                    print(id2e[qid2eid[1]], file=fout)


if __name__ == '__main__':
    # in_file = 'predict_base/predict.txt'
    # out_file = 'predict_base/predict.xlsx'
    # generate_human_read_file(in_file, out_file)

    # in_file = 'predict_chain/eval_predictions.txt'
    # out_file = 'predict_chain/chain_predict_100c.xlsx'
    # out_txt_file = 'predict_chain/predict_new_100c.txt'
    # generate_human_read_file(
    #     in_file,
    #     out_file,
    #     out_txt_file=out_txt_file,
    #     count_limit=100
    # )

    # in_file = 'predict_eval_data/eval_predictions.txt'
    # out_file = 'predict_eval_data/chain_eval_predict_100c.xlsx'
    # data_file = 'data/wt-expert-ratings.dev.json'
    # generate_human_read_file(in_file, out_file, data_file=data_file, has_ans=True)

    in_file = 'predict_train_data/eval_predictions.txt'
    out_file = 'predict_train_data/chain_train_predict.xlsx'
    data_file = 'data/wt-expert-ratings.train.json'
    generate_human_read_file(
        in_file,
        out_file,
        data_file=data_file,
        has_ans=True,
        count_limit=200
    )
