# -*- coding: UTF-8 -*-
# @Time    : 2021/3/23
# @Author  : xiangyuejia@qq.com
# Apache License
# Copyright©2020-2021 xiangyuejia@qq.com All Rights Reserved
import pandas as pd


def load_data(file):
    df = pd.read_excel(file, header=None)
    data = df.values
    return data


def reformat_data(data):
    desk = {}
    qid_list = []
    for d in data:
        qid = d[0]
        eid = d[1]
        que = d[2]
        ans = d[3]
        rel = d[4]
        evi = d[6]
        if qid not in desk:
            qid_list.append(qid)
            desk[qid] = {'que': que, 'ans': ans, 'evi': []}
        desk[qid]['evi'].append([eid, rel, evi])
    return desk, qid_list


def show_analysis(data, analysis):
    print('\n####')
    print('que:', data['que'])
    print('ans:', data['ans'])
    print('@@@@')
    for eid, rel, text in data['evi']:
        score = 0
        if eid in analysis:
            score = analysis[eid]
        print('{}\t{}\t{}\t{}'.format(eid, rel, score, text))


def reranking(data, analysis, weight_mode='1', coefficient=1.0):
    ori_rank = {}
    if weight_mode == '1':
        weight = [
            100, 95, 90, 85, 80,
            75, 75, 75, 75, 75,
            60, 60, 60, 60, 60,
            40, 40, 40, 40, 40,
        ]
    length = len(data['evi'])
    for i in range(length-len(weight)):
        weight.append(0.1/(i+1))
    for (eid, rel, text), w in zip(data['evi'], weight):
        ori_rank[eid] = w
    for eid in ori_rank.keys():
        score = 0
        if eid in analysis:
            score = analysis[eid] * coefficient
        ori_rank[eid] += score
    return ori_rank


def generate_output(qid, rank):
    rank_list = [[k, v] for k,v in rank.items()]
    rank_list.sort(key=lambda x: x[1], reverse=True)
    out_list = []
    for k,v in rank_list:
        out_list.append([qid, k])
    return out_list


def print_output_data(out_list, output_file):
    print('writting to ', output_file)
    with open(output_file, 'w') as fout:
        for qid, eid in out_list:
            print('{}\t{}'.format(qid, eid), file=fout)


def get_rerank_score_other(data_new, mode='1'):
    if mode == '1':
        score={2:11, 1:6, 0:0, -1:-6, -2:-11}
    if mode == '2':
        score={2:6, 1:3, 0:0, -1:-3, -2:-6}
    analysis = {}
    for eid, rel, text in data_new['evi']:
        analysis[eid] = score[rel]
    return analysis
