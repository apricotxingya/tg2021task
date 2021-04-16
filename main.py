# -*- coding: UTF-8 -*-
# @Time    : 2021/3/23
# @Author  : xiangyuejia@qq.com
# Apache License
# Copyright©2020-2021 xiangyuejia@qq.com All Rights Reserved
from tqdm import tqdm
from data_process import load_data, reformat_data, show_analysis, reranking, generate_output, print_output_data, get_rerank_score_other
from feature_process import ERGraph

# base
# input_file = 'predict_eval_data/chain_eval_predict.xlsx'
input_file = 'predict_chain/chain_predict_200c.xlsx'
output_file_base = 'predict_chain/predict'
data = load_data(input_file)
desk, qid_list = reformat_data(data)

# other
input_file_other = 'other_data/predict-ECI.xlsx'
data_other = load_data(input_file_other)
desk_other, qid_list_other = reformat_data(data_other)

out_list = []
cand = 10
vis = 8
weight_mode = '1'
coefficient = 1.0
other_mode = '2'
for qid in tqdm(qid_list):
    data = desk[qid]

    # basic
    model = ERGraph(data, cand=cand, vis=vis)
    model.statistics_by_paths()
    analysis = model.get_rerank_score()
    # show_analysis(data, analysis)
    new_rank = reranking(data, analysis, weight_mode=weight_mode, coefficient=coefficient)
    output_file = output_file_base + '_cand{}_vis{}_wm{}_coe{}.txt'.format(cand, vis, weight_mode, coefficient)

    # other
    # analysis_other = get_rerank_score_other(desk_other[qid], mode=other_mode)
    # show_analysis(data, analysis_other)
    # new_rank = reranking(data, analysis_other, weight_mode=weight_mode, coefficient=coefficient)
    # output_file = output_file_base + '_wm{}_coe{}_om{}.txt'.format(weight_mode, coefficient, other_mode)

    out_list += generate_output(qid, new_rank)
print_output_data(out_list, output_file)
