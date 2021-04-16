# -*- coding: UTF-8 -*-
# @Time    : 2021/3/23
# @Author  : xiangyuejia@qq.com
# Apache License
# Copyright©2020-2021 xiangyuejia@qq.com All Rights Reserved
import nltk
from aitool.data_structure.graph.chain_forward_stars import reform_data, ChainForwardStars
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def load_stop_word():
    with open('stopword.txt', 'r', encoding='utf8') as fin:
        stop_word = [w.strip() for w in fin.readlines()]
    return set(stop_word)


class ERGraph:
    def __init__(self, data, cand=20, vis=4):
        self.num_candidate_evi = cand
        self.visited_node_length_max = vis
        self.stop_word = load_stop_word()
        self.que_words = None
        self.ans_words = None
        self.que_words_unique_set = None
        self.ans_words_unique_set = None
        self.que_words_unique_node_set = set()
        self.ans_words_unique_node_set = set()
        self.graph = ChainForwardStars()
        self.eid2rel = {}
        self.eid2rank = {}
        self.paths = []
        self.get_graph(data)
        self.get_key_words(data)

    def extract_core_words(self, text):
        words = nltk.word_tokenize(text)
        words = map(stemmer.stem, words)
        words = [w for w in words if w not in self.stop_word]
        # print(list(words))
        return words

    def get_graph(self, data):
        evi = data['evi'][:self.num_candidate_evi]
        for rank, (eid, rel, e) in enumerate(evi):
            self.eid2rank[eid] = rank
            self.eid2rel[eid] = rel
            e_words = self.extract_core_words(e)
            e_words_len = len(e_words)
            edges = []
            for i in range(e_words_len):
                for j in range(e_words_len):
                    if i != j:
                        edges.append([e_words[i], eid, e_words[j]])
            self.graph.built(reform_data(edges))

    def get_key_words(self, data):
        que = data['que']
        ans = data['ans']
        self.que_words = self.extract_core_words(que)
        self.ans_words = self.extract_core_words(ans)
        self.que_words_unique_set = set(self.que_words) - set(self.ans_words)
        self.ans_words_unique_set = set(self.ans_words) - set(self.que_words)
        que_clean = []
        ans_clean = []
        for word in self.que_words_unique_set:
            if word in self.graph.node_name2index:
                que_clean.append(word)
                self.que_words_unique_node_set.add(self.graph.node_name2index[word][0])
        for word in self.ans_words_unique_set:
            if word in self.graph.node_name2index:
                ans_clean.append(word)
                self.ans_words_unique_node_set.add(self.graph.node_name2index[word][0])
        # print('@@@')
        # print(self.que_words)
        # print(self.ans_words)
        # print(que_clean)
        # print(ans_clean)
        # print('###')

    def deep_search_unit(self, node, visited_node, visited_eid, visited_word):
        if node in self.que_words_unique_node_set:
            self.paths.append([visited_node, visited_eid, visited_word])
        if len(self.paths) > 900000:
            return
        if len(visited_node) > self.visited_node_length_max:
            return
        p = self.graph.heads[node]
        while p != -1:
            edge = self.graph.edges[p]
            if edge.end not in visited_node \
                    and edge.end not in self.ans_words_unique_node_set \
                    and edge.name not in visited_eid:
                self.deep_search_unit(
                    edge.end,
                    visited_node + [edge.end],
                    visited_eid + [edge.name],
                    visited_word + [self.graph.index2node_name[edge.end]],
                )
            p = self.graph.edges[p].pre

    def statistics_by_paths(self):
        ansW2queW = {}
        for node in self.ans_words_unique_node_set:
            ansW = self.graph.index2node_name[node]
            ansW2queW[ansW] = {'#path#': 0}
            self.paths = []
            self.deep_search_unit(node, [node], ['#'], [ansW])
            print('len of path', len(self.paths))
            for _, es, ws in self.paths:
                ansW2queW[ansW]['#path#'] += 1
                queW = ws[-1]
                if queW not in ansW2queW[ansW]:
                    ansW2queW[ansW][queW] = {'total': 0}
                ansW2queW[ansW][queW]['total'] += 1

                for e in es[1:]:
                    if e not in ansW2queW[ansW][queW]:
                        ansW2queW[ansW][queW][e] = 0
                    else:
                        ansW2queW[ansW][queW][e] += 1
        self.ansW2queW = ansW2queW

    def get_rerank_score(self):
        analysis = {}
        ansW_score = 100
        for ansW in self.ansW2queW.keys():
            total_time = self.ansW2queW[ansW]['#path#']
            for queW in self.ansW2queW[ansW].keys():
                if queW == '#path#':
                    continue
                part_time = self.ansW2queW[ansW][queW]['total']
                for e in self.ansW2queW[ansW][queW].keys():
                    if e == 'total':
                        continue
                    if e not in analysis:
                        analysis[e] = 0
                    analysis[e] += ansW_score * (part_time/total_time) * (self.ansW2queW[ansW][queW][e]/part_time)
        all_sum = 0
        for k,v in analysis.items():
            all_sum += v
        if all_sum > 1:
            for k, v in analysis.items():
                analysis[k] = v/all_sum * 100
        return analysis





