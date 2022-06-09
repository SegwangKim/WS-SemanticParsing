# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for identifying identical substrings in sources and targets."""

from grammar_induction import qcfg_rule
from itertools import product
from collections import defaultdict
import numpy as np
import re

def _in_matched_range(start_idx, end_idx, matched_ranges):

    """Return True if provided indices overlap any spans in matched_ranges."""
    for range_start_idx, range_end_idx in matched_ranges:
        if not (end_idx <= range_start_idx or start_idx >= range_end_idx):
            return True
    return False


def _find_exact_matches(source, target):
    """Returns longest non-overlapping sub-strings shared by source and target."""
    source_len = len(source)
    target_len = len(target)

    matches = set()
    matched_source_ranges = set()
    matched_target_ranges = set()
    for sequence_len in range(max(target_len, source_len), 0, -1):
        for source_start_idx in range(0, source_len - sequence_len + 1):
            source_end_idx = source_start_idx + sequence_len
            if _in_matched_range(source_start_idx, source_end_idx,
                                 matched_source_ranges):
                continue
            for target_start_idx in range(0, target_len - sequence_len + 1):
                target_end_idx = target_start_idx + sequence_len
                if _in_matched_range(target_start_idx, target_end_idx,
                                     matched_target_ranges):
                    continue

                source_span = source[source_start_idx:source_end_idx]
                target_span = target[target_start_idx:target_end_idx]

                if source_span == target_span and len(re.findall("\(|\)|\.|\,", " ".join(source_span))) == 0:
                    matches.add(tuple(source_span))
                    matched_source_ranges.add((source_start_idx, source_end_idx))
                    matched_target_ranges.add((target_start_idx, target_end_idx))

    return matches


def get_exact_match_rules(dataset):
    """Return set of rules for terminal sequences in both source and target."""

    exact_matches = set()
    for source_str, target_str in dataset:
        source = source_str.split()
        target = target_str.split()
        exact_matches.update(_find_exact_matches(source, target))

    exact_match_rules = set()
    for match in exact_matches:
        rule = qcfg_rule.QCFGRule(source=tuple(match), target=tuple(match), arity=0)
        exact_match_rules.add(rule)

    matches_ = _find_approx_clause_matches(dataset, exclude=exact_matches)
    for match in matches_:
        rule = qcfg_rule.QCFGRule(source=tuple(match[0].split()), target=tuple(match[1].split()), arity=0)
        exact_match_rules.add(rule)

    return exact_match_rules



####################
####################
####################
####################
####################
def _get_deepest_clauses(trg):
    level = 0
    temp = defaultdict(list, [])
    for token in trg.split():
        if token == "(":
            level += 1
            continue
        if token == ")":
            temp[level].append("###")
            level -= 1
            continue
        temp[level].append(token)
    top_level = max(temp.keys())
    clauses = set()
    if top_level > 0:
        rules = " ".join(temp[top_level])
        clauses |= set([i.strip() for i in rules.split("###") if len(i.strip()) > 0])
    return clauses


def _build_vocab(examples):
    x_vocab, y_vocab = set(), set()
    for ex in examples:
        x_words, y_words = set(ex[0].split()), _get_deepest_clauses(ex[1])
        x_vocab = x_vocab | x_words
        y_vocab = y_vocab | y_words

    temp = list(y_vocab)
    for idx, w1 in enumerate(temp):
        for w in temp[idx+1:]:
            if w in w1:
                y_vocab -= set([w1])
            elif w1 in w:
                y_vocab -= set([w])

    xy_pairs = []
    for ex in examples:
        x_words = set(ex[0].split())
        y_words = set(clause for clause in y_vocab if clause in ex[1])
        xy_pairs.append((x_words, y_words))
    return xy_pairs, x_vocab, y_vocab


def _compute_joint_prob(xy_pairs, x_vocab, y_vocab, sample_size=0):
    p_xy = defaultdict(dict)
    if sample_size:
        xy_pairs = xy_pairs[:sample_size]
    for x_i, y_j in product(x_vocab, y_vocab):
        for u, v in product([0, 1], [0, 1]):
            temp = [1 for src, trg in xy_pairs if (x_i in src) == u and (y_j in trg) == v]
            p_xy[(x_i, y_j)][(u, v)] = len(temp) / len(xy_pairs)
    return p_xy


def _calculate_mi(v):
    mi = 0
    if v[(0, 0)] > 0:
        mi += v[(0, 0)] * np.log(v[(0, 0)] / ((v[(0, 0)] + v[(0, 1)]) * (v[(0, 0)] + v[(1, 0)])))
    if v[(0, 1)] > 0:
        mi += v[(0, 1)] * np.log(v[(0, 1)] / ((v[(0, 0)] + v[(0, 1)]) * (v[(0, 1)] + v[(1, 1)])))
    if v[(1, 0)] > 0:
        mi += v[(1, 0)] * np.log(v[(1, 0)] / ((v[(1, 0)] + v[(1, 1)]) * (v[(0, 0)] + v[(1, 0)])))
    if v[(1, 1)] > 0:
        mi += v[(1, 1)] * np.log(v[(1, 1)] / ((v[(1, 0)] + v[(1, 1)]) * (v[(0, 1)] + v[(1, 1)])))
    return mi


def _greedy_matching(mi, y_vocab, chosen_src_vocab=[], threshold=0.1):
    sorted_mi = sorted([(k, v) for k, v in mi.items()], key=lambda x: -x[-1])
    ans = {}
    for pair, score in sorted_mi:
        src, trg = pair
        if len(ans.keys()) == len(y_vocab):  # finish
            break
        if trg in ans.keys() or src in chosen_src_vocab:  # already mapped
            continue
        if score >= threshold:
            ans[trg] = src
            chosen_src_vocab.append(src)
    return set(tuple((src, trg)) for trg, src in ans.items())


def _find_approx_clause_matches(examples, exclude, sample_size=0, matching_threshold=0.01):
    xy_pairs, x_vocab, y_vocab = _build_vocab(examples)
    p_xy = _compute_joint_prob(xy_pairs, x_vocab, y_vocab, sample_size=sample_size)
    mi = dict((k, _calculate_mi(v)) for k, v in p_xy.items())

    reserved = [i[0] for i in list(exclude)]
    approx_clause_matches = _greedy_matching(mi, y_vocab, reserved, matching_threshold)

    return approx_clause_matches



