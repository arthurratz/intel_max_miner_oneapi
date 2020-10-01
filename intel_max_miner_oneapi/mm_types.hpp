//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <string>
#include <fstream>
#include <iostream>

typedef struct {
    char* m_buf;
    std::size_t m_size;
} MMN_ITEM;

typedef struct {
    MMN_ITEM* m_v;
    std::size_t m_items;
} MMN_TRANS;

typedef struct {
    MMN_ITEM* m_v;
    double m_conf;
    double m_supp_a;
    double m_supp_b;
    double m_supp_ab;
    std::size_t m_items;
} MMN_RULE;

typedef struct {
    std::size_t m_trans_cnt;
    std::size_t m_items_cnt;
    std::size_t m_item_max_len;
    std::size_t m_trans_min_len;
    std::size_t m_trans_max_len;
} MMN_TRANS_STATS;

typedef struct {
    MMN_ITEM* m_items;
    MMN_TRANS* m_trans;
    MMN_TRANS_STATS m_stats;
} MMN_TRANS_CONTEXT;
