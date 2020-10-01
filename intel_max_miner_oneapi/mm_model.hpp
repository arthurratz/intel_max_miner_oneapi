//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <string.h>

#include <tbb/tbb.h>
#include <tbb/parallel_pipeline.h>

#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "mm_types.hpp"
#include "usm_string.hpp"
#include "mm_vector.hpp"

using namespace std;

class mm_model
{
public:
    mm_model(USM_ALLOC_TYPE alloc_type = USM_ALLOC_TYPE::usm_alloc_crt) : \
            m_alloc_type(alloc_type) {};
    virtual ~mm_model() {}

public:
    void load_trans_from_file(const char* filename, \
        MMN_TRANS_CONTEXT*& mmn_trans_ctx)
    {
        std::ifstream ifs;
        std::size_t items = 0L;
        std::size_t trans = 0L;
        std::size_t item_max_len = 0L;
        std::size_t trans_min_len = 0L;
        std::size_t trans_max_len = 0L;

        MMN_ITEM* uitems_buf = nullptr;
        MMN_TRANS* trans_buf = nullptr;
        sycl_usm_alloc_helper usm_alloc( \
            "mm_model buffers", m_alloc_type);

        ifs.exceptions(std::ifstream::badbit);

        try
        {
            ifs.open(filename, std::ios::in);
            auto mp = tbb::global_control::max_allowed_parallelism;
            tbb::global_control gc(mp, g_threads);

            tbb::task_group task_group;
            task_group.run_and_wait([&] {
                tbb::parallel_pipeline(ULLONG_MAX,
                    tbb::make_filter<void, char*>(
                        tbb::filter_mode::serial_in_order,
                        [&](tbb::flow_control& fc)-> char* {
                            char* lbuf = nullptr;
                            string line_buf = "\0";
                            if (ifs.eof()) { fc.stop(); return lbuf; }
                            else if (getline(ifs, line_buf)) {
                                usm_alloc.alloc_buffer<char>(lbuf, strlen(line_buf.c_str()) + 1);
                                usm_string_helper::strcpy(lbuf, strlen(line_buf.c_str()) + 1, \
                                    const_cast<char*>(line_buf.c_str()), strlen(line_buf.c_str()));
                            }

                            return lbuf;
                        }) &
                    tbb::make_filter<char*, MMN_TRANS*>(
                        tbb::filter_mode::serial_in_order,
                        [&](char* lbuf)-> MMN_TRANS* {
                            std::size_t tti = 0L;
                            static char delim[] = ",";
                            MMN_TRANS* tt_buf = nullptr;
                            char* token = nullptr, *next_token = token;
                            usm_alloc.alloc_buffer<MMN_TRANS>(tt_buf, 1);

                            MMN_ITEM* items_buf = nullptr;
                            for (token = strtok_s(lbuf, delim, &next_token); \
                                token != nullptr; token = strtok_s(nullptr, delim, &next_token))
                            {
                                usm_alloc.realloc_buf_async<MMN_ITEM>(items_buf, (tti + 1));

                                items_buf[tti].m_buf  = nullptr;
                                items_buf[tti].m_size = strlen(token);

                                usm_alloc.alloc_buffer<char>(\
                                    items_buf[tti].m_buf, items_buf[tti].m_size + 1);
                                usm_string_helper::strcpy(items_buf[tti].m_buf, \
                                    items_buf[tti].m_size + 1, token, items_buf[tti].m_size);

                                if (!mm_vector::exists(uitems_buf, items, token))
                                {
                                    usm_alloc.realloc_buf_async<MMN_ITEM>(uitems_buf, (items + 1));

                                    uitems_buf[items].m_buf = nullptr;
                                    uitems_buf[items].m_size = strlen(token);

                                    usm_alloc.alloc_buffer<char>(\
                                        uitems_buf[items].m_buf, uitems_buf[items].m_size + 1);
                                    usm_string_helper::strcpy(uitems_buf[items].m_buf, \
                                        uitems_buf[items].m_size + 1, token, strlen(token));

                                    items++;
                                }

                                if ((strlen(token) > item_max_len) || (item_max_len == 0))
                                    item_max_len = strlen(token);

                                tti++;
                            }

                            if ((tti > 0L) && (items_buf != nullptr)) {
                                tt_buf->m_items = tti;
                                tt_buf->m_v = items_buf;
                            }

                            return tt_buf;
                        }) &
                    tbb::make_filter<MMN_TRANS*, void>(
                        tbb::filter_mode::serial_in_order,
                        [&](MMN_TRANS* tt_buf)-> void {

                            usm_alloc.realloc_buf_async<MMN_TRANS>(trans_buf, (trans + 1));

                            if (tt_buf != nullptr) {
                                trans_buf[trans++] = *tt_buf;

                                if ((tt_buf->m_items < trans_min_len) ||
                                    (trans_min_len == 0L)) {
                                    trans_min_len = tt_buf->m_items;
                                }

                                if ((tt_buf->m_items > trans_max_len) ||
                                    (trans_max_len == 0L)) {
                                    trans_max_len = tt_buf->m_items;
                                }
                            }
                        }));
            });
        }

        catch (std::ifstream::failure e) {
            std::cerr << "Unable to read file: " << filename << "\n";
            exit(1);
        }

        if (mmn_trans_ctx == nullptr)
        {
            MMN_TRANS_STATS stats;
            stats.m_trans_cnt = trans;
            stats.m_items_cnt = items;
            stats.m_item_max_len  = item_max_len;
            stats.m_trans_min_len = trans_min_len;
            stats.m_trans_max_len = trans_max_len;

            if ((uitems_buf != nullptr) && (trans_buf != nullptr)) {
               usm_alloc.alloc_trans_ctx(mmn_trans_ctx, uitems_buf, trans_buf, stats);
            }
        }
    }

private:
    USM_ALLOC_TYPE	m_alloc_type;
};