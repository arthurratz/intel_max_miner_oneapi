//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "tbb/task_group.h"

#include <CL/sycl.hpp>

#include <iomanip>
#include <iostream>
#include <algorithm>

#include "main.hpp"

using namespace cl::sycl;

int main(int argc, char** argv)
{
	// C:\Users\arthu\Desktop\intel_omp_max_miner_exe\datasets\micro200.csv

	static char filename[266] = "\0";

	std::cout << "Parallel Max-Miner Algorithm by Arthur V. Ratz @ Intel DevMesh\n\n";

	std::cout << "Enter filename: "; std::cin >> filename;

	std::size_t rules_size = 0L;
	MMN_RULE* rules_buf = nullptr;
	MMN_TRANS_CONTEXT* trans_ctx = nullptr;
	mm_model model(USM_ALLOC_TYPE::usm_alloc_crt);
	model.load_trans_from_file(filename, trans_ctx);

	std::cout << "\nTransactions:\t" << trans_ctx->m_stats.m_trans_cnt;
	std::cout << "\nTransaction Len(Min):\t" << trans_ctx->m_stats.m_trans_min_len;
	std::cout << "\nTransaction Len(Max):\t" << trans_ctx->m_stats.m_trans_max_len;
	std::cout << "\n\nItems:\t" << trans_ctx->m_stats.m_items_cnt;
	std::cout << "\nItem Len(Max):\t" << trans_ctx->m_stats.m_item_max_len << "\n\n";

	sycl_usm_alloc_helper usm_alloc("");

	parallel_max_miner::compute(trans_ctx, rules_buf, rules_size);

	std::cout << "\n===========================================================\n";
	std::cout << "Results:";
	std::cout << "\n===========================================================\n\n";

	parallel_max_miner::print_rules(rules_buf, rules_size);

	std::cin.get();

	return 0;
}