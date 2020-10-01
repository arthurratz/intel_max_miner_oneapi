//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include "mm_sort.hpp"
#include "mm_vector.hpp"
#include "usm_alloc.hpp"
#include "mm_model.hpp"

typedef std::pair<double, double> minmax_conf_type;
typedef std::pair<std::size_t, std::size_t> trans_range_type;

namespace parallel_max_miner
{
	void print_rules(const MMN_RULE* rules_buf, const std::size_t rules)
	{
		for (std::size_t i = 0; i < rules; i++)
			std::cout << (i + 1) << " --> " << mm_vector::to_string(rules_buf[i].m_v, \
				rules_buf[i].m_items) << "[ size = " << rules_buf[i].m_items << " conf = " << rules_buf[i].m_conf << " ]\n";
	}

	double get_mean_conf(const minmax_conf_type minmax_conf) {
		return (minmax_conf.second - minmax_conf.first) * .5f;
	}

	minmax_conf_type get_minmax_conf(\
		const MMN_RULE* cnds_buf, const std::size_t cnds)
	{
		double conf_min = .0f, conf_max = conf_min;

		auto min_supp_it = std::min_element(cnds_buf, cnds_buf + cnds, \
			[&](const MMN_RULE& rule1, const MMN_RULE& rule2) {
				return (rule1.m_supp_ab < rule2.m_supp_ab);
			});

		conf_min = min_supp_it->m_supp_ab;

		auto max_supp_it = std::max_element(cnds_buf, cnds_buf + cnds, \
			[&](const MMN_RULE& rule1, const MMN_RULE& rule2) {
				return (rule1.m_supp_ab < rule2.m_supp_ab);
			});

		conf_max = max_supp_it->m_supp_ab;


		return std::make_pair<double, double>(double(conf_min), double(conf_max));
	}

	double get_support(const MMN_RULE& rule, \
		MMN_TRANS_CONTEXT* trans_ctx, trans_range_type trans_range)
	{
		double count = 0L;
		for (std::size_t i = trans_range.first; i < trans_range.second; i++)
			count += (mm_vector::intersect_vec(rule.m_v, rule.m_items, \
				trans_ctx->m_trans[i].m_v, trans_ctx->m_trans[i].m_items) == rule.m_items);

		return count;
	}

	template<class _Pred>
	void filter_cands(MMN_RULE* cnds_buf, \
		MMN_RULE*& cnds_new_buf, std::size_t& cnds_size, _Pred pred)
	{
		sycl_usm_alloc_helper \
			usm_alloc("mm_cnds buffers");

		std::size_t cnds_new = 0L;
		MMN_RULE* cnds_buf_new = nullptr;
		for (std::size_t ii = 0; ii < cnds_size; ii++)
			if (pred(cnds_buf[ii]))
			{
				usm_alloc.realloc_buf_async<MMN_RULE>(\
					cnds_buf_new, (cnds_new + 1));

				cnds_buf_new[cnds_new++] = cnds_buf[ii];
			}

			else {
				usm_alloc.free_items_buf(\
					cnds_buf[ii].m_v, cnds_buf[ii].m_items);
			}

		cnds_size = cnds_new;
		cnds_new_buf = cnds_buf_new;
	}

	template<class _Pred>
	void filter_cands(MMN_RULE*& cnds_buf, std::size_t& cnds_size, _Pred pred) {
		MMN_RULE* cnds_buf_new = nullptr;
		filter_cands(cnds_buf, cnds_buf_new, cnds_size, pred);
		cnds_buf = cnds_buf_new;
	}

	cl::sycl::event init_model(MMN_RULE*& cnds_buf, \
		MMN_TRANS_CONTEXT* trans_ctx, std::size_t& cnds, \
			minmax_conf_type& minmax_conf)
	{
		cl::sycl::event event;

		sycl_usm_alloc_helper usm_alloc(\
			"mm_candidate buffers");

		usm_alloc.alloc_cnds_buf(cnds_buf, cnds, 1L, \
			trans_ctx->m_stats.m_item_max_len);

		tbb::parallel_for(tbb::blocked_range<std::size_t>(0, cnds), \
			[&](const tbb::blocked_range<std::size_t>& r) {
				for (std::size_t ii = r.begin(); ii != r.end(); ii++)
				{
					cnds_buf[ii].m_items = 1L;
					cnds_buf[ii].m_supp_a = trans_ctx->m_stats.m_trans_cnt;
					cnds_buf[ii].m_supp_b = .0f;

					usm_string_helper::strcpy(cnds_buf[ii].m_v[0].m_buf, trans_ctx->m_stats.m_item_max_len, \
						trans_ctx->m_items[ii].m_buf, trans_ctx->m_items[ii].m_size);

					trans_range_type trans_range = \
						std::make_pair(0, trans_ctx->m_stats.m_trans_cnt);

					cnds_buf[ii].m_supp_ab = get_support(cnds_buf[ii], trans_ctx, trans_range);
					cnds_buf[ii].m_conf = cnds_buf[ii].m_supp_ab / cnds_buf[ii].m_supp_a;
				}
			});

		filter_cands(cnds_buf, cnds, [&](const MMN_RULE& rule) {
			return (rule.m_supp_ab > 1L) && (rule.m_supp_ab < trans_ctx->m_stats.m_trans_cnt); });

		if ((minmax_conf.first == .0f) && (minmax_conf.second == .0f))
			minmax_conf = get_minmax_conf(cnds_buf, cnds);

		return event;
	}

	cl::sycl::event gen_sub_nodes(MMN_RULE*& cnds_buf, \
		MMN_TRANS_CONTEXT* trans_ctx, std::size_t& cnds, \
		minmax_conf_type minmax_conf, trans_range_type trans_range, \
			cl::sycl::event event, cl::sycl::queue device_queue)
	{
		cl::sycl::event event1;

		std::size_t cnds_new = \
			(std::size_t)std::pow((double)cnds, 2.0);

		sycl_usm_alloc_helper usm_alloc(\
			"mm_candidate buffers", USM_ALLOC_TYPE::usm_alloc_crt);

		MMN_RULE* cnds_buf1 = nullptr;
		usm_alloc.alloc_cnds_buf(cnds_buf1, cnds_new, \
			trans_ctx->m_stats.m_trans_max_len, \
			trans_ctx->m_stats.m_item_max_len);

		cl::sycl::buffer<trans_range_type, 1> trans_rng_buf{ &trans_range, 1L };

		event1 = device_queue.submit([&](cl::sycl::handler& cgh) { \
			cgh.depends_on(event);
		auto trans_rng_acc = trans_rng_buf.get_access<cl::sycl::access::mode::read>(cgh);
		cgh.parallel_for(cl::sycl::nd_range<2>{ \
			cl::sycl::range<2>{cnds, cnds}, cl::sycl::range<2>{1, 1}}, \
			[=, cnds_ptr = &cnds_buf[0], cnds_ptr1 = &cnds_buf1[0]](cl::sycl::nd_item<2> item) {

			cl::sycl::id<2> idx = item.get_global_id();
			std::size_t i = static_cast<std::size_t>(idx[0]);
			std::size_t j = static_cast<std::size_t>(idx[1]);
			std::size_t cand_rule_id = item.get_global_linear_id();

			if (j > i)
			{
				std::size_t isect_size = mm_vector::intersect_vec(cnds_ptr[i].m_v, \
					cnds_ptr[i].m_items, cnds_ptr[j].m_v, cnds_ptr[j].m_items);

				if ((isect_size > 0L) || ((cnds_ptr[i].m_items <= 2) && (cnds_ptr[j].m_items <= 2)))
				{
					cnds_ptr1[cand_rule_id].m_items = mm_vector::union_vec(cnds_ptr[i].m_v, cnds_ptr[i].m_items, \
						cnds_ptr[j].m_v, cnds_ptr[j].m_items, cnds_ptr1[cand_rule_id].m_v);

					cnds_ptr1[cand_rule_id].m_supp_ab = \
						get_support(cnds_ptr1[cand_rule_id], trans_ctx, trans_rng_acc[0]);

					cnds_ptr1[cand_rule_id].m_supp_a = cnds_ptr[i].m_supp_ab;
					cnds_ptr1[cand_rule_id].m_supp_b = cnds_ptr[j].m_supp_ab;

					double conf_ab = cnds_ptr1[cand_rule_id].m_supp_ab / cnds_ptr1[cand_rule_id].m_supp_a;
					double conf_ba = cnds_ptr1[cand_rule_id].m_supp_ab / cnds_ptr1[cand_rule_id].m_supp_b;

					cnds_ptr1[cand_rule_id].m_conf = std::max<double>(conf_ab, conf_ba);
				}
			}
		});
			});

		device_queue.wait_and_throw();

		filter_cands(cnds_buf1, cnds_new, [&](const MMN_RULE& rule) {
			return (rule.m_items > 1L) && (rule.m_supp_ab > 1.0f) && \
				(rule.m_supp_a > 1.0f) && (rule.m_supp_b > 1.0f); });

		if (cnds_new > 0L)
			minmax_conf = get_minmax_conf(cnds_buf1, cnds_new);

		filter_cands(cnds_buf1, cnds_new, [&](const MMN_RULE& rule) {
			return rule.m_supp_ab >= get_mean_conf(minmax_conf); });

		usm_alloc.free_cnds_buf(cnds_buf, cnds);

		usm_alloc.alloc_cnds_buf(cnds_buf, cnds_new, \
			trans_ctx->m_stats.m_trans_max_len, \
			trans_ctx->m_stats.m_item_max_len);

		event1 = device_queue.submit([&](cl::sycl::handler& cgh) { \
			cgh.depends_on(event1);
		cgh.parallel_for(cl::sycl::range<2>{cnds_new, trans_ctx->m_stats.m_trans_max_len}, \
			[=, cnds_ptr = &cnds_buf[0], cnds_ptr1 = &cnds_buf1[0]](cl::sycl::id<2> idx) {
				std::size_t ii = static_cast<std::size_t>(idx[0]);
				std::size_t jj = static_cast<std::size_t>(idx[1]);

				if (jj < cnds_ptr[ii].m_items) 
				{
					cnds_ptr[ii].m_conf = cnds_ptr1[ii].m_conf;
					cnds_ptr[ii].m_items = cnds_ptr1[ii].m_items;
					cnds_ptr[ii].m_supp_a = cnds_ptr1[ii].m_supp_a;
					cnds_ptr[ii].m_supp_b = cnds_ptr1[ii].m_supp_b;
					cnds_ptr[ii].m_supp_ab = cnds_ptr1[ii].m_supp_ab;

					usm_string_helper::strcpy(cnds_ptr[ii].m_v[jj].m_buf, \
						cnds_ptr[ii].m_v[jj].m_size, cnds_ptr1[ii].m_v[jj].m_buf, cnds_ptr1[ii].m_v[jj].m_size);
				}
			});
		});

		device_queue.wait_and_throw();

		cnds = cnds_new;

		usm_alloc.free_cnds_buf(cnds_buf1, cnds_new);

		return event1;
	}

	void update_model(MMN_RULE*& cnds_buf, \
		std::size_t& cnds, minmax_conf_type& minmax_conf)
	{
		if ((cnds > 0L) && (cnds_buf != nullptr))
		{
			filter_cands(cnds_buf, cnds, [&](const MMN_RULE& rule) {
				double conf = rule.m_supp_ab / rule.m_supp_a;
				return (conf >= get_mean_conf(minmax_conf));
			});

			minmax_conf = get_minmax_conf(cnds_buf, cnds);
		}
	}

	cl::sycl::event remove_subsets(MMN_RULE*& cnds_buf, std::size_t& cnds, \
		cl::sycl::event event, cl::sycl::queue device_queue)
	{
		event = device_queue.submit([&](cl::sycl::handler& cgh) { \
			cgh.depends_on(event);
			cgh.parallel_for(cl::sycl::range<2>{cnds, cnds}, \
				[=, cnds_ptr = &cnds_buf[0]](cl::sycl::id<2> idx) {

			std::size_t ii = static_cast<std::size_t>(idx[0]);
			std::size_t jj = static_cast<std::size_t>(idx[1]);

			if ((jj > ii) && (cnds_ptr[ii].m_supp_ab != .0f))
			{
				if (cnds_buf[jj].m_items < cnds_buf[ii].m_items)
				{
					std::size_t isect_size = mm_vector::intersect_vec(cnds_buf[ii].m_v, \
						cnds_buf[ii].m_items, cnds_buf[jj].m_v, cnds_buf[jj].m_items);

					cnds_buf[jj].m_supp_ab = (isect_size != \
						cnds_buf[jj].m_items) ? cnds_buf[jj].m_supp_ab : .0f;
				}
			}
		});
			});

		device_queue.wait_and_throw();

		filter_cands(cnds_buf, cnds, [&](const MMN_RULE& rule) {
			return (rule.m_supp_ab != .0f); });

		return event;
	}

	void remove_subsets(MMN_RULE*& cnds_buf, MMN_RULE* rules_buf, MMN_TRANS_CONTEXT* trans_ctx, \
		std::size_t& cnds, std::size_t rules)
	{
		for (std::size_t ii = 0; ii < cnds; ii++)
			if (cnds_buf[ii].m_supp_ab != .0f)
			{
				bool exists = false;
				for (std::size_t jj = 0; jj < rules && !exists; jj++)
				{
					if (rules_buf[jj].m_items < cnds_buf[ii].m_items)
					{
						std::size_t isect_size = mm_vector::intersect_vec(cnds_buf[ii].m_v, \
							cnds_buf[ii].m_items, rules_buf[jj].m_v, rules_buf[jj].m_items);

						exists = isect_size == rules_buf[jj].m_items;
					}
				}

				if (exists == true) {
					cnds_buf[ii].m_supp_ab = .0f;
				}
			}

		filter_cands(cnds_buf, cnds, [&](const MMN_RULE& rule) {
			return (rule.m_supp_ab != .0f); });
	}

	cl::sycl::event remove_duplicates(MMN_RULE*& cnds_buf, std::size_t& cnds, \
		cl::sycl::event event, cl::sycl::queue device_queue)
	{
		event = device_queue.submit([&](cl::sycl::handler& cgh) { \
			cgh.depends_on(event);
			cgh.parallel_for(cl::sycl::range<2>{cnds, cnds}, \
				[=, cnds_ptr = &cnds_buf[0]](cl::sycl::id<2> idx) {

				std::size_t ii = static_cast<std::size_t>(idx[0]);
				std::size_t jj = static_cast<std::size_t>(idx[1]);

				if ((jj > ii) && (cnds_ptr[ii].m_supp_ab != .0f))
				{
					std::size_t isect_size = mm_vector::intersect_vec(cnds_ptr[ii].m_v, \
						cnds_ptr[ii].m_items, cnds_ptr[jj].m_v, cnds_ptr[jj].m_items);

					cnds_ptr[jj].m_supp_ab = ((mm_vector::is_equal_vec(cnds_ptr[ii].m_v, \
						cnds_ptr[ii].m_items, cnds_ptr[jj].m_v, cnds_ptr[jj].m_items) == false) && \
						((cnds_ptr[ii].m_items < cnds_ptr[jj].m_items) || (isect_size != cnds_ptr[jj].m_items))) ? cnds_ptr[jj].m_supp_ab : .0f;
				}

			});
		});

		device_queue.wait_and_throw();

		filter_cands(cnds_buf, cnds, [&](const MMN_RULE& rule) {
			return (rule.m_supp_ab != .0f); });

		return event;
	}

	cl::sycl::event compute(MMN_RULE*& cnds_buf, const std::size_t cnds, \
		MMN_RULE*& rules_buf, std::size_t& rules_size, MMN_TRANS_CONTEXT* trans_ctx, \
			minmax_conf_type& minmax_conf, trans_range_type trans_range, \
			cl::sycl::event event, cl::sycl::queue device_queue)
	{
		sycl_usm_alloc_helper usm_alloc(\
			"mm_candidate buffers", USM_ALLOC_TYPE::usm_alloc_crt);

		std::size_t cnds_count = cnds;

		MMN_RULE* cnds_buf2 = nullptr;

		usm_alloc.alloc_cnds_buf(cnds_buf2, cnds_count, \
			trans_ctx->m_stats.m_trans_max_len, \
			trans_ctx->m_stats.m_item_max_len);

		event = device_queue.submit([&](cl::sycl::handler& cgh) { \
		cgh.depends_on(event);
		cgh.parallel_for(cl::sycl::range<2>{cnds, trans_ctx->m_stats.m_trans_max_len}, \
			[=, cnds_ptr = &cnds_buf[0], cnds_ptr2 = &cnds_buf2[0]](cl::sycl::id<2> idx) {
			std::size_t ii = static_cast<std::size_t>(idx[0]);
			std::size_t jj = static_cast<std::size_t>(idx[1]);

			if (jj < cnds_ptr[ii].m_items)
			{
				cnds_ptr2[ii].m_conf = cnds_ptr[ii].m_conf;
				cnds_ptr2[ii].m_items = cnds_ptr[ii].m_items;
				cnds_ptr2[ii].m_supp_a = cnds_ptr[ii].m_supp_a;
				cnds_ptr2[ii].m_supp_b = cnds_ptr[ii].m_supp_b;
				cnds_ptr2[ii].m_supp_ab = cnds_ptr[ii].m_supp_ab;

				usm_string_helper::strcpy(cnds_ptr2[ii].m_v[jj].m_buf, \
					cnds_ptr2[ii].m_v[jj].m_size, cnds_ptr[ii].m_v[jj].m_buf, cnds_ptr[ii].m_v[jj].m_size);
			}
		});
			});

		device_queue.wait_and_throw();

		for (std::size_t step = 0L; cnds_count > 0L; step++)
		{
			if (step > 0L) 
			{
				tbb::spin_mutex mutex;
				std::size_t rules_count = 0L;
				tbb::parallel_for(tbb::blocked_range<std::size_t>(0, cnds_count), \
					[&](const tbb::blocked_range<std::size_t>& r) {
						for (std::size_t ii = r.begin(); ii != r.end(); ii++)
						{
							bool has_subset = false;
							tbb::parallel_for(tbb::blocked_range<std::size_t>(ii + 1, cnds_count), \
								[&](const tbb::blocked_range<std::size_t>& r) {
									for (std::size_t jj = r.begin(); jj != r.end() && !has_subset; jj++)
										has_subset = mm_vector::intersect_vec(cnds_buf2[ii].m_v, \
											cnds_buf2[ii].m_items, cnds_buf2[jj].m_v, cnds_buf2[jj].m_items) > 0L;
								});

							if (has_subset == false)
							{
								bool exists = false;
								tbb::parallel_for(tbb::blocked_range<std::size_t>(0, rules_size), \
									[&](const tbb::blocked_range<std::size_t>& r) {
										for (std::size_t tt = r.begin(); tt != r.end() && !exists; tt++)
											exists = mm_vector::intersect_vec(rules_buf[tt].m_v, \
												rules_buf[tt].m_items, cnds_buf2[ii].m_v, \
												cnds_buf2[ii].m_items) == cnds_buf2[ii].m_items;
									});

								if (exists == false)
								{
									tbb::spin_mutex::scoped_lock lock(mutex);

									usm_alloc.realloc_buf_async<MMN_RULE>(\
										rules_buf, (rules_size + 1));

									rules_buf[rules_size].m_v = nullptr;
									rules_buf[rules_size].m_conf = cnds_buf2[ii].m_conf;
									rules_buf[rules_size].m_items = cnds_buf2[ii].m_items;
									rules_buf[rules_size].m_supp_a = cnds_buf2[ii].m_supp_a;
									rules_buf[rules_size].m_supp_b = cnds_buf2[ii].m_supp_b;
									rules_buf[rules_size].m_supp_ab = cnds_buf2[ii].m_supp_ab;

									usm_alloc.alloc_items_buf(rules_buf[rules_size].m_v, \
										cnds_buf2[ii].m_items, trans_ctx->m_stats.m_item_max_len);

									for (std::size_t jj = 0; jj < rules_buf[rules_size].m_items; jj++)
									{
										rules_buf[rules_size].m_v[jj].m_size = \
											cnds_buf2[ii].m_v[jj].m_size;

										usm_string_helper::strcpy(rules_buf[rules_size].m_v[jj].m_buf, \
											rules_buf[rules_size].m_v[jj].m_size, cnds_buf2[ii].m_v[jj].m_buf, cnds_buf2[ii].m_v[jj].m_size);
									}

									rules_size++; rules_count++;
								}
							}
						}
					});

				if (rules_count == 0L) {
					return event;
				}
			}

			event = gen_sub_nodes(cnds_buf2, trans_ctx, \
				cnds_count, minmax_conf, trans_range, event, device_queue);

			std::sort(cnds_buf2, cnds_buf2 + cnds_count, \
				[&](const MMN_RULE& r1, const MMN_RULE& r2) {
					return r1.m_items > r2.m_items;
				});

			event = remove_duplicates(cnds_buf2, cnds_count, event, device_queue);
			event = remove_subsets(cnds_buf2, cnds_count, event, device_queue);
		}

		std::sort(rules_buf, rules_buf + rules_size, \
			[&](const MMN_RULE& r1, const MMN_RULE& r2) {
				return r1.m_items > r2.m_items;
			});

		usm_alloc.free_cnds_buf(cnds_buf2, cnds);

		return event;
	}

	void compute(MMN_TRANS_CONTEXT* trans_ctx, \
		MMN_RULE*& rules_buf, std::size_t& rules_size)
	{
		cl::sycl::event event;

		MMN_RULE* cnds_buf = nullptr;
		std::size_t cnds_size = \
			trans_ctx->m_stats.m_items_cnt;

		minmax_conf_type minmax_conf;
		parallel_max_miner::init_model(cnds_buf, \
			trans_ctx, cnds_size, minmax_conf);

		int nth = 36;
		tbb::task_group tg;
		auto mp = tbb::global_control::max_allowed_parallelism;
		tbb::global_control gc(mp, nth);

		std::size_t chunk_size = trans_ctx->m_stats.m_trans_cnt * 0.1f;
		std::size_t chunks_n = static_cast<std::size_t>( \
			std::ceil(trans_ctx->m_stats.m_trans_cnt / double(chunk_size)));

		tg.run_and_wait([&]() {
			tbb::spin_mutex mutex;
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, chunks_n), \
				[&](const tbb::blocked_range<std::size_t>& r) {
					for (std::size_t ii = r.begin(); ii != r.end(); ii++)
					{
						std::size_t xs = ii * chunk_size;
						std::size_t xe = (ii + 1) * chunk_size;

						if (xe > trans_ctx->m_stats.m_trans_cnt)
							xe = trans_ctx->m_stats.m_trans_cnt;

						cl::sycl::cpu_selector s{};
						cl::sycl::queue device_queue(s);

						trans_range_type trans_range = std::make_pair(xs, xe);
						event = parallel_max_miner::compute(cnds_buf, cnds_size, rules_buf, \
							rules_size, trans_ctx, minmax_conf, trans_range, event, device_queue);

						if (rules_size > 0L) {
							event = remove_duplicates(rules_buf, rules_size, event, device_queue);
							event = remove_subsets(rules_buf, rules_size, event, device_queue);
						}
					}
			});
		});
	}

	MMN_TRANS_CONTEXT* m_trans_ctx;
};