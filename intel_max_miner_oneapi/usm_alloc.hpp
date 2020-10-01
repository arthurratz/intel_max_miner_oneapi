//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>

#include <memory>

#include "mm_types.hpp"

typedef enum {
	usm_alloc_crt	 = 0x01,
	usm_alloc_host	 = 0x02,
	usm_alloc_device = 0x03,
	usm_alloc_shared = 0x04
} USM_ALLOC_TYPE;

constexpr unsigned short g_threads = 36;

class sycl_usm_alloc_ptr
{
public:
	sycl_usm_alloc_ptr(\
		const char* context_name, \
		USM_ALLOC_TYPE alloc_type = USM_ALLOC_TYPE::usm_alloc_crt) : \
			m_alloc_type(alloc_type), m_context_name(context_name) {};

public:
	template<class Type>
	void alloc_buffer(Type*& buf_ptr, const std::size_t size) {
		if ((size > 0L) && (buf_ptr == nullptr)) {
			buf_ptr = alloc_buf_async<Type>(size * sizeof(Type));
		}
	}
	
	template<class Type>
	void free_buffer(Type*& buf_ptr) {
		if (buf_ptr != nullptr) {
			switch (m_alloc_type) {
				case USM_ALLOC_TYPE::usm_alloc_crt:
					std::free(buf_ptr); break;
				default: break;
			}

			buf_ptr = nullptr;
		}
	}

//private:
public:
	template<class Type>
	Type* alloc_buf_async(const std::size_t size) {
 	    bool alloc_failed = false;
		Type* buf_ptr_local = nullptr;
		//do {
			try {
				alloc_failed = false;
				switch (m_alloc_type) {
					case USM_ALLOC_TYPE::usm_alloc_crt:
						buf_ptr_local = static_cast<Type*>(std::malloc(size)); break;
					default: buf_ptr_local = nullptr; break;
				}

				switch (m_alloc_type) {
					case USM_ALLOC_TYPE::usm_alloc_crt: 
						std::memset((void*)buf_ptr_local, 0x00, size); break;
					default: break;
				}
			}
			catch (std::bad_alloc& e) {
				alloc_failed = true;
				std::cerr << "Error: (" << m_context_name << \
					") memory buffer allocation failure: " << e.what() << "\n";
			}
		//} while ((alloc_failed != false) || \
		//	(buf_ptr_local == nullptr));

		return buf_ptr_local;
	}

	template<class Type>
	void realloc_buf_async(Type*& buf_ptr, const std::size_t size) {
		bool alloc_failed = false;
		//do {
		try {
			buf_ptr = static_cast<Type*>( \
				std::realloc(buf_ptr, size * sizeof(Type)));
		}
		catch (std::bad_alloc& e) {
			alloc_failed = true;
			std::cerr << "Error: (" << m_context_name << \
				") memory buffer re-allocation failure: " << e.what() << "\n";
		}
		//} while ((alloc_failed != false) || \
		//	(buf_ptr_local == nullptr));
	}

protected:
	const USM_ALLOC_TYPE m_alloc_type;
	const char*			 m_context_name;
};

class sycl_usm_alloc_helper : \
	public sycl_usm_alloc_ptr
{
public:
	sycl_usm_alloc_helper(const char* context_name, \
		USM_ALLOC_TYPE alloc_type = USM_ALLOC_TYPE::usm_alloc_crt) : \
			sycl_usm_alloc_ptr(context_name, alloc_type) {}

public:
	/*void alloc_trans_buf(MMN_TRANS*& trans_buf, const std::size_t trans, \
		const std::size_t items_per_trans, const std::size_t item_size) {

		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
			if ((trans_buf == nullptr) && (trans > 0L) && \
				(items_per_trans > 0L) && (item_size > 0L)) {
				alloc_buffer<MMN_TRANS>(trans_buf, trans);
				//tbb::parallel_for(tbb::blocked_range<std::size_t>(0, trans), \
				//	[&](const tbb::blocked_range<std::size_t>& r) {
						//for (std::size_t ii = r.begin(); ii != r.end(); ii++)
						for (std::size_t ii = 0; ii < trans; ii++)
						{
							trans_buf[ii].m_items = items_per_trans;
							alloc_items_buf(trans_buf[ii].m_v, items_per_trans, item_size);
						}
				//	});
			}
		//});
	}*/

	void alloc_cnds_buf(MMN_RULE*& cnds_buf, const std::size_t cnds, \
		const std::size_t items_per_cnd, const std::size_t item_len)
	{
		//if (rules_buf != nullptr)
		//	free_rules_buf(rules_buf, rules, items_per_rule);

		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
		if ((cnds_buf == nullptr) && (cnds > 0L) && \
			(items_per_cnd > 0L) && (item_len > 0L)) {
			alloc_buffer<MMN_RULE>(cnds_buf, cnds);
			//tbb::parallel_for(tbb::blocked_range<std::size_t>(0, rules), \
				//[&](const tbb::blocked_range<std::size_t>& r) {
					//for (std::size_t ii = r.begin(); ii != r.end(); ii++)
					for (std::size_t ii = 0; ii < cnds; ii++)
						alloc_cnds_node(cnds_buf[ii], items_per_cnd, item_len);
				//});
		}
		//});
	}

	void alloc_trans_ctx(\
		MMN_TRANS_CONTEXT*& trans_ctx, const MMN_ITEM* items_buf, \
		const MMN_TRANS* trans_buf, const MMN_TRANS_STATS stats) {
		if (trans_ctx == nullptr) {
			MMN_TRANS_CONTEXT* context = nullptr;
			alloc_buffer<MMN_TRANS_CONTEXT>(context, 1);

			if ((items_buf != nullptr) && (trans_buf != nullptr)) {
				context->m_stats = stats;
				context->m_items = const_cast<MMN_ITEM*>(items_buf);
				context->m_trans = const_cast<MMN_TRANS*>(trans_buf);
			}

			if ((context->m_items != nullptr) && \
				(context->m_trans != nullptr)) {
				trans_ctx = context;
			}
		}
	}

public:
	void free_trans_buf(MMN_TRANS*& trans_buf, const std::size_t trans) {

		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
		if ((trans_buf != nullptr) && (trans > 0L)) {
			//tbb::parallel_for(tbb::blocked_range<std::size_t>(0, trans), \
			//	[&](const tbb::blocked_range<std::size_t>& r) {
					//for (std::size_t ii = r.begin(); ii != r.end(); ii++)
			for (std::size_t ii = 0; ii < trans; ii++) {
				//free_items_buf(trans_buf[ii].m_v, trans_buf[ii].m_items); trans_buf[ii].m_v = nullptr;
			}
			//	});

			/*free_buffer<MMN_TRANS>(trans_buf);*/ trans_buf = nullptr;
		}
		//});
	}
	
	void free_cnds_buf(MMN_RULE*& cnds_buf, const std::size_t cnds) 
	{
		if ((cnds_buf != nullptr) && (cnds > 0L)) {
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, cnds), \
				[&](const tbb::blocked_range<std::size_t>& r) {
					for (std::size_t ii = r.begin(); ii != r.end(); ii++)
						free_rule_node(&cnds_buf[ii], cnds_buf[ii].m_items);
				});

			free_buffer<MMN_RULE>(cnds_buf); cnds_buf = nullptr;
		}
	}

	void free_trans_ctx(MMN_TRANS_CONTEXT*& trans_ctx) {
		if (trans_ctx != nullptr) {

			//free_items_buf(trans_ctx->m_tokens, trans_ctx->m_stats.m_tokens_count);
			//free_trans_buf(trans_ctx->m_trans_ptr, trans_ctx->m_stats.m_trans_count);

			/*free_buffer<MMN_TRANS_CONTEXT>(trans_ctx);*/ trans_ctx = nullptr;
		}
	}

public:
	void alloc_items_buf(MMN_ITEM*& items_buf, \
		const std::size_t items_size, const std::size_t item_len) {

		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
		if ((items_buf == nullptr) && (items_size > 0L)) {
			alloc_buffer<MMN_ITEM>(items_buf, items_size);
			//tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), \
				//	[&](const tbb::blocked_range<std::size_t>& r) {
						//for (std::size_t ii = r.begin(); ii != r.end(); ii++)
			for (std::size_t ii = 0; ii < items_size; ii++)
			{
				items_buf[ii].m_size = item_len;
				alloc_buffer<char>(items_buf[ii].m_buf, item_len);
				if (m_alloc_type == USM_ALLOC_TYPE::usm_alloc_crt)
					std::memset((void*)items_buf[ii].m_buf, 0x00, item_len);
			}
			//});
		}
		//});
	}

	void alloc_cnds_node(MMN_RULE& cnds, \
		const std::size_t cnds_size, const std::size_t item_len)
	{
		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
		if ((cnds.m_v == nullptr) && (cnds_size > 0L)) {
			cnds.m_items = cnds_size; /*cnds.m_supp_a = */cnds.m_supp_ab = .0f;
			alloc_items_buf(cnds.m_v, cnds.m_items, item_len);
		}
		//});
	}

public:
	void free_items_buf(MMN_ITEM*& items_buf, const std::size_t items_size) {

		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
		if ((items_buf != nullptr) && (items_size > 0L)) {
			//tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), \
			//	[&](const tbb::blocked_range<std::size_t>& r) {
					//for (std::size_t ii = r.begin(); ii != r.end(); ii++) {
					for (std::size_t ii = 0; ii < items_size; ii++) {
						free_buffer<char>(items_buf[ii].m_buf); 
						items_buf[ii].m_buf = nullptr;
					}
				//});

			free_buffer<MMN_ITEM>(items_buf); 
			items_buf = nullptr;
		}
		//});
	}

	void free_rule_node(MMN_RULE* rule, const std::size_t items)
	{
		//tbb::task_group task_group;
		//auto mp = tbb::global_control::max_allowed_parallelism;
		//tbb::global_control gc(mp, g_threads);

		//task_group.run_and_wait([&]() {
		if ((rule->m_v != nullptr) && (items > 0L)) {
			rule->m_items = 0L; /*rule->m_supp_a = */rule->m_supp_ab = .0f;
			free_items_buf(rule->m_v, rule->m_items);
		}

		/*free_buffer<MMN_RULE>(rule);*/ rule = nullptr;
		//});
	}
};