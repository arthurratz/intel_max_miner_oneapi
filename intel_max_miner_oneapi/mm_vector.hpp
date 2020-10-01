//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include "mm_types.hpp"
#include "usm_string.hpp"

class mm_vector
{
public:
	static bool exists(const MMN_ITEM* items_buf, \
		const std::size_t items_size, const char* item) {
		bool exists = false;
		for (std::size_t ii = 0; ii < items_size && !exists; ii++)
			exists = (strcmp(items_buf[ii].m_buf, item) == 0);

		return exists;
	}

	static bool exists(const MMN_ITEM* items_buf, \
		const std::size_t items_size, const MMN_ITEM& item) {
			return exists(items_buf, items_size, item.m_buf);
	}

	static std::size_t intersect_vec( \
		MMN_ITEM* items_buf1, const std::size_t items_buf1_size, \
		MMN_ITEM* items_buf2, const std::size_t items_buf2_size)
	{
		std::size_t isect_size = 0L;
		for (std::size_t ii = 0; ii < items_buf1_size; ii++)
			isect_size += exists(items_buf2, items_buf2_size, items_buf1[ii]);

		return isect_size;
	}

	static std::size_t union_vec(const MMN_ITEM* items_buf1, const std::size_t items_buf1_size, \
		const MMN_ITEM* items_buf2, const std::size_t items_buf2_size, MMN_ITEM*& union_vec)
	{
		std::size_t union_size = 0L;
		for (std::size_t ii = 0; ii < items_buf1_size; ii++)
			usm_string_helper::strcpy(union_vec[union_size++].m_buf, \
				items_buf1[ii].m_size, items_buf1[ii].m_buf, items_buf1[ii].m_size);

		for (std::size_t ii = 0; ii < items_buf2_size; ii++)
			if (!exists(items_buf1, items_buf1_size, items_buf2[ii]))
				usm_string_helper::strcpy(union_vec[union_size++].m_buf, \
					items_buf2[ii].m_size, items_buf2[ii].m_buf, items_buf2[ii].m_size);

		return union_size;
	}

	static bool is_equal_vec(const MMN_ITEM* items_buf1, const std::size_t items_buf1_size, \
		const MMN_ITEM* items_buf2, const std::size_t items_buf2_size)
	{
		bool is_equal = true;
		if (items_buf1_size != items_buf2_size)	return !is_equal;
		for (std::size_t ii = 0; ii < items_buf1_size && is_equal; ii++)
			is_equal = (!strcmp(items_buf1[ii].m_buf, items_buf2[ii].m_buf));
			
		return is_equal;
	}

	static std::string to_string( \
		const MMN_ITEM* item, const std::size_t size)
	{
		std::string result = "\0";
		for (std::size_t i = 0; i < size; i++)
			result += ((strcmp(item[i].m_buf, "\0") != 0) ? std::string(item[i].m_buf) : \
				"N/A") + ((i != size - 1) ? "," : "\0");

		return result;
	}
};