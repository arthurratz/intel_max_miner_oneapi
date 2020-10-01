//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

class usm_string_helper
{
public:
    static void zerobyte(char*& buf, const std::size_t size)
    {
        std::size_t index = 0L;
        while ((buf != nullptr) && (index < size))
            buf[index++] = 0x00;
    }

    static void strcpy(char*& dst, const std::size_t dst_size, \
        const char* src, const std::size_t src_size) 
    {
        std::size_t index = 0L;
        usm_string_helper::zerobyte(dst, dst_size);
        if ((src != nullptr) && (dst != nullptr)) {
            while ((index < src_size) && \
                ((dst[index] = src[index]) != '\0')) index++;

            dst[index] = '\0';
        }
    }
};