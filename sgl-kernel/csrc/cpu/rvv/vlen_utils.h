
#pragma once

#include <cstddef>
#include <cstdint>

int64_t get_rvv_vlenb();
int64_t get_rvv_vlen();
bool check_vlen_alignment(int64_t size_bytes);
