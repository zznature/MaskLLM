/*
 * 快速PyTorch修复库
 * 专注解决ucc_ee_ack_event等关键符号
 */

#include <stdio.h>
#include <stdlib.h>

// 通用函数
int ret_zero() { return 0; }
void ret_void() { }
void* ret_ptr() { return (void*)0x1; }

// 使用宏快速定义符号
#define ZERO_FUNC(name) int name() { return 0; }
#define VOID_FUNC(name) void name() { }
#define PTR_FUNC(name) void* name() { return (void*)0x1; }

// ================================
// UCC EE (Execution Engine) 关键符号
// ================================
ZERO_FUNC(ucc_ee_ack_event)           // 关键!
ZERO_FUNC(ucc_ee_set_event)
ZERO_FUNC(ucc_ee_wait_event) 
ZERO_FUNC(ucc_ee_get_event_status)
ZERO_FUNC(ucc_ee_trigger_event)
ZERO_FUNC(ucc_ee_cancel_event)
ZERO_FUNC(ucc_ee_create)
VOID_FUNC(ucc_ee_destroy)

// ================================
// UCC 核心符号
// ================================
ZERO_FUNC(ucc_init)
ZERO_FUNC(ucc_finalize)
ZERO_FUNC(ucc_context_create)
VOID_FUNC(ucc_context_destroy)
ZERO_FUNC(ucc_context_config_modify)
ZERO_FUNC(ucc_team_create)
ZERO_FUNC(ucc_team_create_test)
VOID_FUNC(ucc_team_destroy)
ZERO_FUNC(ucc_collective_init)
ZERO_FUNC(ucc_collective_post)
ZERO_FUNC(ucc_collective_test)
VOID_FUNC(ucc_collective_finalize)

// ================================
// UCS 符号
// ================================
ZERO_FUNC(ucs_empty_function_return_zero)
VOID_FUNC(ucs_empty_function_return_void)
VOID_FUNC(ucs_mpool_params_reset)
ZERO_FUNC(ucs_mpool_init)
ZERO_FUNC(ucs_mpool_cleanup)
PTR_FUNC(ucs_mpool_get)
VOID_FUNC(ucs_mpool_put)

// ================================
// UCX 符号
// ================================
ZERO_FUNC(ucp_init)
VOID_FUNC(ucp_cleanup)
ZERO_FUNC(ucp_context_create)
VOID_FUNC(ucp_context_destroy)
ZERO_FUNC(ucp_worker_create)
VOID_FUNC(ucp_worker_destroy)
ZERO_FUNC(ucp_worker_progress)

// ================================
// MPI 符号
// ================================
ZERO_FUNC(MPI_Init)
ZERO_FUNC(MPI_Init_thread)
ZERO_FUNC(MPI_Finalize)
ZERO_FUNC(MPI_Comm_rank)
ZERO_FUNC(MPI_Comm_size)
ZERO_FUNC(MPI_Barrier)
ZERO_FUNC(MPI_Allreduce)
ZERO_FUNC(MPI_Bcast)

// ================================
// OPAL/HWLOC 符号
// ================================
ZERO_FUNC(opal_hwloc201_hwloc_get_type_depth)
ZERO_FUNC(opal_hwloc201_hwloc_get_obj_by_depth)

// 构造函数
__attribute__((constructor))
void quick_fix_init() {
    if (getenv("QUICK_FIX_DEBUG")) {
        printf("[QUICK_FIX] 快速修复库已加载\n");
        printf("[QUICK_FIX] 解决ucc_ee_ack_event等关键符号\n");
    }
}
