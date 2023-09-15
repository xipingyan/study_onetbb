
#include <iostream>
#include <vector>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/global_control.h"
#include "tbb/parallel_reduce.h"

#include <chrono>
#include <thread>

#ifndef TEST_ITM_TIP
#define TEST_ITM_TIP() std::cout << "==== " << __FUNCTION__ << " ===\n"
#endif

#ifndef PRINT_ITM
#define PRINT_ITM(VAL) std::cout << #VAL << " = " << VAL << std::endl
#endif

#ifndef TIME_DIFF
#define TIME_DIFF(t2, t1) std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
#endif
#ifndef PRINT_TIME_DIFF
#define PRINT_TIME_DIFF(t2, t1) std::cout << "Take time: " << TIME_DIFF(t2, t1) << " ms" << std::endl
#endif

#ifndef PRINT_THREAD_ID
#define PRINT_THREAD_ID() std::cout << __FUNCTION__ << ":" << __LINE__ << ": Thread id:" << std::this_thread::get_id() << std::endl
#endif

void test_parallel_reduce();
void test_scaled_dot_product_attention();
