#include "utils.hpp"

/*****************************************************
 1: tbb support multi-threads
 2: second param:(lhs + rhs;), will be executed in main thread.
 ****************************************************/

const int64_t my_range = 30000000;
inline void original_implment()
{
    TEST_ITM_TIP();
    PRINT_ITM(my_range);

    int64_t sum_src = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < my_range; i++)
    {
        sum_src += 1;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << TIME_DIFF(t2, t1) << ", src sum=" << sum_src << std::endl;
}

inline void parallel_reduce_impl()
{
    TEST_ITM_TIP();
    PRINT_ITM(my_range);
    PRINT_THREAD_ID();

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t sum = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int64_t>(0, my_range), 0,
        [](oneapi::tbb::blocked_range<int64_t> const &r, int64_t init) -> int64_t
        {
            // std::cout << "fun1 " << std::this_thread::get_id() << std::endl;
            for (int64_t v = r.begin(); v != r.end(); v++)
            {
                init += 1;
            }
            return init;
        },
        [](int64_t lhs, int64_t rhs) -> int64_t
        {
            return lhs + rhs;
        });

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << TIME_DIFF(t2, t1) << ", parallel_reduce sum=" << sum << std::endl;
}

inline void parallel_reduce_impl_2_threads()
{
    TEST_ITM_TIP();
    auto cur_range = my_range / 2;
    PRINT_ITM(cur_range);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto fun1 = [&]()
    {
        PRINT_THREAD_ID();
        int64_t sum = oneapi::tbb::parallel_reduce(
            oneapi::tbb::blocked_range<int64_t>(0, cur_range), 0,
            [](oneapi::tbb::blocked_range<int64_t> const &r, int64_t init) -> int64_t
            {
                // std::cout << "fun1 " << std::this_thread::get_id() << std::endl;
                for (int64_t v = r.begin(); v != r.end(); v++)
                {
                    init += 1;
                }
                return init;
            },
            [](int64_t lhs, int64_t rhs) -> int64_t
            {
                return lhs + rhs;
            });
        std::cout << "fun1 parallel_reduce sum=" << sum << std::endl;
    };

    auto fun2 = [&]()
    {
        PRINT_THREAD_ID();
        int64_t sum = oneapi::tbb::parallel_reduce(
            oneapi::tbb::blocked_range<int64_t>(0, cur_range), 0,
            [](oneapi::tbb::blocked_range<int64_t> const &r, int64_t init) -> int64_t
            {
                // std::cout << "fun2 " << std::this_thread::get_id() << std::endl;
                for (int64_t v = r.begin(); v != r.end(); v++)
                {
                    init += 1;
                }
                return init;
            },
            [](int64_t lhs, int64_t rhs) -> int64_t
            {
                return lhs + rhs;
            });
        std::cout << "fun2 parallel_reduce sum=" << sum << std::endl;
    };

    std::thread thrd1 = std::thread(fun1);
    std::thread thrd2 = std::thread(fun2);
    if (thrd1.joinable())
        thrd1.join();
    if (thrd2.joinable())
        thrd2.join();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << TIME_DIFF(t2, t1) << std::endl;
}

void test_parallel_reduce()
{
    original_implment();
    parallel_reduce_impl();
    parallel_reduce_impl_2_threads();
}