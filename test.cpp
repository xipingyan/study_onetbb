
#include <iostream>
#include <vector>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/global_control.h"
#include "tbb/parallel_reduce.h"

#include <chrono>
#include <thread>

void test_parallel_reduce()
{
    int64_t my_range = 30000000;
    int64_t sum_src = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int64_t i = 0; i < my_range; i++) {
        sum_src += 1;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << ", src sum=" << sum_src << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    auto fun1 = [&](){
        std::cout << "**fun1 " << std::this_thread::get_id() << std::endl;
    int64_t sum = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int64_t>(0, my_range), 0,
        [](oneapi::tbb::blocked_range<int64_t> const &r, int64_t init) -> int64_t
        {
            std::cout <<  "fun1 " <<std::this_thread::get_id() << std::endl;
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
        std::cout << "**fun2 " << std::this_thread::get_id() << std::endl;
        int64_t sum = oneapi::tbb::parallel_reduce(
            oneapi::tbb::blocked_range<int64_t>(0, my_range), 0,
            [](oneapi::tbb::blocked_range<int64_t> const &r, int64_t init) -> int64_t
            {
                std::cout << "fun2 " << std::this_thread::get_id() << std::endl;
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
    if(thrd1.joinable()) thrd1.join();
    if(thrd2.joinable()) thrd2.join();
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()  << ", parallel_reduce sum=" << std::endl;
}

int main(int, char **)
{
    std::cout << "======================\n";
    auto mp = tbb::global_control::max_allowed_parallelism;
    int nth=2;
    tbb::global_control gc(mp, nth + 1);

    // 
    test_parallel_reduce();

    return 0;
}
