#include "utils.hpp"

int main(int, char **)
{
    TEST_ITM_TIP();
    // Set max thread num for oneTBB(Threading Building Blocks)
    int nth = 3;
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nth);

    PRINT_ITM(gc.active_value(tbb::global_control::max_allowed_parallelism));

    //
    test_parallel_reduce();

    return 0;
}
