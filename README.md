# study_onetbb
Learn how to optimize cpp code based on oneTBB(Threading Building Blocks).

# How to build and run

```
source /opt/intel/oneapi/setvars.sh
source /opt/intel/oneapi/tbb/latest/env/vars.sh
mkdir build && cd build
cmake ..
make -j20
./samples_tests 
numactl -C 0-5 ./samples_tests 
```

# Refer
[1]https://spec.oneapi.io/versions/latest/elements/oneTBB/source/algorithms/functions/parallel_reduce_func.html
