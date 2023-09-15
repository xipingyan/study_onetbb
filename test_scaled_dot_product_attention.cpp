#include "utils.hpp"

/*****************************************************
 Test scaled_dot_product_attention's optimization.
 1: original impl
 2: tbb impl
 3: onednn version
 4: amx instrisic
 5: tbb+amx 
 ****************************************************/
struct MyTensor
{
    float *data = nullptr;
    int b = 0, n = 0, w = 0, h = 0;
};

MyTensor* randn_my_tensor(int b, int n, int h, int w, bool init_zero=false) {
    MyTensor *tensor = (MyTensor *)malloc(sizeof(MyTensor));
    tensor->data = (float *)malloc(sizeof(float)*b*n*w*h);
    tensor->b=b;
    tensor->n=n;
    tensor->w=w;
    tensor->h=h;
    for (auto b = 0; b < tensor->b; b++)
    {
        float *dst1 = &tensor->data[tensor->n * tensor->h * tensor->w * b];
        for (auto n = 0; n < tensor->n; n++)
        {
            float *dst2 = &dst1[tensor->h * tensor->w * n];
            for (auto h = 0; h < tensor->h; h++)
            {
                float *dst3 = &dst1[tensor->w * n];
                for (auto w = 0; w < tensor->w; w++)
                {
                    dst3[w] = init_zero ? 0 : static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                }
            }
        }
    }
    return tensor;
}

void del_my_tensor(MyTensor *tensor)
{
    if (tensor)
    {
        if (tensor->data)
        {
            free(tensor->data);
            tensor->data = nullptr;
        }
        free(tensor);
    }
}

inline MyTensor* original_impl(MyTensor* q,MyTensor* k,MyTensor* v)
{
    TEST_ITM_TIP();
    auto q_k = randn_my_tensor(q->b, q->n, q->h, k->h, true);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto b = 0; b < q->b; b++)
    {
        float *dst1 = &q_k->data[q_k->n * q_k->h * q_k->w * b];
        float *q1 = &q->data[q->n * q->h * q->w * b];
        float *k1 = &k->data[k->n * k->h * k->w * b];
        for (auto n = 0; n < q->n; n++)
        {
            float *dst2 = &dst1[q_k->h * q_k->w * n];
            float *q2 = &q->data[q->h * q->w * b];
            float *k2 = &k->data[k->h * k->w * b];
            for (auto h = 0; h < q->h; h++)
            {
                float *dst3 = &dst1[q_k->w * n];
                for (auto w = 0; w < q->w; w++)
                {
//
#if 1
                    dst3[w] parallel_reduce(
                        blocked_range<float *>(array, array + n),
                        0.f,
                        [](const blocked_range<float *> &r, float init) -> float
                        {
                            for (float *a = r.begin(); a != r.end(); ++a)
                                init += *a;
                            return init;
                        },
                        [](float x, float y) -> float
                        {
                            return x + y;
                        });
#else
                    for (int i_q = 0; i_q < q->w; i_q++)
                    {
                        dst3[w] += (q2[q->w * h + i_q] * k2[q->w * w + i_q]);
                    }
#endif
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    PRINT_TIME_DIFF(t2, t1);
    return q_k;
}

inline MyTensor* tbb_impl(MyTensor* q,MyTensor* k,MyTensor* v)
{
    TEST_ITM_TIP();
    auto q_k = randn_my_tensor(q->b, q->n, q->h, k->h, true);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto b = 0; b < q->b; b++)
    {
        float *dst1 = &q_k->data[q_k->n * q_k->h * q_k->w * b];
        float *q1 = &q->data[q->n * q->h * q->w * b];
        float *k1 = &k->data[k->n * k->h * k->w * b];
        for (auto n = 0; n < q->n; n++)
        {
            float *dst2 = &dst1[q_k->h * q_k->w * n];
            float *q2 = &q->data[q->h * q->w * b];
            float *k2 = &k->data[k->h * k->w * b];
            for (auto h = 0; h < q->h; h++)
            {
                float *dst3 = &dst1[q_k->w * n];
                for (auto w = 0; w < q->w; w++)
                {
                    //
                    dst3[w] = 0;
                    for (int i_q = 0; i_q < q->w; i_q++) {
                        dst3[w] += (q2[q->w * h + i_q] * k2[q->w * w + i_q]);
                    }
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    PRINT_TIME_DIFF(t2, t1);
    return q_k;
}

void test_scaled_dot_product_attention()
{
    auto q = randn_my_tensor(2, 5, 9216, 64);
    auto k = randn_my_tensor(2, 5, 9216, 64);
    auto v = randn_my_tensor(2, 5, 9216, 64);

    for (auto i = 0; i < 5; i++)
    {
        auto attenion_tensor = original_impl(q, k, v);
        del_my_tensor(attenion_tensor);
    }

    for (auto i = 0; i < 5; i++)
    {
        auto attenion_tensor = tbb_impl(q, k, v);
        del_my_tensor(attenion_tensor);
    }

    del_my_tensor(q);
    del_my_tensor(k);
    del_my_tensor(v);
}