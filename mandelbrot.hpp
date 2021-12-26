#include <CL/sycl.hpp>

using namespace cl::sycl;

template<typename S>
std::vector<S> mandelbrot(queue &q, size_t res, size_t max_iter, S min_X, S max_X, S min_Y, S max_Y) {

    nd_range<2> work_items{range<2>{res, res}, range<2>{8, 8}};

    std::vector<S> output(res * res);

    {
        buffer<S> buf_out(output.data(), output.size());

        q.submit([&](handler &cgh) {
            accessor<S> out(buf_out, cgh);

            cgh.parallel_for<class vector_add>(work_items, [=](nd_item<2> tid) {
               size_t i = tid.get_global_id().get(0);
               size_t j = tid.get_global_id().get(1);

               S c_x = (max_X - min_X) * (i / (S) res) + min_X;
               S c_y = (max_Y - min_Y) * (j / (S) res) + min_Y;

               complex<S> c(c_x, c_y);

               auto z = complex<S>::zero();

               int iterations = 0;

               for (int k = 0; k < max_iter; k++) {
                   z = z * z + c;

                   if(mag_sqr(z) > (S)4) {
                       iterations = k;
                       break;
                   }
               };

               out[i + res * j] = (S) iterations / (S) max_iter;
           });
        });
    }

    return output;
}