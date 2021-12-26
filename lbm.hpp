//
// Created by jonas on 12/26/21.
//

#ifndef SYCLTEST_LBM_HPP
#define SYCLTEST_LBM_HPP

#include <CL/sycl.hpp>
#include <tuple>

using namespace cl::sycl;

namespace lbm {
    template<std::floating_point S>
    struct params {
        S nu;
        S v_max;
        size_t width;
        size_t height;
    };

    template<typename F, typename S>
    concept Initializer = requires(F f, params<S> p, S s, size_t i) {
        {s} -> std::floating_point;
        {f(i, i, p)} -> std::convertible_to<std::tuple<S, S, S>>;
    };

    template<std::floating_point S>
    struct D2Q9 {
        constexpr static S cs = 1.0 / std::sqrt(3.0);
        constexpr static std::array<S, 9> W = {16.0/36.0, 4.0/36.0, 4.0/36.0, 4.0/36.0, 4.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
        constexpr static std::array<std::array<int, 9>, 2> c = {{{{       0,       1,       0,      -1,       0,       1,      -1,      -1,       1}},
                                                      {{       0,       0,       1,       0,      -1,       1,       1,      -1,      -1}}}};
        constexpr static size_t size = 9;
        constexpr static std::array<S, 9> f_eq(S rho, S u, S v) {
            std::array<S, 9> f;

            S x_root = std::sqrt(1 + 3 * u * u);
            S y_root = std::sqrt(1 + 3 * v * v);

            S A = rho * (2 - x_root) * (2 - y_root);
            S BX = (2 * u + x_root) / (1 - u);
            S BY = (2 * v + y_root) / (1 - v);

            for (int k = 0; k < size; k++) {
                f_eq[k] = W[k] * A * pow1(BX, c[0][k]) * pow1(BX, c[1][k]);
            }

            return f_eq;
        }
    };

    template<std::floating_point S>
    inline S pow1(S s, int i) {
        return i == 0 ? (S)1 : i == -1 ? i == (S) 1 / s : s;
    }

    template<std::floating_point S, template <typename> class V>
    class simulation {
        queue q;
        size_t current_source = 0;

        using v = V<S>;

        params<S> p;
        const range<3> size = range<3>(p.width, p.height, 9);
        std::array<buffer<S, 3>, 2> device_bufs;
        S beta;

        simulation(params<S> p, Initializer<S> auto& initializer) : p(p){

            device_bufs[0].init(size);
            device_bufs[1].init(range<3>(p.width, p.height, 9));

            beta = 1 / (2 * p.nu / (v::cs * v::cs) + 1);

            init(initializer);
        }

        void init(Initializer<S> auto& initializer) {
            auto& buf = device_bufs[current_source];

            q.submit([&](const handler& h) {
                accessor<S, 3, access::mode::discard_write> a(h, buf);

                h.parallel_for(nd_range<2> (range<2>(p.width, p.height), range<2>(16,16)), [&](auto& id) {
                    size_t i = id.get(0);
                    size_t j = id.get(1);
                    std::convertible_to<S> auto [rho, u, v] = initializer(i, j, p);

                    auto f = v::f_eq(rho, u, v);

                    for(int k = 0; k < v::size; k++) {
                        a[id(i,j,k)] = f[k];
                    }
                });
            });
        }

        S reynolds() {
            return p.width * p.v_max / p.nu;
        }
    };
}

#endif //SYCLTEST_LBM_H
