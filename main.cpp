#include <cassert>
#include <iostream>

#include <concepts>
#include <png++/png.hpp>

#include "simple_complex.hpp"
#include "mandelbrot.hpp"
#include "lbm.hpp"

using data_type = float;

template <std::floating_point S>
std::vector<S> shear_layer(size_t width, size_t height, S Re) {
    S kappa = 80.0;
    S delta = 0.05;

    lbm::simulation<float, lbm::D2Q9> sim(
            lbm::params<float>{
                    .nu = 1e-4,
                    .v_max = 0.01,
                    .width = 1024,
                    .height = 1024,
            },
            [&] (const lbm::params<float>& p, size_t i, size_t j) {

                S u = j <=  p.height / 2 ?
                      p.v_max * std::tanh(kappa * (j / (S) p.height - 0.25)) :
                      p.v_max * std::tanh(kappa * (0.75 - j / (S) p.height));

                S v = p.v_max * delta * sin(2 * M_PI * ((S) i / p.width + 0.25));
                S rho = 1.0;

                return std::tuple{rho, u, v};
            }
    );
}

int main() {
    queue q;

    const size_t RES = 4096;

    float x = -0.74529;
    float y = 0.113075;



    png::image<png::gray_pixel> image(RES, RES);

    for(int i = 0; i < RES; i++) {
        for(int j = 0; j < RES; j++) {
            image[j][i] = (uint8_t) (result[i + RES * j] * 256);
        }
    }

    image.write("mandelbrot.png");
}