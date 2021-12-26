//
// Created by jonas on 12/26/21.
//

#ifndef SYCLTEST_SIMPLE_COMPLEX_HPP
#define SYCLTEST_SIMPLE_COMPLEX_HPP


template<typename S>
struct complex {
    S a;
    S b;

    complex(S a, S b) : a(a), b(b) {}

    static complex<S> zero() {
        return {S{}, S{}};
    }
};

template<typename S>
complex<S> operator+(complex<S> z1, complex<S> z2) {
    return {z1.a+z2.a, z1.b + z2.b};
}

template<typename S>
complex<S> operator*(complex<S> z1, complex<S> z2) {
    return {z1.a*z2.a-z1.b*z2.b, z1.a * z2.b + z1.b * z2.a};
}

template<typename S>
S mag_sqr (complex<S> z) {
    return z.a * z.a - z.b * z.b;
}



#endif //SYCLTEST_SIMPLE_COMPLEX_HPP
