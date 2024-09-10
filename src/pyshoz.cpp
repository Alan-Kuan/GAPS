#include <cstddef>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pyshoz, m) {
    auto node =
        nb::class_<Node>(m, "Node").def(nb::init<const char*, size_t>());

    nb::class_<Publisher>(m, "Publisher", node)
        .def(nb::init<const char*, const char*, size_t>())
        .def("put", &Publisher::put)
        .def("copy_tensor", &Publisher::copyTensor)
        .def("malloc", &Publisher::malloc, "ndim"_a, "shape"_a, "dtype_tup"_a,
             "clean"_a = true, nb::rv_policy::reference);

    nb::class_<Subscriber>(m, "Subscriber", node)
        .def(nb::init<const char*, const char*, size_t>())
        .def("sub", &Subscriber::sub);
}