#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"

namespace nb = nanobind;

NB_MODULE(pyshoz, m) {
    auto node =
        nb::class_<Node>(m, "Node").def(nb::init<const char*, size_t>());

    nb::class_<Publisher>(m, "Publisher", node)
        .def(nb::init<const char*, const char*, size_t>())
        .def("put", &Publisher::put);

    nb::class_<Subscriber>(m, "Subscriber", node)
        .def(nb::init<const char*, const char*, size_t>())
        .def("sub", &Subscriber::sub);
}