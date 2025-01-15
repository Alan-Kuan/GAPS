#include <cstddef>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "zenoh_wrapper.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pyshoz, m) {
    nb::class_<ZenohSession>(m, "ZenohSession").def(nb::init<const char*>());

    nb::class_<Node>(m, "Node").def(nb::init<const char*, size_t>());

    nb::class_<Publisher, Node>(m, "Publisher")
        .def(nb::init<const ZenohSession&, std::string&&, size_t>())
        .def("put", &Publisher::put)
        .def("copy_tensor", &Publisher::copyTensor)
        .def("malloc", &Publisher::malloc, "shape"_a, "dtype_tup"_a,
             "clean"_a = true, nb::rv_policy::reference);

    nb::class_<Subscriber, Node>(m, "Subscriber")
        .def(nb::init<const ZenohSession&, std::string&&, size_t,
                      Subscriber::MessageHandler>());
}