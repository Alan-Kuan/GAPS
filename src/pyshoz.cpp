#include "pyshoz.hpp"

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
        .def("malloc", &Publisher::malloc, "shape"_a, "dtype"_a,
             "clean"_a = true, nb::rv_policy::reference);

    nb::class_<Subscriber, Node>(m, "Subscriber")
        .def(nb::init<const ZenohSession&, std::string&&, size_t,
                      Subscriber::MessageHandler>());

    nb::enum_<Dtype>(m, "dtype")
        .value("int8", Dtype::int8)
        .value("int16", Dtype::int16)
        .value("int32", Dtype::int32)
        .value("int64", Dtype::int64)
        .value("uint8", Dtype::uint8)
        .value("float16", Dtype::float16)
        .value("float32", Dtype::float32)
        .export_values();
}