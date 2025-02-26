#include "pygaps.hpp"

#include <cstddef>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "zenoh_wrapper.hpp"

namespace nb = nanobind;

NB_MODULE(pygaps, m) {
    nb::class_<ZenohSession>(m, "ZenohSession").def(nb::init<const char*>());

    nb::class_<Node>(m, "Node").def(nb::init<const char*, size_t, int>());

    nb::class_<Publisher, Node>(m, "Publisher")
        .def(nb::init<const ZenohSession&, std::string&&, size_t, int>())
        .def("put", &Publisher::put)
        .def("empty", &Publisher::empty, nb::rv_policy::reference);

    nb::class_<Subscriber, Node>(m, "Subscriber")
        .def(nb::init<const ZenohSession&, std::string&&, size_t, int,
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