#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "metadata.hpp"
#include "node/python_iface/publisher.hpp"
#include "node/python_iface/subscriber.hpp"

namespace nb = nanobind;

NB_MODULE(pyshoz, m) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("kHost", DeviceType::kHost)
        .value("kGPU", DeviceType::kGPU);

    nb::class_<Domain>(m, "Domain")
        .def(nb::init<DeviceType, uint16_t>())
        .def_rw("dev_type", &Domain::dev_type)
        .def_rw("dev_id", &Domain::dev_id)
        .def("__repr__", [](const Domain& self) {
            std::stringstream ss;
            ss << "<pyshoz.Domain dev_type: " << self.dev_type
               << ", dev_id: " << self.dev_id << ">";
            return ss.str();
        });

    auto node = nb::class_<Node>(m, "Node").def(
        nb::init<const char*, size_t, uint16_t>());

    nb::class_<Publisher>(m, "Publisher", node)
        .def(nb::init<const char*, const char*, const Domain&, size_t>())
        .def("put", &Publisher::put);

    nb::class_<Subscriber>(m, "Subscriber", node)
        .def(nb::init<const char*, const char*, const Domain&, size_t>())
        .def("sub", &Subscriber::sub);
}