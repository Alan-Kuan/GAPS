#include "pygaps.hpp"

#include <cstddef>

#include <iceoryx_hoofs/log/logmanager.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "profiling.hpp"

namespace nb = nanobind;

NB_MODULE(pygaps, m) {
    // simple Iceoryx wrapper
    m.def("init_runtime", [](const char* name) {
        iox::RuntimeName_t runtime_name(iox::cxx::TruncateToCapacity, name);
        iox::runtime::PoshRuntime::initRuntime(runtime_name);
    });
    m.def("turn_off_logging", []() {
        iox::log::LogManager::GetLogManager().SetDefaultLogLevel(
            iox::log::LogLevel::kOff);
    });

    nb::class_<Node>(m, "Node").def(nb::init<const char*, size_t, int>());

    nb::class_<Publisher, Node>(m, "Publisher")
        .def(nb::init<const char*, size_t, int>())
        .def("put", &Publisher::put)
        .def("empty", &Publisher::empty, nb::rv_policy::reference);

    nb::class_<Subscriber, Node>(m, "Subscriber")
        .def(nb::init<const char*, size_t, int, Subscriber::MessageHandler>());

    nb::enum_<Dtype>(m, "dtype")
        .value("int8", Dtype::int8)
        .value("int16", Dtype::int16)
        .value("int32", Dtype::int32)
        .value("int64", Dtype::int64)
        .value("uint8", Dtype::uint8)
        .value("float16", Dtype::float16)
        .value("float32", Dtype::float32)
        .export_values();

#ifdef PROFILING
    nb::module_ m2 = m.def_submodule("profiling");
    m2.def("dump_records", &profiling::dump_records);
#endif
}