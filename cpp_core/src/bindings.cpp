#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Enables automatic conversion of std::optional
#include "../include/spsc_queue.h"
#include "../include/lob_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_core, m) {
    m.doc() = "Optiver MVP Deterministic Execution Core";

    // Expose the StrategyUpdate struct
    py::class_<StrategyUpdate>(m, "StrategyUpdate")
        .def(py::init<double, double, int, double>(),
             py::arg("spread_bps"), py::arg("skew_bps"), py::arg("regime_id"), py::arg("agent_uncertainty"))
        .def_readwrite("spread_bps", &StrategyUpdate::spread_bps)
        .def_readwrite("skew_bps", &StrategyUpdate::skew_bps)
        .def_readwrite("regime_id", &StrategyUpdate::regime_id)
        .def_readwrite("agent_uncertainty", &StrategyUpdate::agent_uncertainty);

    // Expose the SPSC Queue
    py::class_<SPSCQueue>(m, "SPSCQueue")
        .def(py::init<size_t>(), py::arg("capacity"))
        .def("push", &SPSCQueue::push, py::arg("update"),
             "Push a StrategyUpdate from Python to C++ (Non-blocking)")
        .def("pop", &SPSCQueue::pop,
             "Pop a StrategyUpdate in C++ (Non-blocking)");

    // Expose the LOBEngine to drive our simulation ticks
    py::class_<LOBEngine>(m, "LOBEngine")
        .def(py::init<SPSCQueue&, double, double>(),
             py::arg("queue"), py::arg("max_pos"), py::arg("max_spread"))
        // Expose the OS-level thread pinning method to Python
        .def("pin_thread_to_core", &LOBEngine::pin_thread_to_core, py::arg("core_id"))
        // Ingest L1 Order Book Data instead of just a mid price
        .def("on_market_tick", &LOBEngine::on_market_tick,
            py::arg("bid_price"), py::arg("bid_qty"),
            py::arg("ask_price"), py::arg("ask_qty"));
}
