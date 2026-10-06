#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct delayed_backend {
    ggml_backend_t cpu;
    ggml_backend backend{};
    ggml_backend_device device{};
    ggml_backend_buffer_type buft{};
    std::vector<std::function<void()>> pending;
    int syncs = 0;

    delayed_backend(ggml_backend_t cpu_backend) : cpu(cpu_backend) {
        backend.context = this;
        backend.device = &device;
        backend.iface.get_name = [](ggml_backend_t) { return "delayed"; };
        backend.iface.synchronize = [](ggml_backend_t b) {
            auto & self = *static_cast<delayed_backend *>(b->context);
            ++self.syncs;
            auto work = std::move(self.pending);
            self.pending.clear();
            for (auto & op : work) {
                op();
            }
        };
        backend.iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * src, size_t offset, size_t size) {
            auto & self = *static_cast<delayed_backend *>(b->context);
            self.pending.emplace_back([=] { ggml_backend_tensor_set(t, src, offset, size); });
        };
        backend.iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * graph) {
            auto & self = *static_cast<delayed_backend *>(b->context);
            auto nodes = std::vector<ggml_tensor *>(graph->nodes, graph->nodes + graph->n_nodes);
            const ggml_cgraph saved = *graph;
            self.pending.emplace_back([&self, nodes, saved]() mutable {
                ggml_cgraph copy = saved;
                copy.nodes = nodes.data();
                GGML_ASSERT(ggml_backend_graph_compute(self.cpu, &copy) == GGML_STATUS_SUCCESS);
            });
            return GGML_STATUS_SUCCESS;
        };
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "delayed"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) { return &static_cast<delayed_backend *>(d->context)->buft; };
        device.iface.supports_buft = [](ggml_backend_dev_t d, ggml_backend_buffer_type_t t) {
            return t == &static_cast<delayed_backend *>(d->context)->buft;
        };
        device.iface.supports_op = [](ggml_backend_dev_t d, const ggml_tensor * op) {
            return ggml_backend_supports_op(static_cast<delayed_backend *>(d->context)->cpu, op);
        };
        device.iface.offload_op = [](ggml_backend_dev_t, const ggml_tensor * op) { return op->op == GGML_OP_MUL_MAT_ID; };
        buft.context = this;
        buft.device = &device;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "delayed-buffer"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(64); };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
            auto * self = static_cast<delayed_backend *>(type->context);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(self->cpu, size);
            if (buffer) {
                buffer->buft = type;
            }
            return buffer;
        };
    }
};

struct test_cache {
    delayed_backend & backend;
    ggml_tensor * weights[2];
    ggml_tensor * banks[2];
    int32_t remapped = -1;
    int begins = 0;
    int prepares = 0;
    bool fail = false;

    static bool resolve(void * data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** bank, void ** entry) {
        auto & self = *static_cast<test_cache *>(data);
        if (backend != &self.backend.backend) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            if (node->src[0] == self.weights[i]) {
                *bank = self.banks[i];
                *entry = data;
                return true;
            }
        }
        return false;
    }

    static void begin(void * data) {
        auto & self = *static_cast<test_cache *>(data);
        GGML_ASSERT(self.backend.pending.empty());
        self.remapped = -1;
        ++self.begins;
    }

    static bool prepare(void * data, void * entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped) {
        auto & self = *static_cast<test_cache *>(data);
        GGML_ASSERT(entry == data && n_ids == 1 && ids[0] >= 0 && ids[0] < 3);
        ++self.prepares;
        if (self.fail) {
            return false;
        }
        self.remapped = 0;
        for (int i = 0; i < 2; ++i) {
            ggml_backend_tensor_set_async(&self.backend.backend, self.banks[i],
                    static_cast<char *>(self.weights[i]->data) + ids[0] * self.weights[i]->nb[2], 0, self.weights[i]->nb[2]);
        }
        *remapped = &self.remapped;
        return true;
    }
};

static ggml_context * make_context() {
    return ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true});
}

static void expect_assert(const std::function<void()> & operation) {
    const auto previous = ggml_set_abort_callback([](const char * message) { throw std::runtime_error(message); });
    bool rejected = false;
    try {
        operation();
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("sched->is_reset && !sched->is_alloc") != std::string::npos;
    }
    ggml_set_abort_callback(previous);
    GGML_ASSERT(rejected);
}

struct test_graph {
    ggml_context * ctx = make_context();
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_tensor * outputs[2]{};

    test_graph(test_cache & cache, ggml_tensor * input, ggml_tensor * ids, bool repeated, ggml_tensor * other_ids = nullptr) {
        outputs[0] = ggml_mul_mat_id(ctx, cache.weights[0], input, ids);
        outputs[1] = ggml_mul_mat_id(ctx, cache.weights[repeated ? 0 : 1], input, other_ids ? other_ids : ids);
        for (auto * output : outputs) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
        }
    }

    ~test_graph() { ggml_free(ctx); }

    void check(float first, float second) {
        const float expected[] = { first, second };
        for (int i = 0; i < 2; ++i) {
            float result;
            ggml_backend_tensor_get(outputs[i], &result, 0, sizeof(result));
            GGML_ASSERT(std::fabs(result - expected[i]) < 1e-6f);
        }
    }
};

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);
    delayed_backend delayed(cpu);
    delayed_backend unused(cpu);
    ggml_backend_t backends[] = { &delayed.backend, &unused.backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, 128, false, true);
    ggml_context * host = make_context();
    ggml_context * data = make_context();
    ggml_context * device = make_context();
    test_cache cache{delayed, {}, {}};
    for (int i = 0; i < 2; ++i) {
        cache.weights[i] = ggml_new_tensor_3d(host, GGML_TYPE_F32, 2, 1, 3);
        cache.banks[i] = ggml_new_tensor_3d(device, GGML_TYPE_F32, 2, 1, 2);
    }
    ggml_tensor * input = ggml_new_tensor_3d(data, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(data, GGML_TYPE_I32, 1, 1);
    ggml_tensor * other_ids = ggml_new_tensor_2d(data, GGML_TYPE_I32, 1, 1);
    ggml_set_input(input);
    ggml_set_input(ids);
    ggml_set_input(other_ids);
    ggml_backend_buffer_t host_buffer = ggml_backend_alloc_ctx_tensors(host, cpu);
    ggml_backend_buffer_t data_buffer = ggml_backend_alloc_ctx_tensors(data, cpu);
    ggml_backend_buffer_t device_buffer = ggml_backend_alloc_ctx_tensors(device, &delayed.backend);
    GGML_ASSERT(host_buffer && data_buffer && device_buffer);
    ggml_backend_buffer_set_usage(host_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const float first[] = {1, 2, 3, 4, 5, 6};
    const float second[] = {2, 3, 4, 5, 6, 7};
    const float x[] = {2, 3};
    const int32_t id = 2;
    const int32_t other_id = 1;
    ggml_backend_tensor_set(cache.weights[0], first, 0, sizeof(first));
    ggml_backend_tensor_set(cache.weights[1], second, 0, sizeof(second));
    ggml_backend_tensor_set(input, x, 0, sizeof(x));
    ggml_backend_tensor_set(ids, &id, 0, sizeof(id));
    ggml_backend_tensor_set(other_ids, &other_id, 0, sizeof(other_id));
    auto enable = [&](test_cache * state) {
        ggml_backend_sched_set_moe_cache(sched, &delayed.backend, test_cache::resolve, test_cache::begin, test_cache::prepare, state);
    };
    auto disable = [&] { ggml_backend_sched_set_moe_cache(sched, nullptr, nullptr, nullptr, nullptr, nullptr); };
    enable(&cache);
    {
        test_graph graph(cache, input, ids, false);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph.graph));
        expect_assert(disable);
        expect_assert([&] { enable(&cache); });
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(cache.prepares == 1 && !delayed.pending.empty());
        const int unused_syncs = unused.syncs;
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(cache.prepares == 2 && cache.begins == 2);
        GGML_ASSERT(unused.syncs == unused_syncs);
        ggml_backend_sched_synchronize(sched);
        graph.check(28, 33);
        cache.fail = true;
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph.graph) == GGML_STATUS_FAILED);
        cache.fail = false;
        ggml_backend_sched_reset(sched);
        disable();
    }
    {
        test_graph graph(cache, input, ids, false);
        const int prepares = cache.prepares;
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph.graph) == GGML_STATUS_SUCCESS);
        graph.check(28, 33);
        GGML_ASSERT(cache.prepares == prepares);
        ggml_backend_sched_reset(sched);
    }
    test_cache replacement{delayed, {cache.weights[0], cache.weights[1]}, {cache.banks[0], cache.banks[1]}};
    enable(&replacement);
    {
        test_graph graph(replacement, input, ids, false);
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(!delayed.pending.empty());
        ggml_backend_sched_reset(sched);
        disable();
        GGML_ASSERT(delayed.pending.empty());
        graph.check(28, 33);
    }
    enable(&replacement);
    {
        test_graph graph(replacement, input, ids, false);
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(!delayed.pending.empty());
        ggml_backend_sched_reset(sched);
        enable(&cache);
        GGML_ASSERT(delayed.pending.empty());
        graph.check(28, 33);
    }
    enable(&replacement);
    for (bool differing_ids : { false, true }) {
        test_graph graph(replacement, input, ids, true, differing_ids ? other_ids : nullptr);
        const int prepares = replacement.prepares;
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph.graph) == GGML_STATUS_SUCCESS);
        for (auto * output : graph.outputs) {
            GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, output) == cpu);
        }
        graph.check(28, differing_ids ? 18 : 28);
        GGML_ASSERT(replacement.prepares == prepares);
        ggml_backend_sched_reset(sched);
    }
    {
        test_graph graph(replacement, input, ids, false);
        ggml_backend_sched_split_graph(sched, graph.graph);
        expect_assert(disable);
        ggml_backend_sched_reset(sched);
        disable();
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(device_buffer);
    ggml_backend_buffer_free(host_buffer);
    ggml_backend_buffer_free(data_buffer);
    ggml_free(device);
    ggml_free(host);
    ggml_free(data);
    ggml_backend_free(cpu);
    std::puts("scheduler MoE cache lifecycle: PASS");
    return 0;
}
