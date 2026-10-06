#include "llama.h"
#include "llama-cpp.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-model.h"
#include "../src/llama-moe-cache.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("MoE cache regression failed");
    }
}

struct padded_gpu {
    ggml_backend_device device = {};
    ggml_backend_buffer_type buft = {};
    ggml_backend backend = {};
    size_t uploaded = 0;
    size_t allocated = 0;
    bool fail_alloc = false;

    padded_gpu() {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "padded GPU"; };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_memory = [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 1ULL << 30; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) { return &((padded_gpu *) dev->context)->buft; };
        device.iface.get_props = [](ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
            *props = {};
            props->name = "padded GPU";
            props->description = "padded GPU";
            props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
            props->memory_free = props->memory_total = 1ULL << 30;
            props->caps.async = false;
            (void) dev;
        };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            return type == &((padded_gpu *) dev->context)->buft;
        };
        buft.device = &device;
        buft.context = this;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "padded GPU"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return (size_t) 128; };
        buft.iface.get_alloc_size = [](ggml_backend_buffer_type_t, const ggml_tensor * t) {
            return ggml_nbytes(t) + (ggml_is_quantized(t->type) ? (size_t) 384 : 0);
        };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
            auto & gpu = *(padded_gpu *) type->context;
            if (gpu.fail_alloc) {
                return nullptr;
            }
            void * data = std::malloc(size + 127);
            if (!data) {
                return nullptr;
            }
            ggml_backend_buffer_i iface = {};
            iface.free_buffer = [](ggml_backend_buffer_t b) { std::free(b->context); };
            iface.get_base = [](ggml_backend_buffer_t b) {
                return (void *) (((uintptr_t) b->context + 127) & ~(uintptr_t) 127);
            };
            iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
                ((padded_gpu *) b->buft->context)->uploaded += size;
                std::memcpy((uint8_t *) t->data + offset, data, size);
            };
            iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t offset, size_t size) {
                std::memcpy(data, (const uint8_t *) t->data + offset, size);
            };
            iface.clear = [](ggml_backend_buffer_t b, uint8_t value) {
                std::memset(ggml_backend_buffer_get_base(b), value, b->size);
            };
            gpu.allocated += size;
            return ggml_backend_buffer_init(type, iface, data, size);
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "padded GPU"; };
    }
};

static void fill_weights(ggml_tensor * t, void *) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    for (int64_t expert = 0; expert < t->ne[2]; ++expert) {
        std::memset(bytes.data() + expert*t->nb[2], (int) (expert + 1 + t->name[4]*7), t->nb[2]);
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static llama_model_ptr make_model(padded_gpu & gpu) {
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture", "qwen2moe");
    gguf_set_val_u32(meta.get(), "qwen2moe.context_length", 128);
    gguf_set_val_u32(meta.get(), "qwen2moe.embedding_length", 32);
    gguf_set_val_u32(meta.get(), "qwen2moe.block_count", 3);
    gguf_set_val_u32(meta.get(), "qwen2moe.feed_forward_length", 32);
    gguf_set_val_u32(meta.get(), "qwen2moe.expert_feed_forward_length", 32);
    gguf_set_val_u32(meta.get(), "qwen2moe.expert_shared_feed_forward_length", 32);
    gguf_set_val_u32(meta.get(), "qwen2moe.attention.head_count", 1);
    gguf_set_val_u32(meta.get(), "qwen2moe.attention.head_count_kv", 1);
    gguf_set_val_f32(meta.get(), "qwen2moe.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(meta.get(), "qwen2moe.expert_count", 4);
    gguf_set_val_u32(meta.get(), "qwen2moe.expert_used_count", 2);
    gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "test");
    const char * tokens[] = { "a", "b", "c", "d" };
    gguf_set_arr_str(meta.get(), "tokenizer.ggml.tokens", tokens, 4);
    ggml_context_ptr ctx(ggml_init({ 9*ggml_tensor_overhead(), nullptr, true }));
    for (int layer = 0; layer < 3; ++layer) {
        for (const char * projection : { "gate", "up", "down" }) {
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_0, 32, 32, 4);
            ggml_format_name(t, "blk.%d.ffn_%s_exps.weight", layer, projection);
            gguf_add_tensor(meta.get(), t);
        }
    }
    ggml_backend_dev_t devices[] = { &gpu.device, nullptr };
    llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.devices = devices;
    params.n_gpu_layers = 99;
    params.tensor_buft_overrides = overrides;
    params.use_extra_bufts = false;
    llama_model_ptr model(llama_model_init_from_user(meta.get(), fill_weights, nullptr, params));
    require(model != nullptr);
    return model;
}

static void check_experts(llama_moe_cache & cache, padded_gpu & gpu, ggml_context * ctx,
                          ggml_tensor * source, const std::vector<int32_t> & ids) {
    ggml_tensor * id_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor node = {};
    node.op = GGML_OP_MUL_MAT_ID;
    node.src[0] = source;
    node.src[2] = id_tensor;
    ggml_tensor * cached = nullptr;
    void * entry = nullptr;
    require(llama_moe_cache::sched_resolve(&cache, &node, &gpu.backend, &cached, &entry));
    const int32_t * remapped = nullptr;
    require(llama_moe_cache::sched_prepare(&cache, entry, ids.data(), ids.size(), &remapped));
    for (size_t i = 0; i < ids.size(); ++i) {
        std::vector<uint8_t> bytes(source->nb[2]);
        ggml_backend_tensor_get(cached, bytes.data(), remapped[i]*cached->nb[2], bytes.size());
        require(std::memcmp(bytes.data(), (const uint8_t *) source->data + ids[i]*source->nb[2], bytes.size()) == 0);
    }
}

int main() {
    ggml_backend_load_all();
    llama_backend_init();
    padded_gpu gpu;
    auto model = make_model(gpu);
    ggml_context_ptr ctx(ggml_init({ 64*ggml_tensor_overhead(), nullptr, true }));
    const size_t budget = 3*2176;
    const size_t before_cache = gpu.allocated;
    llama_moe_cache cache(*model, &gpu.backend, &gpu.buft, budget);
    require(cache.memory_breakdown().at(&gpu.buft) == budget);
    require(gpu.allocated - before_cache == budget);
    const size_t uploads = gpu.uploaded;
    llama_moe_cache::sched_begin(&cache);
    check_experts(cache, gpu, ctx.get(), model->layers[0].ffn_gate_exps, { 0, 1 });
    require(gpu.uploaded - uploads == 6*576);
    const size_t warm = gpu.uploaded;
    cache.reset_stats();
    check_experts(cache, gpu, ctx.get(), model->layers[0].ffn_up_exps, { 0, 1 });
    require(gpu.uploaded == warm);
    llama_moe_cache::sched_begin(&cache);
    check_experts(cache, gpu, ctx.get(), model->layers[1].ffn_gate_exps, { 0, 1 });
    require(gpu.uploaded - warm == 6*576);
    llama_moe_cache::sched_begin(&cache);
    check_experts(cache, gpu, ctx.get(), model->layers[1].ffn_gate_exps, { 2, 1 });
    llama_moe_cache::sched_begin(&cache);
    const size_t before_duplicate = gpu.uploaded;
    check_experts(cache, gpu, ctx.get(), model->layers[1].ffn_gate_exps, { 1, 1 });
    require(gpu.uploaded == before_duplicate);
    llama_moe_cache::sched_begin(&cache);
    check_experts(cache, gpu, ctx.get(), model->layers[0].ffn_gate_exps, { 0, 3 });
    ggml_tensor ids = {};
    ids.ne[0] = 2;
    ids.ne[1] = 1;
    ggml_tensor node = {};
    node.op = GGML_OP_MUL_MAT_ID;
    node.src[0] = model->layers[0].ffn_gate_exps;
    node.src[2] = &ids;
    ggml_tensor * cached = nullptr;
    void * entry = nullptr;
    require(llama_moe_cache::sched_resolve(&cache, &node, &gpu.backend, &cached, &entry));
    const int32_t * remapped = nullptr;
    const int32_t bad_ids[] = { -1, 4 };
    require(!llama_moe_cache::sched_prepare(&cache, entry, bad_ids, 2, &remapped));
    require(!llama_moe_cache::sched_prepare(&cache, entry, nullptr, 0, &remapped));
    ids.ne[1] = 33;
    require(!llama_moe_cache::sched_resolve(&cache, &node, &gpu.backend, &cached, &entry));
    model->hparams.no_alloc = true;
    const size_t allocated = gpu.allocated;
    llama_moe_cache measured(*model, &gpu.backend, &gpu.buft, budget);
    require(measured.memory_breakdown().at(&gpu.buft) == budget);
    require(gpu.allocated == allocated);
    model->hparams.no_alloc = false;
    bool rejected = false;
    try {
        llama_moe_cache too_small(*model, &gpu.backend, &gpu.buft, budget - 1);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected);
    gpu.fail_alloc = true;
    rejected = false;
    try {
        llama_moe_cache failed(*model, &gpu.backend, &gpu.buft, budget);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected);
    gpu.fail_alloc = false;
    ggml_context_ptr mixed_ctx(ggml_init({ 3*ggml_tensor_overhead(), nullptr, true }));
    ggml_tensor * mixed[] = {
        ggml_new_tensor_3d(mixed_ctx.get(), GGML_TYPE_F32, 32, 16, 4),
        ggml_new_tensor_3d(mixed_ctx.get(), GGML_TYPE_F32, 32, 16, 4),
        ggml_new_tensor_3d(mixed_ctx.get(), GGML_TYPE_F32, 16, 32, 4),
    };
    ggml_backend_buffer_ptr mixed_buf(ggml_backend_alloc_ctx_tensors_from_buft(mixed_ctx.get(), ggml_backend_cpu_buffer_type()));
    require(mixed_buf != nullptr);
    ggml_backend_buffer_set_usage(mixed_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (ggml_tensor * t : mixed) {
        ggml_set_name(t, "blk.2.mixed");
        fill_weights(t, nullptr);
    }
    const llama_layer saved = model->layers[2];
    model->layers[2].ffn_gate_exps = mixed[0];
    model->layers[2].ffn_up_exps = mixed[1];
    model->layers[2].ffn_down_exps = mixed[2];
    llama_moe_cache grouped(*model, &gpu.backend, &gpu.buft, 32768);
    require(grouped.memory_breakdown().at(&gpu.buft) <= 32768);
    llama_moe_cache::sched_begin(&grouped);
    check_experts(grouped, gpu, ctx.get(), model->layers[0].ffn_gate_exps, { 0, 1 });
    llama_moe_cache::sched_begin(&grouped);
    check_experts(grouped, gpu, ctx.get(), mixed[0], { 0, 1 });
    const size_t before_group_hit = gpu.uploaded;
    llama_moe_cache::sched_begin(&grouped);
    check_experts(grouped, gpu, ctx.get(), model->layers[0].ffn_gate_exps, { 0, 1 });
    require(gpu.uploaded == before_group_hit);
    model->layers[2] = saved;
    return 0;
}
