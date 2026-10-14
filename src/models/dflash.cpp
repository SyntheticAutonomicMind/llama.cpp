#include "llama-impl.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache.h"
#include "models.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }
    const auto &  model_df   = static_cast<const llama_model_dflash &>(*this);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
            const ggml_tensor * gate_meta = ml->get_tensor_meta(tn(LLM_TENSOR_ATTN_GATE, "weight", i).str().c_str());
            if (gate_meta != nullptr) {
                const int64_t n_gate_out = gate_meta->ne[1];
                if (n_gate_out != n_head && n_gate_out != n_embd_head_k * n_head) {
                    GGML_ABORT(
                        "DFlash: unexpected attention gate width %lld at layer %d "
                        "(expected %lld per-head or %lld per-element)",
                        (long long) n_gate_out, i, (long long) n_head, (long long) (n_embd_head_k * n_head));
                }
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, n_gate_out }, 0);
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
        return rope_type == GGML_ROPE_TYPE_MROPE ?
                   ggml_rope_multi(ctx0, cur, pos, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base,
                                   freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                   ggml_rope_ext(ctx0, cur, pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
    };

    // KV cache injection
    ASSERT_EMBD_OR_TOKEN(ubatch);
            cur_a               = ggml_rms_norm(ctx0, cur_a, hparams.f_norm_rms_eps);
            cur_a               = ggml_mul(ctx0, cur_a, model_df.aux_norm);
            cur_a               = ggml_reshape_2d(ctx0, cur_a, n_feat * n_aux, n_tokens);
        ggml_tensor * cur = use_iswa ? build_attn(inp_attn_iswa, wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr,
                                                  layer.attn_sinks, nullptr, kq_scale, il) :
                                       build_attn(inp_attn, wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr,
                                                  layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }
        if (gated) {
            // Softplus output gate on the pre-attention hidden state, per-head
            // (broadcast over head_dim) or per-element -- same as the Laguna
            // target arch.
            ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, noise_norm);
            gate               = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            const int64_t n_tok = cur->ne[1];
            if (layer.wqkv_gate->ne[1] == n_head) {
                cur  = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tok);
                gate = ggml_reshape_3d(ctx0, gate, 1, n_head, n_tok);
        cur = build_ffn(cur, layer.ffn_up, NULL, layer.ffn_up_s, layer.ffn_gate, NULL, layer.ffn_gate_s, layer.ffn_down,
                        NULL, layer.ffn_down_s, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);