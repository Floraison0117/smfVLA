#!/usr/bin/env python3
"""
把 RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT 的 model.safetensors 完整转成 openpi JAX 格式。

修正点（相比旧版）：
- RLinf **微调了 VLM**（VLM 权重与 smf_base 不同，maxdiff~0.22），所以必须转换 RLinf 的
  完整 VLM（视觉塔+语言模型），不能用 smf_base 的 VLM。
- 用 openpi 官方 jax<->pytorch 映射（examples/convert_jax_model_to_pytorch.py）的逆。
- RLinf 源缺 VLM embed_tokens -> 从 smf_base 补（embed_tokens 在所有 pi0.5 里冻结一致）。
- 全模型 round-trip 自检：smf_base jax->pt(openpi 正向)->jax(本脚本逆向) 必须复原。
"""
import sys, pathlib, importlib.util
import numpy as np
sys.path.insert(0, "/root/autodl-tmp/openpi/src")
sys.path.insert(0, "/root/autodl-tmp/eval/scripts")
import jax.numpy as jnp
import orbax.checkpoint as ocp
from safetensors import safe_open
import flax.traverse_util as tu
from openpi.models import model as openpi_model

SMF = pathlib.Path("/root/autodl-tmp/checkpoints/smf_base/pi05_libero")
RLINF = pathlib.Path("/root/autodl-tmp/checkpoints/pi05_calvin_pt/model.safetensors")
OUT = pathlib.Path("/root/autodl-tmp/checkpoints/pi05_calvin_corrected")


def load_jax(ck): return openpi_model.restore_params(pathlib.Path(ck)/"params", dtype=jnp.float32)
def load_pt(p):
    import torch
    w={}
    with safe_open(str(p), framework="pt") as f:
        for k in f.keys():
            t=f.get_tensor(k)
            if t.dtype==torch.bfloat16: t=t.to(torch.float32)
            w[k]=t.numpy()
    return w

# 加载 openpi 正向转换（权威映射）
_spec=importlib.util.spec_from_file_location("cvrt","/root/autodl-tmp/openpi/examples/convert_jax_model_to_pytorch.py")
cvrt=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cvrt)


# ── 视觉塔 pt->jax（逆 openpi slice_paligemma vision）──
def vision_pt_to_jax(pt, V):
    nH,HD,H,inter,L = V["nH"],V["HD"],V["H"],V["inter"],V["L"]
    VV="paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    def stk(a): return np.stack([pt[f"{VV}.encoder.layers.{i}.{a}"] for i in range(L)])
    ln0w=stk("layer_norm1.weight"); ln0b=stk("layer_norm1.bias")
    ln1w=stk("layer_norm2.weight"); ln1b=stk("layer_norm2.bias")
    fc1w=stk("mlp.fc1.weight"); fc1b=stk("mlp.fc1.bias")
    fc2w=stk("mlp.fc2.weight"); fc2b=stk("mlp.fc2.bias")
    kw=stk("self_attn.k_proj.weight"); qb=stk("self_attn.q_proj.bias")
    qw=stk("self_attn.q_proj.weight"); kb=stk("self_attn.k_proj.bias")
    vw=stk("self_attn.v_proj.weight"); vb=stk("self_attn.v_proj.bias")
    ow=stk("self_attn.out_proj.weight"); ob=stk("self_attn.out_proj.bias")
    J={}
    J["PaliGemma/img/embedding/kernel"]=pt[f"{VV}.embeddings.patch_embedding.weight"].transpose(2,3,1,0)
    J["PaliGemma/img/embedding/bias"]=pt[f"{VV}.embeddings.patch_embedding.bias"]
    J["PaliGemma/img/pos_embedding"]=pt[f"{VV}.embeddings.position_embedding.weight"].reshape(1,256,H)
    # layernorm (1D transpose=no-op)
    J["PaliGemma/img/Transformer/encoderblock/LayerNorm_0/scale"]=ln0w
    J["PaliGemma/img/Transformer/encoderblock/LayerNorm_0/bias"]=ln0b
    J["PaliGemma/img/Transformer/encoderblock/LayerNorm_1/scale"]=ln1w
    J["PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias"]=ln1b
    # mlp: pt fc=jax[i].T -> jax[i]=pt.T
    J["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"]=np.stack([fc1w[i].T for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"]=fc1b
    J["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"]=np.stack([fc2w[i].T for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"]=fc2b
    # attn: pt=jax[i].reshape(-1,H).T -> jax[i]=pt.T.reshape(...)
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"]=np.stack([kw[i].T.reshape(H,nH,HD) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"]=np.stack([qw[i].T.reshape(H,nH,HD) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"]=np.stack([vw[i].T.reshape(H,nH,HD) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"]=np.stack([ow[i].T.reshape(nH,HD,H) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"]=np.stack([kb[i].reshape(nH,HD) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"]=np.stack([qb[i].reshape(nH,HD) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"]=np.stack([vb[i].reshape(nH,HD) for i in range(L)])
    J["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"]=ob
    J["PaliGemma/img/Transformer/encoder_norm/scale"]=pt[f"{VV}.post_layernorm.weight"]
    J["PaliGemma/img/Transformer/encoder_norm/bias"]=pt[f"{VV}.post_layernorm.bias"]
    J["PaliGemma/img/head/kernel"]=pt["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"].T
    J["PaliGemma/img/head/bias"]=pt["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"]
    return J


# ── 语言模型(VLM) pt->jax（RMSNorm scale，逆 openpi slice_paligemma llm）──
def llm_pt_to_jax(pt, M):
    nH,HD,H,inter,L = M["nH"],M["HD"],M["H"],M["inter"],M["L"]
    LM="paligemma_with_expert.paligemma.model.language_model"
    def stk(a): return np.stack([pt[f"{LM}.layers.{i}.{a}"] for i in range(L)])
    J={}
    if f"{LM}.embed_tokens.weight" in pt:
        J["PaliGemma/llm/embedder/input_embedding"]=pt[f"{LM}.embed_tokens.weight"]
    qw=stk("self_attn.q_proj.weight"); kw=stk("self_attn.k_proj.weight"); vw=stk("self_attn.v_proj.weight"); ow=stk("self_attn.o_proj.weight")
    gw=stk("mlp.gate_proj.weight"); uw=stk("mlp.up_proj.weight"); dw=stk("mlp.down_proj.weight")
    inw=stk("input_layernorm.weight"); pw=stk("post_attention_layernorm.weight")
    J["PaliGemma/llm/layers/attn/q_einsum/w"]=np.stack([qw[i].reshape(nH,HD,H).transpose(0,2,1) for i in range(L)])
    J["PaliGemma/llm/layers/attn/kv_einsum/w"]=np.stack([np.stack([[kw[i].T],[vw[i].T]]) for i in range(L)])
    J["PaliGemma/llm/layers/attn/attn_vec_einsum/w"]=np.stack([ow[i].T.reshape(nH,HD,H) for i in range(L)])
    J["PaliGemma/llm/layers/mlp/gating_einsum"]=np.stack([np.stack([gw[i].T,uw[i].T]) for i in range(L)])
    J["PaliGemma/llm/layers/mlp/linear"]=np.stack([dw[i].T for i in range(L)])
    J["PaliGemma/llm/layers/pre_attention_norm/scale"]=inw
    J["PaliGemma/llm/layers/pre_ffw_norm/scale"]=pw
    J["PaliGemma/llm/final_norm/scale"]=pt[f"{LM}.norm.weight"]
    return J


# ── 专家 pt->jax（pi05 Dense 自适应 norm，逆 openpi slice_gemma）──
def expert_pt_to_jax(pt, M):
    nH,HD,H,L = M["nH"],M["HD"],M["H"],M["L"]
    g="paligemma_with_expert.gemma_expert.model"
    def stk(a): return np.stack([pt[f"{g}.layers.{i}.{a}"] for i in range(L)])
    J={}
    qw=stk("self_attn.q_proj.weight"); kw=stk("self_attn.k_proj.weight"); vw=stk("self_attn.v_proj.weight"); ow=stk("self_attn.o_proj.weight")
    gw=stk("mlp.gate_proj.weight"); uw=stk("mlp.up_proj.weight"); dw=stk("mlp.down_proj.weight")
    inb=stk("input_layernorm.dense.bias"); inw=stk("input_layernorm.dense.weight")
    pb=stk("post_attention_layernorm.dense.bias"); pw=stk("post_attention_layernorm.dense.weight")
    J["PaliGemma/llm/layers/attn/q_einsum_1/w"]=np.stack([qw[i].reshape(nH,HD,H).transpose(0,2,1) for i in range(L)])
    J["PaliGemma/llm/layers/attn/kv_einsum_1/w"]=np.stack([np.stack([[kw[i].T],[vw[i].T]]) for i in range(L)])
    J["PaliGemma/llm/layers/attn/attn_vec_einsum_1/w"]=np.stack([ow[i].T.reshape(nH,HD,H) for i in range(L)])
    J["PaliGemma/llm/layers/mlp_1/gating_einsum"]=np.stack([np.stack([gw[i].T,uw[i].T]) for i in range(L)])
    J["PaliGemma/llm/layers/mlp_1/linear"]=np.stack([dw[i].T for i in range(L)])
    J["PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias"]=inb
    J["PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel"]=np.stack([inw[i].T for i in range(L)])
    J["PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias"]=pb
    J["PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel"]=np.stack([pw[i].T for i in range(L)])
    J["PaliGemma/llm/final_norm_1/Dense_0/kernel"]=pt[f"{g}.norm.dense.weight"].T
    J["PaliGemma/llm/final_norm_1/Dense_0/bias"]=pt[f"{g}.norm.dense.bias"]
    return J


def projection_pt_to_jax(pt):
    J={}
    for k in ["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]:
        J[f"{k}/kernel"]=pt[f"{k}.weight"].T
        J[f"{k}/bias"]=pt[f"{k}.bias"]
    return J


def get_dims(smf):
    P=smf["PaliGemma"]
    q=P["llm"]["layers"]["attn"]["q_einsum_1"]["w"].shape            # (L,nH,H,HD) expert
    L,nH,H,HD=q
    vk=P["img"]["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["key"]["kernel"].shape
    e={"L":L,"nH":nH,"H":H,"HD":HD,"inter":P["llm"]["layers"]["mlp_1"]["linear"].shape[1]}
    v={"L":vk[0],"H":vk[1],"nH":vk[2],"HD":vk[3],
       "inter":P["img"]["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_0"]["kernel"].shape[2]}
    l={"L":L,"nH":nH,"H":P["llm"]["layers"]["attn"]["q_einsum"]["w"].shape[2],
       "HD":P["llm"]["layers"]["attn"]["q_einsum"]["w"].shape[3],"inter":P["llm"]["layers"]["mlp"]["linear"].shape[1]}
    return e,v,l


def roundtrip_full(smf, e,v,l):
    """smf 全模型 jax->pt(openpi 正向)->jax(本脚本逆向)->比对原 smf。"""
    P=smf["PaliGemma"]
    flat=tu.flatten_dict(P, sep="/")
    sd={k+"/value":np.asarray(x) for k,x in flat.items()}
    pg_cfg=type("c",(),{"vision_config":type("v",(),{"hidden_size":v["H"],"num_hidden_layers":v["L"],"num_attention_heads":v["nH"]})(),
                        "text_config":type("t",(),{"hidden_size":l["H"],"num_hidden_layers":l["L"],"num_attention_heads":l["nH"],"head_dim":l["HD"]})()})()
    pg_pt, expert_jax = cvrt.slice_paligemma_state_dict(sd, pg_cfg)
    e_cfg=type("c",(),{"width":e["H"],"depth":e["L"],"num_heads":e["nH"],"head_dim":e["HD"]})()
    expert_pt = cvrt.slice_gemma_state_dict(expert_jax, e_cfg, num_expert=1, checkpoint_dir="...pi05...", pi05=True)
    full_pt = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in {**pg_pt, **expert_pt}.items()}
    # 我的逆向
    back={}; back.update(vision_pt_to_jax(full_pt, v)); back.update(llm_pt_to_jax(full_pt, l)); back.update(expert_pt_to_jax(full_pt, e))
    ok=True
    for k,val in back.items():
        ok_key=k.replace("PaliGemma/","",1)
        if ok_key not in flat: print("  [MISS]",ok_key); ok=False; continue
        if not np.allclose(np.asarray(val),np.asarray(flat[ok_key]),atol=1e-3):
            print(f"  [MISMATCH] {ok_key}: {np.abs(np.asarray(val)-np.asarray(flat[ok_key])).max():.2e}"); ok=False
    return ok


def main():
    print("加载 smf_base ..."); smf=load_jax(SMF)
    e,v,l=get_dims(smf); print(f"专家:{e}\n视觉:{v}\n语言:{l}")
    print("\n全模型 round-trip 自检 ...")
    if not roundtrip_full(smf,e,v,l): print("FAILED"); sys.exit(1)
    print("PASSED\n")
    print("加载 RLinf safetensors ..."); pt=load_pt(RLINF); print(f"  {len(pt)} keys")
    print("转换 RLinf 完整模型 (视觉+语言+专家+投影) ...")
    J={}; J.update(vision_pt_to_jax(pt,v)); J.update(llm_pt_to_jax(pt,l)); J.update(expert_pt_to_jax(pt,e))
    proj=projection_pt_to_jax(pt)
    # embed_tokens 缺失 -> 从 smf_base 补
    if "PaliGemma/llm/embedder/input_embedding" not in J:
        J["PaliGemma/llm/embedder/input_embedding"]=np.asarray(smf["PaliGemma"]["llm"]["embedder"]["input_embedding"])
        print("  补 embedder/input_embedding from smf_base")
    print("组装 checkpoint ...")
    # J 覆盖全部 PaliGemma 权重键(视觉+语言+专家+embedder)，unflatten 成嵌套结构
    pg_flat = {tuple(k.replace("PaliGemma/", "", 1).split("/")): jnp.asarray(v) for k, v in J.items()}
    P = tu.unflatten_dict(pg_flat)
    merged = {"PaliGemma": P}
    for k, val in proj.items():
        name, leaf = k.split("/")
        merged.setdefault(name, {})[leaf] = jnp.asarray(val)
    print(f"保存到 {OUT} ..."); OUT.mkdir(parents=True,exist_ok=True);
    if (OUT/"params").exists():
        import shutil; shutil.rmtree(OUT/"params")
    with ocp.PyTreeCheckpointer() as ck: ck.save(str(OUT/"params"),{"params":merged})
    import shutil,json
    shutil.copy("/root/autodl-tmp/checkpoints/pi05_calvin_pt/norm_stats.json",OUT/"norm_stats.json")
    json.dump({"config_name":"pi05_libero","source":"RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT full model"},open(OUT/"metadata.json","w"),indent=2)
    print(f"DONE: {OUT}")


if __name__=="__main__":
    main()
