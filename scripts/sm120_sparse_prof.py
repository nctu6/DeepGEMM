"""One sparse MQA invocation for ncu profiling (warms up, then runs once under the profiler)."""
import sys
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
sys.path.insert(0, 'tests')
from test_attention import make_sparse_kv_block_indices, to_mqa_weights

def main():
    deep_gemm.set_pdl(False)
    torch.manual_seed(0)
    num_heads, head_dim, sparse_block_kv, nmsb = 32, 128, 8, 512
    num_q_tokens, kv_len = 2048, 8192
    context_lens = [kv_len]*num_q_tokens; context_starts=[0]*num_q_tokens
    q_fp,q_sf = per_token_cast_to_fp8(torch.randn((num_q_tokens,num_heads,head_dim),device='cuda',dtype=torch.bfloat16).view(-1,head_dim),use_ue8m0=True,gran_k=32,use_packed_ue8m0=True)
    q=q_fp.view(num_q_tokens,num_heads,head_dim),q_sf.view(num_q_tokens,num_heads)
    weights=to_mqa_weights(torch.randn((num_q_tokens,num_heads),device='cuda',dtype=torch.bfloat16))
    si,_=make_sparse_kv_block_indices(context_lens,None,sparse_block_kv,nmsb,0,context_starts)
    kv_fp,kv_sf=per_token_cast_to_fp8(torch.randn((kv_len,head_dim),device='cuda',dtype=torch.bfloat16),use_ue8m0=True,gran_k=32,use_packed_ue8m0=True)
    kv=kv_fp,kv_sf.view(kv_len)
    starts=torch.tensor(context_starts,device='cuda',dtype=torch.int32); ends=torch.tensor(context_lens,device='cuda',dtype=torch.int32)
    meta=deep_gemm.get_sparse_mqa_logits_metadata(cu_seq_len_k_start=starts,cu_seq_len_k_end=ends,num_kv_tokens=kv_len,sparse_kv_block_indices=si,qk_dtype=q[0].dtype,sparse_block_kv=sparse_block_kv)
    run=lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(q=q,kv=kv,weights=weights,metadata=meta,num_max_sparse_blocks=nmsb,sparse_block_kv=sparse_block_kv)
    for _ in range(5): run()
    torch.cuda.synchronize()
    run(); torch.cuda.synchronize()

if __name__=='__main__': main()
