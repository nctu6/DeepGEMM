# Peer access between two GPUs (run with CUDA_VISIBLE_DEVICES=6,7): `can_device_access_peer`, the copy bandwidth each
# way, and a kernel on GPU 0 loading a GPU 1 tensor (what an EP=2 mega MoE over CUDA IPC would do)
import time
import torch
import triton
import triton.language as tl


@triton.jit
def _copy(src, dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + i, tl.load(src + i, mask=i < n), mask=i < n)


if __name__ == '__main__':
    print('peer access 0->1, 1->0:', torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0))
    n = 256 << 20
    a = torch.full((n, ), 1, dtype=torch.uint8, device='cuda:0')
    b = torch.full((n, ), 2, dtype=torch.uint8, device='cuda:1')
    for name, (src, dst) in {'0->1': (a, b), '1->0': (b, a)}.items():
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            dst.copy_(src)
        torch.cuda.synchronize()
        print(f'{name}: {n * 20 / (time.time() - start) / 1e9:.1f} GB/s')
    # The copies above overwrote `b`: refill it, then read it from a kernel on GPU 0
    b.fill_(3)
    torch.cuda.synchronize()
    torch.cuda.set_device(0)
    out = torch.empty(n, dtype=torch.uint8, device='cuda:0')
    _copy[(n // 4096, )](b, out, n, BLOCK=4096)
    torch.cuda.synchronize()
    print('kernel peer load:', 'OK' if bool((out == 3).all()) else 'WRONG')
