import numpy as np
import pyopencl as cl
from coincurve import PrivateKey
import ecdsa
import time
from ecdsa import SECP256k1
import json, gzip, time, tempfile, os
from pathlib import Path
from array import array

# ---------------------------
# Helpers
# ---------------------------
def build_kernel(context, filename="opencl/funcional.cl"):
    with open(filename, "r", encoding="utf-8") as f:
        src = f.read()
    return cl.Program(context, src).build()
   
def int_to_le8_words(k: int) -> np.ndarray:
    out = np.empty(8, dtype=np.uint32)
    for i in range(8):
        out[i] = (k >> (32 * i)) & 0xFFFFFFFF
    return out
def limbs_le_to_int(words: np.ndarray) -> int:
    acc = 0
    for i in range(8):
        acc |= int(words[i]) << (32 * i)
    return acc
def expected_xy_hex(k: int):
    sk = PrivateKey((k).to_bytes(32, 'big'))
    uncompressed = sk.public_key.format(compressed=False) # 0x04 || X || Y
    return uncompressed[1:33].hex(), uncompressed[33:].hex()


# ---------------------------
# OpenCL setup
# ---------------------------
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
program = build_kernel(ctx, "opencl/main.cl")
   
    

def run_kernel_for_ks(ks: list[int]):
    total_ks = len(ks)
    total_ks-=(total_ks%32  )
    results_expected = []
    kernel = program.point_mul_xy
    zero_array = np.zeros(total_ks * 8, dtype=np.uint32)  # Flat for x or y separately
    mf = cl.mem_flags
    buf_x = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zero_array)
    buf_y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zero_array)
    k_words = np.zeros(8 * total_ks, dtype=np.uint32)
    start_time_cpu = time.perf_counter()
    for i, k_int in enumerate(ks[:total_ks]):
        k_words[i*8:(i+1)*8] = int_to_le8_words(k_int)
        results_expected.append(expected_xy_hex(k_int))
    end_time_cpu = time.perf_counter()
    buf_k = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_words)
    cl.enqueue_copy(queue, buf_x, zero_array)
    cl.enqueue_copy(queue, buf_y, zero_array)
  
    kernel.set_args(buf_x, buf_y, buf_k)
    start_time = time.perf_counter()
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (total_ks,), (32,))
    evt.wait()
    queue.finish()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    execution_time_cpu = end_time_cpu - start_time_cpu
    x_host = np.empty_like(zero_array)
    y_host = np.empty_like(zero_array)
    cl.enqueue_copy(queue, x_host, buf_x).wait()
    cl.enqueue_copy(queue, y_host, buf_y).wait()
    queue.finish()
  
    print(f"Não-Sequencial: GPU: {execution_time} seg = {total_ks/execution_time} p/sec | CPU:  {execution_time_cpu} seg = {total_ks/execution_time_cpu} p/sec  " )
    results = []
    for i in range(total_ks):
        x_words = x_host[i*8:(i+1)*8]
        y_words = y_host[i*8:(i+1)*8]
        x_hex = ''.join(f'{w:08x}' for w in reversed(x_words))
        y_hex = ''.join(f'{w:08x}' for w in reversed(y_words))
        if x_hex != results_expected[i][0]:
            print(f"Hexadecimal  bateu: \U0001F680 {x_hex} {i}= {results_expected[i][0]}")
        results.append((x_hex, y_hex))
    return results
def run_kernel_for_seq(from_values, size):
    total_ks = len(from_values)

    work_group_size = 32
    total_ks -= (total_ks % work_group_size)
    if(total_ks <= 0):
        total_ks = work_group_size

    kernel = program.point_mul_xy_seq
    # Output size: total_ks * size * 8 for x and y each
    print(total_ks * size * 8)
    zero_array_x = np.zeros(total_ks * size * 8, dtype=np.uint32)
    zero_array_y = np.zeros(total_ks * size * 8, dtype=np.uint32)
    mf = cl.mem_flags
    buf_x = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zero_array_x)
    buf_y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zero_array_y)
    from_array = np.array(from_values, dtype=np.uint64)  # ulong
    buf_from = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=from_array)
    buf_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([size], dtype=np.uint64))
    buf_results = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(total_ks, dtype=np.uint64))  # If used
    cl.enqueue_copy(queue, buf_x, zero_array_x)
    cl.enqueue_copy(queue, buf_y, zero_array_y)
    kernel.set_args(buf_x, buf_y, buf_from, buf_size, buf_results)
    start_time = time.perf_counter()
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (total_ks,), (work_group_size,))
    evt.wait()
    queue.finish()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Sequencial: GPU time: {execution_time} sec = {total_ks * size / execution_time} points/sec")
    x_host = np.empty_like(zero_array_x)
    y_host = np.empty_like(zero_array_y)
    cl.enqueue_copy(queue, x_host, buf_x).wait()
    cl.enqueue_copy(queue, y_host, buf_y).wait()
    queue.finish()
    results = []
    print('Copia finalizada')
    '''
    for i in range(total_ks):
        for j in range(size):
            offset = (i * size + j) * 8
            x_words = x_host[offset:offset+8]
            y_words = y_host[offset:offset+8]
            x_hex = ''.join(f'{w:08x}' for w in reversed(x_words))
            y_hex = ''.join(f'{w:08x}' for w in reversed(y_words))
            results.append((x_hex, y_hex))
    '''
    return results

# Example call
from_base_values = np.random.randint(0, np.iinfo(np.uint64).max, size=10000, dtype=np.uint64) # Bases
size_per_base = 12000
print("Números aleatórios gerados")
gpu_xy = run_kernel_for_seq(from_base_values, size_per_base)  # 100 sequential points per base
run_kernel_for_ks(range(1,200000))

# Computar janelas NAF
def point_to_uint32_array(val):
    le_bytes = val.to_bytes(32, 'little')
    return [int.from_bytes(le_bytes[j:j+4], 'little') for j in range(0, 32, 4)]

def conmputed_naf_numbers(naf_computed_bits):
    precomputed =[]
    max_scalar = 1 << (naf_computed_bits - 1)
    num_points = 1 << (naf_computed_bits - 2)
    array_size = num_points * 24
    print(f"Tamanho em bytes {array_size}")
    curve = SECP256k1.curve
    G = SECP256k1.generator
    for i in range(1, max_scalar, 2): 
        P = i * G 
        x_arr = point_to_uint32_array(P.x())
        pos_y_arr = point_to_uint32_array(P.y())
        neg_y_arr = point_to_uint32_array(curve.p() - P.y())
        precomputed.extend(x_arr + pos_y_arr + neg_y_arr)
    return precomputed
