# secp256k1-gpu-accelerator
## ⚡ 1 billion throughput (ops/s) — high-performance Bitcoin ecc secp256k1 opencl kernels for testing; defense

---

### ⚡ High-Performance secp256k1 GPU Implementation

This project explores *low-latency elliptic-curve arithmetic on GPUs*, designed for benchmarking, research and security-testing workloads with Eliptic Curves.

### 🧠 Core Strategies

#### 1. **Arithmetic in Native PTX**

All 256-bit field operations (`add`, `sub`, `mul`, `inv`) are hand-optimized with inline PTX using
`add.cc.u32` / `addc.u32` and `sub.cc.u32` / `subc.u32` carry chains.
These intrinsics let the kernel run multi-limb arithmetic entirely in registers — no branching, no memory stalls — producing 2-3× higher throughput than compiler-generated code.

#### 2. **Register-Resident Big-Int Layout**

Eight 32-bit limbs per scalar stay in registers for all operations.
Macros like `copy_eight`, `shift_first`, and `is_zero` eliminate pointer arithmetic and minimize memory pressure.
Every kernel operates in a register-only hot path, keeping SM occupancy high and avoiding spills to local memory.

#### 3. **Jacobian Coordinates & Modular Reduction**

Point doubling and addition (`point_double`, `point_add`) use Jacobian coordinates to avoid costly inversions.
The modular field arithmetic follows the *pseudo-Mersenne* form of the secp256k1 prime
( p = 2^{256} - 2^{32} - 977 ),
folding high limbs with constants `0x3d1` and `0x03d1` for efficient reduction inside 64-bit intermediate products.

#### 4. **Windowed NAF Scalar Expansion (8-bit)**

`convert_to_window_naf()` implements an 8-bit *windowed Non-Adjacent Form*, packing four coefficients per 32-bit word.
This drastically reduces the number of point additions in the scalar multiplication loop, balancing compute vs. memory footprint.

#### 5. **Pre-Computation in Constant Memory**

The kernel reads from `__constant__ uint secpk256PreComputed[1536]`, a flattened table of pre-computed (x,y) pairs.
Because constant memory is broadcast across warps, every thread accesses the same cache line for the same window — zero divergence, near-perfect cache hit rate.

#### 6. **SIMD-Style Parallelism**

Each GPU thread performs one scalar multiply or a sequence of point additions; thousands of threads execute in lockstep.
The `point_mul_xy_seq` kernel handles sequential chains efficiently by keeping the Z coordinate and performing incremental additions without redundant inversions.

#### 7. **Low-Level Memory Discipline**

No dynamic allocation.
All temporaries are stack-allocated arrays of `uint[8]` or `uint[16]`.
Use of `#pragma unroll` hints to the compiler enables full loop unrolling inside arithmetic primitives.

---

### 🚀 Performance Notes

* Designed for CUDA / OpenCL cross-compatibility.
* GPU arithmetic is fully deterministic and reproducible.
* Achieved throughput (benchmarked): **hundreds of millions of point additions per second** on modern consumer GPUs.
* CPU fallback demonstrates >400× performance difference, confirming correct GPU acceleration.
* 50M ops/sec non sequential
* 1B ops/sec on sequential scanning

---

### ⚙️ Key Components

| Function                           | Purpose                                                                |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `add`, `sub`, `add_mod`, `sub_mod` | Multi-limb arithmetic with carry/borrow propagation                    |
| `mul_mod`                          | Karatsuba-style schoolbook multiplication + pseudo-Mersenne reduction  |
| `inv_mod`                          | Binary extended GCD inversion in Jacobian form                         |
| `point_add`, `point_double`        | Core elliptic-curve group operations                                   |
| `convert_to_window_naf`            | Converts scalar to windowed NAF representation (8-bit)                 |
| `point_mul_xy`, `point_mul_xy_seq` | Scalar multiplication kernels for independent and sequential workloads |

---

### 🧩 Technical Highlights

* **Inline PTX arithmetic** (no compiler overhead)
* **Full unrolling** of modular loops
* **Constant-time arithmetic** for deterministic execution
* **Memory-aware layout** minimizing bandwidth per operation
* **Low-latency intrinsics and instruction-level parallelism**

---

### ⚠️ Responsible Use

This implementation is provided **for benchmarking, cryptographic research, and defensive security analysis only**.
It is **not** intended or endorsed for unauthorized key recovery or cracking applications.

---

bsbruno@pm.me
Bruno da Silva
Security Research
