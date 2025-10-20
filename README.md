

# **secp256k1-gpu-accelerator**

---

## ⚡ High-Performance secp256k1 GPU Implementation

This project explores **low-latency elliptic-curve arithmetic on GPUs**, designed for **benchmarking, cryptographic research, and defensive security testing** involving the Bitcoin **secp256k1** curve.


---


#### ✅  NVIDIA 5090 
<img width="945" height="171" alt="image" src="https://github.com/user-attachments/assets/41052068-cfa7-4337-959f-7ce1afeb5079" />

#### ✅  NVIDIA 5070 TI
<img width="1244" height="203" alt="image" src="https://github.com/user-attachments/assets/9906067c-bab3-43b6-968e-cb5f3fcc5f1d" />




---

### 🧠 Core Engineering Strategies

#### 1. **Native PTX Arithmetic**

All 256-bit field operations (`add`, `sub`, `mul`, `inv`) are manually optimized with **inline PTX**, using
`add.cc.u32` / `addc.u32` and `sub.cc.u32` / `subc.u32` carry chains.
This approach lets the GPU execute full 256-bit arithmetic directly in registers — **no branching, no memory stalls** — achieving up to **3× higher throughput** than compiler-generated code.

#### 2. **Register-Resident Big-Integer Layout**

Eight 32-bit limbs per scalar are held entirely in registers.
Helper macros (`copy_eight`, `shift_first`, `is_zero`) remove pointer overhead and keep every operation inside a **hot register path**, preserving high SM occupancy and avoiding local memory spills.

#### 3. **Jacobian Coordinates & Pseudo-Mersenne Reduction**

Group operations (`point_add`, `point_double`) use **Jacobian coordinates**, avoiding costly inversions.
Field arithmetic is implemented using the pseudo-Mersenne prime
( p = 2^{256} - 2^{32} - 977 ),
folding upper limbs with constants `0x3d1` and `0x03d1` for **branch-free modular reduction**.

#### 4. **8-Bit Windowed NAF Expansion**

`convert_to_window_naf()` encodes scalars in **8-bit Non-Adjacent Form**, packing four coefficients per 32-bit word.
This reduces the number of point additions in scalar multiplication while balancing compute intensity and memory footprint.

#### 5. **Precomputation in Constant Memory**

Precomputed (x,y) pairs are stored in
`__constant__ uint secpk256PreComputed[1536]`.
Constant memory broadcasts the same line to all threads in a warp, resulting in **zero divergence** and **near-perfect cache efficiency**.

#### 6. **SIMD-Style Parallelism**

Each thread computes one scalar multiplication or sequential chain.
The `point_mul_xy_seq` kernel optimizes **sequential additions** by reusing the same Z coordinate and avoiding redundant modular inversions.

#### 7. **Low-Level Memory Discipline**

No dynamic memory.
All temporaries are stack-allocated arrays (`uint[8]` or `uint[16]`).
`#pragma unroll` fully unrolls critical arithmetic loops, maximizing ILP (instruction-level parallelism).

---

### 🚀 Performance Notes

* Cross-compatible with **CUDA** and **OpenCL**.
* Arithmetic path is **deterministic and reproducible**.
* Benchmarked throughput:

  * **≈50M ops/s** (non-sequential kernel)
  * **≈1B ops/s** (sequential scan mode)
* CPU baseline: **>400× slower**, confirming correct GPU acceleration.

---

### ⚙️ Key Components

| Function                           | Purpose                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `add`, `sub`, `add_mod`, `sub_mod` | Multi-limb arithmetic with carry/borrow propagation        |
| `mul_mod`                          | Pseudo-Mersenne modular multiplication                     |
| `inv_mod`                          | Binary extended-GCD inversion                              |
| `point_add`, `point_double`        | Core elliptic-curve group ops                              |
| `convert_to_window_naf`            | 8-bit NAF scalar expansion                                 |
| `point_mul_xy`, `point_mul_xy_seq` | Scalar multiplication kernels (independent and sequential) |

---

### 🧩 Technical Highlights

* **Inline PTX** for 256-bit carry chains
* **Full loop unrolling** for modular arithmetic
* **Constant-time execution** — deterministic across warps
* **Cache-aware data layout** for minimal bandwidth
* **Low-latency SIMD pipelines** leveraging GPU intrinsics

---

### ⚠️ Responsible Use

This repository is provided **solely for benchmarking, cryptographic research, and defensive security testing**.
It is **not intended or approved for private-key recovery or unauthorized data access**.

---

**Author:** Bruno da Silva

**Contact:** [bsbruno@pm.me](mailto:bsbruno@pm.me)

**Focus:** GPU Cryptography • Security Research • High-Performance Computing

---
