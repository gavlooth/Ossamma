# Wave-PDE Neural Network Layer — Complete Implementation Guide

---

## Part 1 — The Physics: What You Are Actually Simulating

### 1.1 The Second-Order Damped Wave Equation

You start from classical continuum mechanics. A 1-D wave on a medium where each point can also lose energy (damping):

```
∂²u/∂t² = c(x)² · ∂²u/∂x² − γ(x) · ∂u/∂t
```

Every symbol has a concrete meaning:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `u(x,t)` | scalar field | displacement — what moves |
| `∂u/∂t` | scalar field | velocity — how fast it moves |
| `∂²u/∂t²` | scalar field | acceleration — what forces act |
| `c(x)` | per-position | **wave speed** — how fast information travels |
| `γ(x)` | per-position | **damping** — how fast energy dissipates |
| `∂²u/∂x²` | Laplacian | spatial curvature — the restoring force |

The right-hand side says:
- `c(x)² · ∂²u/∂x²` — wherever the field curves (is concave/convex), neighbouring values push it back. This is the wave term.
- `−γ(x) · ∂u/∂t` — the faster it moves, the more friction it experiences. This kills oscillations.

### 1.2 Why This Is Better Than An Uncoupled Oscillator Layer

An uncoupled oscillator layer treats each hidden dimension as an independent harmonic oscillator with no interaction between dimensions.

The wave PDE is fundamentally different: the Laplacian `∂²u/∂x²` **couples** every position to its neighbours. Information can genuinely propagate across the hidden state. This is a richer, physically motivated inductive bias for sequential data.

---

## Part 2 — Reformulating as a First-Order System

A second-order ODE in time is hard to integrate directly. The standard trick: introduce the velocity field as an independent variable.

Define:
```
v(x,t) = ∂u/∂t        (velocity = rate of change of displacement)
```

Now differentiate both sides with respect to time. The original equation becomes:

```
∂u/∂t = v
∂v/∂t = c(x)² · ∂²u/∂x² − γ(x) · v
```

You now have a **coupled first-order system** in `(u, v)`. This is the standard form for numerical integration — every ODE solver expects it.

Your full state at any moment is the pair `(u, v)`:

- `u ∈ ℝᴺ` — displacement field, N values (one per hidden dimension)
- `v ∈ ℝᴺ` — velocity field, N values

**As a neural network layer:** `u` carries the "signal content" and is what you read out as the layer's output. `v` carries "momentum" and is passed between time steps just like an RNN's hidden state. Both `c(x)` and `γ(x)` are learned vectors of length N.

---

## Part 3 — The Spectral Laplacian via FFT

### 3.1 Why FFT?

You need to compute `∂²u/∂x²` — the second spatial derivative of the field `u`. You could use finite differences:

```
∂²u/∂x²|_j ≈ (u[j-1] − 2u[j] + u[j+1]) / Δx²
```

But this is O(N) matrix-vector multiplication with a tridiagonal matrix, and it introduces numerical diffusion (second-order accuracy only).

The **spectral method** is exact for periodic domains: it computes the true Laplacian in O(N log N) using the FFT. No numerical diffusion. Infinitely differentiable.

### 3.2 The Math

For a periodic signal `u` of length N with spacing Δx = 1/N, the discrete Fourier transform gives you the frequency-domain representation `û`:

```
û[k] = FFT(u)[k]  =  Σⱼ u[j] · exp(−2πi·j·k/N)
```

The key theorem: **differentiation in space = multiplication in frequency space**.

```
∂u/∂x   →  FFT  →  (2πi·k/L) · û[k]
∂²u/∂x² →  FFT  →  −(2πk/L)² · û[k]
```

So for our unit domain (L=1):

```
∂²u/∂x² = IFFT( −k² · FFT(u) )
```

where the wavenumbers `k` follow the FFT convention:

```
k = 2π/N · [0, 1, 2, ..., N/2−1, −N/2, ..., −2, −1]
      └── fftfreq order (DC first, then positive, then negative)
```

The factor of `2π/N` comes from: angular frequency = 2π × (mode index / domain length), with domain length N points and unit spacing.

### 3.3 Computing k² in Julia

```julia
N = 64   # state dimension
fft_j = vcat(collect(0 : N÷2 - 1),   # [0, 1, ..., 31]
             collect(-(N÷2) : -1))    # [-32, -31, ..., -1]

k_sq = Float32.((2π / N .* fft_j) .^ 2)
# k_sq[1]  = 0         (DC component, zero curvature)
# k_sq[2]  = (2π/64)²  (fundamental mode, very small)
# k_sq[33] = π²        (Nyquist mode, largest wavenumber)
```

Then the Laplacian of any field `u`:

```julia
Lu = real.(ifft(-k_sq .* fft(u)))
```

The `real.()` is required because IFFT of a Hermitian-symmetric spectrum should be real, but floating point introduces tiny imaginary residuals.

---

## Part 4 — Numerical Integration: Semi-Implicit Euler

### 4.1 Why Not Plain Euler?

The explicit Euler scheme:
```
v_new = v + dt · (c²·Lu − γ·v)
u_new = u + dt · v_new
```

is simple but **conditionally stable**. For large γ (heavy damping), you need `dt < 2/γ` or the velocity oscillates and blows up. In a neural network, γ is learned and can become large.

### 4.2 The Semi-Implicit Scheme

Treat the wave term explicitly (cheap) and the damping term implicitly (stable):

```
v_new = (v + dt · (c²·Lu + f)) / (1 + dt·γ)
u_new = u + dt · v_new
```

Where `f` is the forcing input from the current token.

Derivation: write the implicit equation for v:
```
v_new = v + dt·(c²·Lu + f) − dt·γ·v_new
v_new·(1 + dt·γ) = v + dt·(c²·Lu + f)
v_new = (v + dt·(c²·Lu + f)) / (1 + dt·γ)
```

This scheme is **unconditionally stable for the damping term** regardless of dt or γ. No matter how large γ gets, `v_new` just decays toward zero — it never diverges.

### 4.3 Adding the Input Token as Forcing

At each time step `t`, the input token `xₜ` acts as an external force `f`:

```
f = W_in · xₜ      (input_projection: N × input_dim)
```

The full step:
```
Lu      = IFFT(−k² ⊙ FFT(u))
v_new   = (v + dt·(c²⊙Lu + f)) / (1 + dt·γ)
u_new   = u + dt·v_new
y_t     = W_out · u_new   (output_projection: output_dim × N)
```

---

## Part 5 — Mapping to Sequence Modelling

### 5.1 The Analogy

| PDE concept | Neural network concept |
|-------------|----------------------|
| Spatial domain x, N points | Hidden state dimension (embedding_dim) |
| Time dimension t | Token sequence position |
| Displacement field u(x,t) | Hidden activations at position t |
| Velocity field v(x,t) | Momentum state carried between tokens |
| Input forcing f(x,t) | Projected token embedding at step t |
| Output y(x,t) | Layer output embedding at step t |
| c(x) — wave speed | Learned per-dimension propagation rate |
| γ(x) — damping | Learned per-dimension memory decay |

### 5.2 Sequence Processing Loop

For an input sequence `X` of shape `(input_dim, T)`:

```
state = (u₀, v₀) = zeros    ← initial condition

for t = 1, 2, ..., T:
    f     = W_in · X[:, t]   ← project token to forcing
    Lu    = IFFT(−k²·FFT(u)) ← spectral Laplacian
    v_new = (v + dt·(c²·Lu + f)) / (1 + dt·γ)
    u_new = u + dt·v_new
    Y[:, t] = W_out · u_new  ← read out output
    u, v  = u_new, v_new     ← carry state forward
```

This is a **causal** (left-to-right) recurrence. Each token's output depends on all previous tokens through the accumulated state `(u, v)`.

### 5.3 Batched Processing with `accumulate`

In Julia/Lux we process batches. For input `(input_dim, T, B)`:

- At each time step, state is `(N, B)` for both u and v
- Stack them: `stacked = [u; v]` shape `(2N, B)`
- Use `accumulate(step_fn, input_slices; init=stacked_init)`
- This gives a vector of T matrices, each `(2N, B)` — all states over time

```julia
state_history = accumulate(evolve_state, input_slices; init = init_stacked)
# state_history[t] :: (2N, B)  =  [u at step t ; v at step t]
```

The output is then built by extracting u from each state and projecting:

```julia
u_history  = [s[1:N, :] for s in state_history]   # T × (N, B)
u_flat_TB  = reduce(hcat, u_history)               # (N, T*B)  [B varies fastest]
output_flat = W_out * u_flat_TB                    # (out_dim, T*B)
output_BT   = reshape(output_flat, out_dim, B, T)  # reshape: B dim before T
output      = permutedims(output_BT, (1, 3, 2))    # (out_dim, T, B)
```

Why `(out_dim, B, T)` first and then permute? Because `hcat` of T matrices each `(N, B)` gives columns in order `[b1_t1, b2_t1, ..., bB_t1, b1_t2, ...]` — B varies fastest. `reshape` fills in column-major order so B lands in dim 2, T in dim 3. Then `permutedims` swaps them to the standard Lux layout.

---

## Part 6 — Julia/Lux Implementation, Line by Line

### 6.1 Struct Definition

```julia
struct WavePDELayer <: Lux.AbstractLuxLayer
    input_dimension::Int       # dim of input tokens
    state_dimension::Int       # N — spatial domain size = hidden dim
    output_dimension::Int      # dim of output tokens
    minimum_frequency::Float32 # used to space-initialize c(x)
    maximum_frequency::Float32
    default_time_step::Float32 # Δt for numerical integration
end
```

No inner constructor is defined. Julia auto-generates a positional one matching all six fields:

```julia
WavePDELayer(input_dim, state_dim, output_dim, min_f, max_f, dt)
```

This is exactly the same call signature as `WavePDE`, making `WavePDELayer` a drop-in replacement in any code that constructs oscillator layers.

**Why not define an explicit outer constructor with the same signature?**
In Julia, defining `function WavePDELayer(a::Int, b::Int, c::Int, d::Float32, e::Float32, f::Float32)` creates an outer constructor with the same positional types as the auto-generated inner one. When you then call `WavePDELayer(...)` inside that function, it calls itself — infinite recursion and a stack overflow. Always let the struct's implicit constructor handle the fully-specified case.

### 6.2 Parameter Initialization

```julia
function Lux.initialparameters(rng::Random.AbstractRNG, layer::WavePDELayer)
    N = layer.state_dimension

    # Wave speed c(x) = softplus(log_wave_speed)
    # We want c spread over [min_freq, max_freq] at initialisation.
    # First create target c values linearly spaced:
    c_range = range(layer.minimum_frequency, layer.maximum_frequency; length = N)
    # Then invert softplus: softplus⁻¹(y) = log(exp(y) - 1)
    # so that softplus(log_wave_speed) ≈ c_range at init
    log_wave_speed = log.(exp.(Float32.(c_range)) .- 1.0f0)

    # Damping γ(x) = softplus(log_damping)
    # Light initial damping: γ ≈ 0.01 everywhere
    # softplus⁻¹(0.01) = log(exp(0.01) - 1) ≈ -4.6
    log_damping = fill(log(exp(0.01f0) - 1.0f0), N)

    # Input/output projections — small std (0.02) matches WavePDE convention
    input_projection  = randn(rng, Float32, N, layer.input_dimension)  .* 0.02f0
    output_projection = randn(rng, Float32, layer.output_dimension, N) .* 0.02f0

    return (
        log_wave_speed    = log_wave_speed,
        log_damping       = log_damping,
        input_projection  = input_projection,
        output_projection = output_projection,
    )
end
```

**Why softplus and not exp?**
`exp` collapses near zero very fast for negative inputs — the gradient of `exp(x)` at `x = -5` is `≈ 0.007`, which causes near-zero gradients for parameters that drift negative. `softplus(x) = log(1 + exp(x))` is smoother and its gradient `sigmoid(x)` is always in `(0, 1)`, never saturating in the negative direction. This makes it easier to train c and γ through many layers of backprop.

**Why linearly spaced c?**
Matching WavePDE's stiffness range initialisation. Low-speed modes (`c ≈ min_freq`) have long wavelengths and capture slow, long-range dependencies. High-speed modes (`c ≈ max_freq`) have short wavelengths and capture sharp local features. The diversity across N dimensions encourages gradient flow and prevents dead modes.

**Why 0.02 std for projections?**
A common convention for SSM-style layers. Large initial projections amplify the forcing term and destabilise the PDE early in training. Small projections let the PDE dynamics dominate initially, which tends to produce stable training trajectories.

### 6.3 State Initialization

```julia
function Lux.initialstates(_rng::Random.AbstractRNG, layer::WavePDELayer)
    # Row 1: u (displacement), Row 2: v (velocity)
    # Layout (2, N) matches WavePDE's (2, N) oscillator_state convention
    (wave_state = zeros(Float32, 2, layer.state_dimension),)
end
```

The wave starts at rest: zero displacement, zero velocity. This is the standard zero initial condition for wave problems. In the RNN framing, it means the layer carries no cross-sequence memory — each new batch starts fresh. If you wanted persistent memory across batches you would manage and feed back `wave_state` explicitly.

### 6.4 Forward Pass — Step by Step

#### Step A: Handle batch dimensions

```julia
is_batched = ndims(input_sequence) == 3

# Standardize to 3D: (input_dim, T, B)
standardized = is_batched ? input_sequence :
    reshape(input_sequence, size(input_sequence, 1), size(input_sequence, 2), 1)

(n_features, n_timesteps, n_batches) = size(standardized)
```

The layer accepts both `(F, T)` and `(F, T, B)`. We always work in 3D internally. At the very end we squeeze the synthetic batch dimension away if input was 2D.

Why add a synthetic batch dim instead of writing two code paths? Because all subsequent operations (matrix multiply, FFT, accumulate) work identically on `(*, B)` matrices regardless of whether B=1 or B=32. One code path, zero branching.

#### Step B: Constrain physics parameters

```julia
c    = NNlib.softplus.(parameters.log_wave_speed)   # (N,) — wave speeds, all > 0
γ    = NNlib.softplus.(parameters.log_damping)       # (N,) — dampings, all ≥ 0
c_sq = c .^ 2                                        # (N,) — used in PDE
```

We store `log_wave_speed` and `log_damping` as unconstrained reals and apply `softplus` at every forward pass. The gradient of the loss flows back through `softplus` cleanly. `c_sq = c²` is precomputed once per forward call and reused inside the scan loop rather than squaring N times inside the closure.

#### Step C: Spectral wavenumbers — outside the autodiff graph

```julia
neg_k_sq = @ignore let
    fft_j  = vcat(collect(0:(N÷2 - 1)), collect(-(N÷2):-1))
    k_vals = Float32.((2π / N .* fft_j) .^ 2)
    buf    = similar(c_sq, N)   # same device as parameters (CPU or CuArray)
    buf   .= -k_vals            # mutation OK — Zygote never sees this block
    reshape(buf, N, 1)          # (N, 1) for broadcasting over batch dim
end
```

Three things to understand here:

**1. `@ignore`**
`Zygote.@ignore` wraps the expression in `Zygote.ignore()`. In the forward pass the block executes normally. In the backward pass Zygote does not record any computation graph through it and does not differentiate it. This is correct because `neg_k_sq` has zero gradient — it is a fixed physical constant derived from N, not from any learned parameter.

**2. `similar(c_sq, N)`**
`similar(A, dims...)` creates an uninitialised array with the same element type AND the same storage type as `A`. If `c_sq` is a `Vector{Float32}` (CPU), `similar(c_sq, N)` is also a `Vector{Float32}`. If `c_sq` is a `CuArray{Float32}` (GPU), `similar(c_sq, N)` is also a `CuArray{Float32}`. This is the idiomatic Julia way to allocate on whichever device the computation is running on without any explicit `isa(x, CuArray)` checks.

**3. The mutation `buf .= -k_vals`**
Inside `@ignore`, Zygote will never see this `.=` broadcast-assignment mutation. Mutations are forbidden by Zygote in the AD graph because they break the functional programming model that AD requires. But here the mutation is invisible to AD, so it is safe. The CPU vector `k_vals` is broadcast into the (potentially GPU) `buf` — CUDA.jl handles this broadcast assignment from host to device transparently for small vectors.

**Why (N, 1) shape?**
The Laplacian step needs `neg_k_sq .* U_hat` where `U_hat` is `(N, B)`. Broadcasting a `(N, 1)` array against a `(N, B)` array applies the same k² to every batch item — which is what we want (the wavenumbers are the same for all inputs in a batch).

#### Step D: Project input tokens to forcing

```julia
input_flat     = reshape(standardized, n_features, :)          # (input_dim, T*B)
projected_flat = parameters.input_projection * input_flat      # (N, T*B)
projected      = reshape(projected_flat, N, n_timesteps, n_batches)  # (N, T, B)

input_slices = [copy(projected[:, t, :]) for t in 1:n_timesteps]
```

We project all tokens at once with a single matrix multiply — cheaper than T separate matmuls. The result is then split into a vector of T matrices, each of shape `(N, B)`, for `accumulate`.

The `copy()` call is not optional on GPU. `projected[:, t, :]` returns a `SubArray` — a view into the underlying memory with non-contiguous strides. CUDA kernels require contiguous memory. `copy()` materialises the view into a fresh contiguous `CuArray`. On CPU it is a small overhead but harmless.

#### Step E: Build initial stacked state

```julia
init_u = repeat(copy(state.wave_state[1, :]), 1, n_batches)   # (N, B)
init_v = repeat(copy(state.wave_state[2, :]), 1, n_batches)   # (N, B)
init_stacked = vcat(init_u, init_v)                            # (2N, B)
```

`state.wave_state` has shape `(2, N)`. Row 1 holds the saved displacement `u`, row 2 the saved velocity `v`. We extract each row as a length-N vector, then broadcast across the batch dimension using `repeat`.

`repeat(v, 1, B)` turns a column vector `(N,)` into a matrix `(N, B)` where every column is a copy of v. The `1` means "repeat 1 time along dim 1" (no repetition along N), and `B` means "repeat B times along dim 2".

The stacked layout `[u; v]` puts displacement on top (rows `1:N`) and velocity below (rows `N+1:2N`). This convention is maintained throughout the forward pass. Every `evolve_state` call unpacks at the top and repacks at the bottom.

#### Step F: The PDE step function (closure)

```julia
dt     = layer.default_time_step
c_sq_b = reshape(c_sq, N, 1)    # (N, 1) — broadcasts over batch
γ_b    = reshape(γ,    N, 1)    # (N, 1)

evolve_state = (stacked, forcing) -> begin
    # Unpack state
    u = copy(stacked[1:N, :])          # (N, B) — displacement
    v = copy(stacked[(N+1):end, :])    # (N, B) — velocity

    # Spectral Laplacian: Lu = IFFT(−k² ⊙ FFT(u))
    U_hat = fft(u, 1)                           # (N, B) Complex{Float32}
    Lu    = real.(ifft(neg_k_sq .* U_hat, 1))   # (N, B) Float32

    # Semi-implicit Euler step
    v_new = (v .+ dt .* (c_sq_b .* Lu .+ forcing)) ./
            (1.0f0 .+ dt .* γ_b)
    u_new = u .+ dt .* v_new

    return vcat(u_new, v_new)   # (2N, B) stacked new state
end
```

Line-by-line breakdown of the body:

**`u = copy(stacked[1:N, :])`**
Slicing `stacked[1:N, :]` returns a SubArray view. `copy()` forces materialisation into a contiguous array. Necessary on GPU; harmless on CPU. We do this for both u and v.

**`U_hat = fft(u, 1)`**
Computes the 1-D FFT of each column of `u` independently. `u` is `(N, B)` real, `U_hat` is `(N, B)` complex. The `1` argument means "transform along dimension 1" — the spatial/hidden dimension. Dimension 2 (batch) is left untouched. On CPU this dispatches to FFTW's plan-based algorithm. On GPU (CuArray input) it dispatches to NVIDIA cuFFT via CUDA.jl's AbstractFFTs extension.

**`neg_k_sq .* U_hat`**
Element-wise multiply `(N, 1)` × `(N, B)` complex → `(N, B)` complex. Broadcasting applies the same `−k²` multiplier to every batch item. This is the spectral derivative operator: multiplying by `−k²` in Fourier space is equivalent to applying `∂²/∂x²` in physical space.

**`real.(ifft(..., 1))`**
IFFT back to physical space. The result is theoretically real-valued, but floating-point accumulation leaves imaginary parts of magnitude `~1e-7`. `real.()` discards these. The gradient of `real.(z)` is `real.(dz)` — it simply zeroes the imaginary component of any upstream complex gradient, which is correct.

**`c_sq_b .* Lu .+ forcing`**
The wave acceleration: wave-speed-squared times the Laplacian, plus the input forcing term. `c_sq_b` is `(N, 1)`, `Lu` is `(N, B)`, `forcing` is `(N, B)`. Broadcasting works correctly.

**`(... ) ./ (1.0f0 .+ dt .* γ_b)`**
The implicit damping denominator. Key properties:
- Always ≥ 1.0 (since γ ≥ 0 and dt > 0)
- As γ → ∞, v_new → 0 (over-damped limit)
- As γ → 0, v_new = v + dt·(c²·Lu + f) (undamped limit, explicit Euler)
- Gradient: `∂(1/(1+dt·γ))/∂γ = −dt/(1+dt·γ)²` — always negative, well-behaved

**`u_new = u .+ dt .* v_new`**
Position update using the **new** velocity (not the old one). This is the symplectic Euler style — it is more energy-preserving than using `v_old` and leads to more stable long-horizon dynamics.

**`return vcat(u_new, v_new)`**
Repacks into the `(2N, B)` stacked format. `accumulate` passes this as the `stacked` argument to the next step.

#### Step G: Run the time scan

```julia
state_history = accumulate(evolve_state, input_slices; init = init_stacked)
# state_history :: Vector{Matrix{Float32}}, length T
# state_history[t] :: (2N, B)
```

`Base.accumulate` is Julia's prefix scan: for a sequence `[a₁, a₂, ..., aₜ]` with initial value `s₀` and binary function `f`, it computes:

```
[f(s₀, a₁), f(f(s₀, a₁), a₂), ..., f(f(...f(s₀, a₁)..., aₜ₋₁), aₜ)]
```

This is exactly the recurrence we want, and it returns **all intermediate states** — which is what we need to build the full output sequence. Compare with `foldl` which only returns the final state.

`accumulate` is sequential — it cannot be parallelised across T because each call depends on the previous result. This is fundamental to causal sequence modelling, not a limitation of this implementation. Parallelism happens across N (the hidden dimension) and B (the batch dimension) inside each `evolve_state` call.

Zygote differentiates through `accumulate` by unrolling it into a chain of T function calls and applying BPTT (backpropagation through time). The memory cost is O(T) to store all intermediate states for the backward pass.

#### Step H: Extract output and reshape

```julia
u_history    = [copy(s[1:N, :]) for s in state_history]   # T × (N, B)
u_flat_TB    = reduce(hcat, u_history)                      # (N, T*B)
output_flat  = parameters.output_projection * u_flat_TB    # (out_dim, T*B)
output_BT    = reshape(output_flat, layer.output_dimension, n_batches, n_timesteps)
output_tensor = permutedims(output_BT, (1, 3, 2))           # (out_dim, T, B)
```

Full shape trace:

```
u_history[1]          :: (N, B)        state at t=1
u_history[T]          :: (N, B)        state at t=T
reduce(hcat, ...)     :: (N, T*B)      hcat of T matrices (N,B); columns: [B@t1, B@t2, ...]
W_out * ...           :: (out_dim, T*B)
reshape(..., D, B, T) :: (D, B, T)     column-major fill: B varies fastest ✓
permutedims(1,3,2)    :: (D, T, B)     standard Lux output layout
```

Why does the `reshape` put B in dim 2 and T in dim 3? Because `hcat` stacks matrices left-to-right, so the column order is `[b1_t1, b2_t1, ..., bB_t1, b1_t2, ...]`. In Julia's column-major layout, `reshape(M, D, B, T)` fills: for fixed `D`, index 2 (B) varies before index 3 (T). That matches the column order exactly.

#### Step I: Preserve the final state

```julia
last_stacked   = state_history[end]                          # (2N, B)
next_u_row     = transpose(copy(last_stacked[1:N,         1]))  # (1, N)
next_v_row     = transpose(copy(last_stacked[(N+1):end,   1]))  # (1, N)
next_wave_state = vcat(next_u_row, next_v_row)               # (2, N)
```

We save only the first batch item's final state (column 1). This matches WavePDE's convention and Lux's stateful layer contract: the state is per-layer and per-example, not per-batch. The saved state would be fed back in as `state` on the next call if you run the model autoregressively or across document chunks.

`transpose(copy(...))` turns a length-N vector into a `(1, N)` row matrix so that `vcat(next_u_row, next_v_row)` gives the `(2, N)` layout expected by `initialstates`.

#### Step J: Return

```julia
final_output = is_batched ? output_tensor : dropdims(output_tensor, dims = 3)
return (final_output, (wave_state = next_wave_state,))
```

If the caller passed a 2D input `(F, T)`, they expect a 2D output `(out_dim, T)`. `dropdims(..., dims=3)` removes the synthetic B=1 batch dimension. The returned named tuple `(wave_state = ...,)` is the new state for the next call — Lux passes this through automatically during training.

---

## Part 7 — Device Compatibility (CPU / GPU)

The layer uses three patterns that need to work on both CPU and CUDA without any explicit device checks:

### 7.1 FFT dispatch via AbstractFFTs

```julia
using FFTW   # registers fft/ifft for CPU AbstractArray via AbstractFFTs.jl
# CUDA.jl (already in Project.toml) registers fft/ifft for CuArray via cuFFT
```

When the user calls `Lux.setup(gpu_device, model)` to move parameters to GPU, the forward pass receives CuArrays everywhere. `fft(cuarray, 1)` automatically dispatches to cuFFT because CUDA.jl registers an AbstractFFTs extension method. No code changes needed between CPU and GPU.

### 7.2 Device-agnostic constant allocation

```julia
buf = similar(c_sq, N)   # c_sq is on device X → buf is on device X
buf .= -k_vals            # k_vals is a CPU Float32 vector
                          # CUDA.jl allows broadcast-fill from CPU scalars/vectors
```

`similar(A, dims)` is the Julia idiom for "give me a new array on the same device as A, with the given shape." This avoids explicit `CUDA.cu(...)` calls or `isa(x, CuArray)` branches. The subsequent `.=` broadcast-assignment fills the device buffer from a CPU vector — CUDA.jl handles this transparently for simple 1D vectors (it copies to device memory).

### 7.3 Mutation safely inside `@ignore`

```julia
neg_k_sq = @ignore let
    buf = similar(c_sq, N)
    buf .= -k_vals           # ← mutation: Zygote.ignore() hides this
    reshape(buf, N, 1)
end
```

Zygote forbids mutations inside the AD graph because they break the tape-based differentiation model (you cannot undo an in-place write during the backward pass). `@ignore` marks this entire `let` block as opaque to Zygote — it runs in the forward pass normally, returns its value, and Zygote treats that value as an external constant with zero gradient contribution. This is the canonical pattern for non-differentiable setup code that needs device-aware allocation.

---

## Part 8 — Integration Into SwammaBlock

The Wave-PDE layer slips in as the `WaveGateLayer` in the GLU gate branch:

```
input (dim, T, B)
        │
        ▼
TimeConditionedLayerNorm
        │
        ▼
GluProjection: Dense(dim → 2·dim)
        │
        ├─ content_half (dim, T, B)
        │       │
        │       ▼
        │   LinearAttention
        │       │
        │       ▼
        │   RMSNorm ─────────────────────────────────────┐
        │                                                 │
        └─ gate_half (dim, T, B)                          │ GLU: content ⊙ gate
                │                                         │
                ▼                                         │
          WavePDELayer                                   │
          (damped wave PDE, spectral Laplacian)           │
                │                                         │
                ▼                                         │
           RMSNorm                                        │
                │                                         │
                ▼                                         │
           sigmoid ──────────────────────────────────────►│
                                                          │
                                               glu_output (dim, T, B)
                                                          │
                               ┌──────────────────────────┘
                               │
                 input_gate = sigmoid(Dense(glu_output))
                               │
                  gated_x = x_norm ⊙ input_gate
                               │
                        SWAttention (local branch)
                               │
                  local_output (dim, T, B)
                               │
               α = sigmoid(AlphaProjection(x_norm) + α_bias)
                               │
            mixed = α·glu_output + (1−α)·local_output
                               │
                    Dropout → FFN → LayerNorm
                               │
                         residual + output
```

**What the Wave-PDE gate learns:**
`gate_half` is a projected copy of the normalised input. The Wave-PDE evolves this over the sequence using the wave equation, producing a gate signal that is:
- **Spatially coupled** across the N hidden dimensions (via the Laplacian)
- **Temporally smooth** (wave propagation is smooth)
- **Frequency-selective** (different c values respond to different temporal frequencies)

This gate then modulates the content signal via element-wise multiplication. The coupled spatial structure gives the gate a richer, more coherent form than an uncoupled oscillator gate.

---

## Part 9 — Hyperparameter Guide

| Parameter | Default | Effect | How to tune |
|-----------|---------|--------|-------------|
| `state_dimension` | = `embedding_dim` | Domain size N. More = richer dynamics but O(N log N) FFT cost. | Keep equal to embedding_dim. |
| `min_frequency` | `0.1` | Slowest wave speed at init; controls long-range coupling. | Lower for longer sequences, sparser data. |
| `max_frequency` | `10.0` | Fastest wave speed at init; controls local sharpness. | Higher for tasks needing sharp local features. |
| `default_time_step` dt | `0.1` | Integration step. Smaller = more stable, slower per-token propagation. | Reduce if training diverges early. |
| Projection std | `0.02` | Initial forcing magnitude. | Reduce if loss spikes at step 1. |
| `log_damping` init | `≈ −4.6` (γ ≈ 0.01) | Initial memory decay rate. | Raise for tasks needing short memory; lower for long-range tasks. |

**Soft CFL stability condition** (the semi-implicit scheme makes this advisory, not hard):

```
For the wave term to not amplify:
    dt · c_max · k_max ≲ 1

k_max = π  (Nyquist wavenumber for N points)

→ dt ≲ 1 / (π · c_max)

For c_max = 10:  dt ≲ 0.032
Default dt = 0.1 is slightly above this — the implicit damping
absorbs any mild instability, but keep c_max reasonable.
```

---

## Part 10 — What Gradients Flow Through

When you call `Zygote.gradient(loss, parameters)`, the computation graph looks like:

```
loss
  │
  ├─► ∂L/∂W_out   (output_projection — standard linear layer grad)
  │
  ├─► ∂L/∂u[T]    (displacement at final step)
  │     │
  │     └─► BPTT through accumulate (T steps of evolve_state)
  │           │
  │           ├─► ∂L/∂c_sq   via   c_sq_b .* Lu   in each step
  │           │               chain: ∂(c²·Lu)/∂(c²) = Lu
  │           │               then:  ∂(c²)/∂c = 2c
  │           │               then:  ∂c/∂log_c = softplus'(log_c) = sigmoid(log_c)
  │           │
  │           ├─► ∂L/∂γ      via   denominator (1 + dt·γ)
  │           │               chain: ∂(1/(1+dt·γ))/∂γ = −dt/(1+dt·γ)²
  │           │               then:  ∂γ/∂log_γ = sigmoid(log_γ)
  │           │
  │           └─► ∂L/∂W_in   (input_projection — accumulates across T)
  │
  └─► neg_k_sq   →  ZERO (wrapped in @ignore, not part of the graph)
```

**Key differentiable operations and their gradients:**

| Operation | Forward | Backward (chain rule) |
|-----------|---------|----------------------|
| `softplus(x)` | `log(1+exp(x))` | `sigmoid(x)` |
| `c²` | `c .^ 2` | `2c .* dc` |
| `fft(u, 1)` | DFT matrix multiply | conjugate DFT (FFTW has registered ChainRules) |
| `neg_k_sq .* U_hat` | element-wise multiply | `neg_k_sq .* dU_hat` (neg_k_sq treated as constant) |
| `ifft(..., 1)` | inverse DFT | conjugate inverse DFT (registered ChainRules) |
| `real.(z)` | discard imag | zero the imag part of upstream gradient |
| `v_new = (...) ./ (1 + dt·γ)` | element-wise divide | upstream grad / (1 + dt·γ) |
| `vcat(u_new, v_new)` | stack arrays | split gradient, route to u and v |
| `accumulate(f, xs)` | sequential scan | BPTT: reverse-order chain of gradients |

---

## Part 11 — Complete Source File Annotated

```julia
module WavePDE

using Lux
using Random
using NNlib
using FFTW          # CPU FFT; CUDA.jl auto-extends to GPU via AbstractFFTs
using Zygote: @ignore

# ============================================================
# Struct
# ============================================================

struct WavePDELayer <: Lux.AbstractLuxLayer
    input_dimension::Int
    state_dimension::Int       # N — spatial domain size
    output_dimension::Int
    minimum_frequency::Float32
    maximum_frequency::Float32
    default_time_step::Float32
end
# Julia auto-generates:
#   WavePDELayer(input_dim, state_dim, output_dim, min_f, max_f, dt)
# Do NOT add an outer constructor with the same signature — it recurses infinitely.

# ============================================================
# Parameters
# ============================================================

function Lux.initialparameters(rng::Random.AbstractRNG, layer::WavePDELayer)
    N = layer.state_dimension

    c_range        = range(layer.minimum_frequency, layer.maximum_frequency; length = N)
    log_wave_speed = log.(exp.(Float32.(c_range)) .- 1.0f0)   # softplus⁻¹(c_range)
    log_damping    = fill(log(exp(0.01f0) - 1.0f0), N)        # softplus⁻¹(0.01)

    return (
        log_wave_speed    = log_wave_speed,
        log_damping       = log_damping,
        input_projection  = randn(rng, Float32, N, layer.input_dimension)  .* 0.02f0,
        output_projection = randn(rng, Float32, layer.output_dimension, N) .* 0.02f0,
    )
end

# ============================================================
# State
# ============================================================

function Lux.initialstates(_rng::Random.AbstractRNG, layer::WavePDELayer)
    (wave_state = zeros(Float32, 2, layer.state_dimension),)
    # Row 1 = u (displacement), Row 2 = v (velocity)
end

# ============================================================
# Forward pass
# ============================================================

function (layer::WavePDELayer)(input_sequence::AbstractArray, parameters, state)

    # A. Standardise to 3D: (input_dim, T, B)
    is_batched   = ndims(input_sequence) == 3
    standardized = is_batched ? input_sequence :
        reshape(input_sequence, size(input_sequence, 1), size(input_sequence, 2), 1)
    (n_features, n_timesteps, n_batches) = size(standardized)
    N = layer.state_dimension

    # B. Constrain physics parameters
    c    = NNlib.softplus.(parameters.log_wave_speed)   # (N,) > 0
    γ    = NNlib.softplus.(parameters.log_damping)       # (N,) ≥ 0
    c_sq = c .^ 2                                        # (N,)

    # C. Wavenumbers — constant, invisible to Zygote, allocated on correct device
    neg_k_sq = @ignore let
        fft_j  = vcat(collect(0:(N÷2 - 1)), collect(-(N÷2):-1))
        k_vals = Float32.((2π / N .* fft_j) .^ 2)
        buf    = similar(c_sq, N)   # same device as parameters
        buf   .= -k_vals
        reshape(buf, N, 1)          # (N, 1) for batch broadcasting
    end

    # D. Project all tokens to forcing in one matmul
    input_flat     = reshape(standardized, n_features, :)
    projected_flat = parameters.input_projection * input_flat
    projected      = reshape(projected_flat, N, n_timesteps, n_batches)
    input_slices   = [copy(projected[:, t, :]) for t in 1:n_timesteps]

    # E. Initial stacked state: [u₀; v₀] of shape (2N, B)
    init_u       = repeat(copy(state.wave_state[1, :]), 1, n_batches)
    init_v       = repeat(copy(state.wave_state[2, :]), 1, n_batches)
    init_stacked = vcat(init_u, init_v)

    # F. PDE step closure
    dt     = layer.default_time_step
    c_sq_b = reshape(c_sq, N, 1)
    γ_b    = reshape(γ,    N, 1)

    evolve_state = (stacked, forcing) -> begin
        u     = copy(stacked[1:N, :])
        v     = copy(stacked[(N+1):end, :])
        U_hat = fft(u, 1)                            # FFT along spatial dim
        Lu    = real.(ifft(neg_k_sq .* U_hat, 1))    # spectral Laplacian
        v_new = (v .+ dt .* (c_sq_b .* Lu .+ forcing)) ./
                (1.0f0 .+ dt .* γ_b)                 # semi-implicit damping
        u_new = u .+ dt .* v_new
        return vcat(u_new, v_new)
    end

    # G. Sequential scan over time
    state_history = accumulate(evolve_state, input_slices; init = init_stacked)

    # H. Extract u, project to output, reshape to (out_dim, T, B)
    u_history     = [copy(s[1:N, :]) for s in state_history]
    u_flat_TB     = reduce(hcat, u_history)
    output_flat   = parameters.output_projection * u_flat_TB
    output_BT     = reshape(output_flat, layer.output_dimension, n_batches, n_timesteps)
    output_tensor = permutedims(output_BT, (1, 3, 2))

    # I. Save last state (first batch item)
    last          = state_history[end]
    next_u        = transpose(copy(last[1:N,        1]))
    next_v        = transpose(copy(last[(N+1):end,  1]))
    next_state    = (wave_state = vcat(next_u, next_v),)

    final_output = is_batched ? output_tensor : dropdims(output_tensor, dims = 3)
    return (final_output, next_state)
end

end # module WavePDE
```
