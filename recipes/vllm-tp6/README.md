# Serve full GLM-5.3 with vLLM TP6

## What this recipe reproduces

The current P27 inference configuration on six GB10 nodes: original 3.25 bpw experts,
MXFP8 target/draft dense paths, native MTP up to four draft tokens, adaptive
request/phase scheduling, native/E3 prefill dispatch, and two validated fabric
ports for both NCCL and the custom RoCEnante collectives.

**Prerequisite:** the retained, qualified image must already be installed
on every node. This repository does not yet build that
image from a clean machine or provide a downloadable registry image.
Do not substitute a stock vLLM image: it lacks the required mixed-K/TP6 patches.

```text
Current image ID: sha256:20f485f5fbe17eec01a2b1c1713c53b55503b97b10efac3d011dff0b857f87c6
E3 boundary base: sha256:286e0a42c8caa3a7d45a76f006bd400e391e161b6ec9f0bbdc9dacb7dfb0d20e
E3 row32 base: sha256:b4988201229054893df527528c38e8091d4d5392d44c314406e66493ec9300da
E3 diagnostic base: sha256:340a9bae07ab134120249cf0224108fda7b3706720724bd1b2fe705245e7ce20
MTP-control base: sha256:2e947348fd26b8d58e4535e0321126935069a5e521784fa573f19fb560975d9d
Budget-control base: sha256:7176241a30ba8349fccb979ad7a35d0eba422d23017543da8c0147b28cb0c1cd
P27 base image ID: sha256:8935c96fe1670c4016bbc47c905477cb652a5a2b89214c0e952920d2ed232e90
```

The October 3 image adds bounded prefill and MTP controls. It retains qualified
**3072** prefill and the **original adaptive MTP policy**. Before starting rank 0,
install both persistent control files and read their qualification limits:
[prefill budgets](PREFILL_BUDGETS.md) and [MTP tuning](MTP_TUNING.md).
Smaller/adaptive budgets failed the numerical gate; MTP retuning finished without
a promoted policy. [Dual-port RoCEnante](COMMUNICATION.md) is now selected for
modest overall and concurrent gains, with explicit per-workload tradeoffs.
The current image also selects [32-row E3 prefill](E3_PREFILL.md). **Every rank**
requires persistent `/root/.cache/amos-e3-rows.json` and
`/root/.cache/amos-e3-boundary.json` controls described there. The selected boundary
is native through 32 rows, E3 above 32; missing controls fail visibly.

The [copy/ngram comparison](COPY_DRAFTING.md) is complete. The current forward
image retains the tested history/configuration fixes and selects original adaptive
MTP; copy passed correctness but lost general throughput. No new drafter weights
or target-precision changes were introduced.

A Docker image ID is not a registry digest. Move an existing image with
`docker save` / `docker load`, then compare `docker image inspect --format
'{{.Id}}' IMAGE` on every node. Do not try to pull this ID from a registry.

## Hardware and configuration

- Six DGX Sparks / GB10, 128 GB unified memory each, one CUDA rank per node.
- ConnectX-7 RoCE fabric; explicit interface, HCA, GID and per-node fabric IP.
- Local verified rank directories from the [weight recipe](../weights/README.md).
- Linux cgroup v2, Docker with NVIDIA support, systemd, and the host memory guard.
- A quiet, drained fleet with sufficient memory headroom before starting.

| Setting | P27 value |
| --- | --- |
| TP / nodes / ranks | 6 / 6 / 0–5 |
| Profile | `mtp4` |
| `DENSE` / `DRAFT_DENSE` | `mxfp8` / `mxfp8` |
| `GPU_MEMORY_UTILIZATION` | `0.75` |
| `KV_CACHE_MEMORY_BYTES` | `25769803776` (24 GiB) |
| Maximum batch tokens | 3072 |
| Maximum sequence length / admitted requests | 360000 / 4 |
| Shared cache slots | Not re-counted in the hardening pass; see historical P24 measurement |
| Memory guard | 8 GiB available floor sustained for 2 seconds, plus pressure/refault checks |

Four-request admission does not mean four simultaneous 360K contexts fit.
All requests share the cache. The 804K resident allocation in our TensorFold
experiment is a separate backend configuration.

## Launch contract

Use your fleet supervisor to create one container per rank with
[`runtime/vllm/node.sh`](../../runtime/vllm/node.sh) as its entrypoint.
Our private controller also handles application admission and exact-container
guards; its site-specific host mappings and credentials are deliberately absent here.

1. Drain application requests and own the fleet maintenance window.
2. Confirm the image ID and verified local shard on every rank.
3. Mount the rank directory at `/model:ro`, the entrypoint at
   `/opt/amos-tp6/node.sh:ro`, and a writable persistent local cache at
   `/root/.cache`. Rank 0 requires the prefill/MTP controls; every rank also
   requires the selected E3 row control described above.
4. Use host network/IPC, GPU access, `/dev/infiniband`, unlimited memlock,
   `IPC_LOCK`, and the host's performance CPU cores.
5. Set `NODE_RANK`, `HEAD_IP`, `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`,
   `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA`, `B12X_ROCE_HCA`,
   and `B12X_ROCE_GID_INDEX`. Use the **qualified port** for each host.
   P27 uses both validated fabric HCAs for NCCL with IPv4 RoCEv2 and an
   address-range filter for your fabric (`NCCL_IB_ADDR_FAMILY=AF_INET`,
   `NCCL_IB_ROCE_VERSION_NUM=2`, and a site-specific `NCCL_IB_ADDR_RANGE`).
   Leave a fixed `NCCL_IB_GID_INDEX` unset for this NCCL policy. Validate
   per-host GIDs against live interface/subnet state. The custom b12x collective
   uses both validated HCAs and one explicit GID index shared by those two HCAs
   on each node; see the [mapping constraints](COMMUNICATION.md#configuration-and-reproduction).
   `AMOS_TP6_NCCL_RAILS=2` records the controller selection policy; the node
   entrypoint alone does not discover or configure a second fabric.
6. Apply the table above and [tuning.json](tuning.json) as environment values,
   plus `VLLM_ENABLE_ROCE_ALLREDUCE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
   `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
7. Before starting each created container, arm the
   [memory guard](../../runtime/vllm/memguard.py) with that exact Docker ID in
   `STATE_DIR/container-id`, `HARD_AVAIL_GIB=8`, and `HARD_AVAIL_S=2`.
   Keep its existing PSI/refault/swap/self-stall protections. Verify fresh
   armed status after startup. A container memory limit alone is insufficient
   for unified GPU memory.
8. Start followers, then rank 0. Rank 0 needs `API_HOST`, `PORT`, and an
   `API_KEY_FILE` mounted read-only when binding beyond loopback. Mount
   `capacity_middleware.py` as `vllm/amos_capacity.py` and set
   `AMOS_CAPACITY_MIDDLEWARE=1` to retain four-request admission.
9. Verify native API, generation, tools, and the actual client/router path;
   reopen application admission only after these checks pass.

Keep credentials in mounted files. The included node script reads the key
inside the container; no key value belongs in a Docker command or Git.

## Source and build status

[`runtime/vllm/`](../../runtime/vllm/README.md) preserves our integration
modules, patch installers, collective sources, and E3 source with licenses.
Some files are diagnostic helpers or inactive experiments; source presence
does not mean the P27 image executes them. The patch installers check specific
input hashes and are not a build-order script. Portable image assembly and a
general fleet launcher remain [explicit release work](../../docs/ROADMAP.md).

## Host persistence and recovery

See [operating notes](OPERATIONS.md) for the pinned Spark OS, permanent boot
selection, persistent dual-fabric settings, memory guards and recovery scope.
