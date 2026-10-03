# Operating the current six-Spark deployment

[Public closeout evidence](../../results/production-hardening.json).
Original 3.25 bpw weights and the qualified P27 inference configuration are retained.

## Pinned host and runtime

| Component | Verified deployment |
| --- | --- |
| OS | Kindling-derived `0.9.2-1019-64k-amos3` |
| Kernel | `7.0.0-1019-nvidia-64k`; 64 KiB pages |
| NVIDIA driver | `580.178.04` |
| Runtime | P27 image ID in the [serving recipe](README.md) |
| Fabric | Two validated NCCL rails and dual-HCA RoCEnante; addresses and MTU 9000 persistent |
| Memory setting | `vm.compaction_proactiveness=0`, persistent |
| Protection | Exact-container guard: 8 GiB available for 2 seconds, plus PSI/refault/swap checks |

Kindling Spark OS imports network configuration from the persistent host disk
into its runtime overlay. Persist changes in those host-disk sources, not only
the running overlay. Our five dedicated-fabric nodes use netplan; the node whose
second NIC also carries management uses its persistent NetworkManager connection,
preserving DHCP and the management default route. Validate the candidate offline
before changing persistent files.

The existing OS was promoted with its upstream boot-promotion mechanism. The
compaction setting is supplied as `sysctl.vm.compaction_proactiveness=0` on the
kernel command line. All six controlled reboots confirmed the selected OS,
network, setting, identity, original shard receipts and clear GPU-error checks.
This repository does not include our private site files or an OS image build.

Sources: [Kindling Spark OS](https://github.com/kindlingai/kindling-spark-os),
[Linux boot parameters](https://www.kernel.org/doc/html/v6.6/admin-guide/kernel-parameters.html).

## Maintenance and recovery

The completed [copy comparison](COPY_DRAFTING.md) selected original adaptive MTP
on the forward compatibility-fix image. Fresh short/8K/32K/128K numerical checks,
functional/cancellation checks and native/router/Code/Chat acceptance passed.
The final six-container inventory confirmed exclusive GPU ownership and clear
exact-container guards. This pass added no OS reboot or long-duration soak.

The earlier E35 boundary comparison selected the required
row32 / native-through32 controls in [E3 prefill](E3_PREFILL.md). Both cache
files must persist on every rank before a restart. Native/router/Code/Chat
acceptance passed on the selected image; no new OS reboot test was part of E35.

October 3: [dual-HCA RoCEnante](COMMUNICATION.md) was selected with unchanged
image/precision/weights after numerical and application acceptance. The HCA pair
and per-node shared GID-index checks are required on subsequent launches.

1. Reserve all six GPUs for full TP6. Stop competing model tests and their
   automatic restart policies before starting the fleet.
2. Drain application admission and hold the fleet supervisor for planned work.
3. Resume the current pinned image/configuration with fresh exact-container guards.
4. Check native identity, authentication, answers, streaming and capacity; then
   verify the actual router/client path before ending the maintenance window.
5. Investigate a guard trip before resuming. Keep the guard threshold and its
   pressure protections enabled. Never treat an expired observation as proof
   that the underlying process has stopped.

The supervisor restart retained all six live containers without another fleet
launch. Six host reboots were tested under an owned maintenance hold, followed
by manual resumption of the same configuration. Existing supervisor recovery
is bounded to three launches/hour. Any failed Spark interrupts the TP6 model;
no new arbitrary power-loss or unattended node-failure test was performed.

## Closeout and the competing-workload incident

A separate Flash canary started on two nodes during the first resume. Available
RAM on one node fell below the floor and its guard stopped the TP6 worker.
The canaries were stopped, their Docker restart policies disabled, and the same
TP6 configuration resumed. No weight, KV allocation or guard threshold changed.

Final native and Code/Chat checks passed, including streaming, tool calls and
memory integration. A 90.4-second observation retained all six containers with
clear guards; final minimum available memory was 13.14 GiB. GPU process ownership
was verified to belong exclusively to those containers. The user declined a
12-hour soak; none was launched. This is controlled-recovery and brief-acceptance
evidence, not a long-duration reliability claim.
