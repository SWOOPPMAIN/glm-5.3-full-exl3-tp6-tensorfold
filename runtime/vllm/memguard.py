#!/usr/bin/python3 -I
"""Container-ID-bound host-liveness guard derived from the installed int4mix v2 guard.

The launcher writes an exact Docker ID before starting this guard. It never
scans unrelated vLLM containers and never reboots a host. Pressure stops only
that container using cgroup.kill, with per-process SIGKILL as a fallback.
The sampling loop uses locked memory and real-time scheduling; a background
thread writes status and drops clean page cache under host-memory pressure.
"""
import ctypes
import json
import os
import re
import select
import socket
import threading
import time

E = os.environ.get
GIB = 1 << 30
PAGE = os.sysconf("SC_PAGE_SIZE")
NAME = E("NAME", "glm53-int4mix-tp4")
STATE = E("STATE_DIR", "/run/amos/int4mix")
CIDFILE = STATE + "/container-id"          # written by `docker run --cidfile`
TRIPFILE = STATE + "/tripped"
STATUSFILE = STATE + "/guard-status"
PERIOD = float(E("PERIOD_S", "0.25"))
HARD_AVAIL = float(E("HARD_AVAIL_GIB", "2")) * GIB      # MemAvailable floor ...
HARD_AVAIL_S = float(E("HARD_AVAIL_S", "1"))           # ... sustained this long
PSI_TRIGGER = E("PSI_TRIGGER", "full 700000 1000000")  # >=70% full stall in a 1 s window
PSI_EVENTS = int(E("PSI_EVENTS", "3"))                  # this many trigger events ...
PSI_EVENT_WINDOW_S = float(E("PSI_EVENT_WINDOW_S", "5"))  # ... within this window
PSI_FULL_AVG10 = float(E("PSI_FULL_AVG10", "60"))
REFAULT_BPS = float(E("REFAULT_MIB_S", "200")) * (1 << 20)   # workingset_refault_file
SWAPOUT_BPS = float(E("SWAPOUT_MIB_S", "128")) * (1 << 20)   # pswpout
RATE_S = float(E("RATE_WINDOW_S", "2"))
SELF_STALL_S = float(E("SELF_STALL_S", "1.5"))
CANARY_S = float(E("SSH_CANARY_S", "10"))              # 0 disables
CANARY_ADDR = E("SSH_CANARY_ADDR", "127.0.0.1")  # an address sshd listens on
CANARY_TIMEOUT_S = float(E("SSH_CANARY_TIMEOUT_S", "5"))
CANARY_FAILS = int(E("SSH_CANARY_FAILS", "2"))
ASSIST_FREE = float(E("ASSIST_FREE_GIB", "16")) * GIB  # drop clean cache below this MemFree
ASSIST_MIN_CACHE = float(E("ASSIST_MIN_CACHE_GIB", "2")) * GIB
ASSIST_EVERY_S = float(E("ASSIST_EVERY_S", "2"))
REBOOT_AFTER_S = 0.0  # A stale trip must never cause a future host reboot.
HEALTHY_AVAIL = 8 * GIB          # only touch tmpfs/procfs discovery while this healthy
DRY = E("DRY_RUN", "0") == "1"
SCOPE_OVERRIDE = E("SCOPE_OVERRIDE", "")  # tests only
CGROOT = "/sys/fs/cgroup"
HEX64 = re.compile(rb"^[0-9a-f]{64}$")
MEMKEYS = (b"MemFree", b"MemAvailable", b"Active(file)", b"Inactive(file)", b"Mapped",
           b"SwapTotal", b"SwapFree")


def rd(fd, size=65536):
    return os.pread(fd, size, 0)


class Guard:
    def __init__(self):
        self.kmsg = None
        try:
            self.kmsg = os.open("/dev/kmsg", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError:
            pass
        self.fd_mem = os.open("/proc/meminfo", os.O_RDONLY | os.O_CLOEXEC)
        self.fd_vm = os.open("/proc/vmstat", os.O_RDONLY | os.O_CLOEXEC)
        self.fd_psi = os.open("/proc/pressure/memory", os.O_RDONLY | os.O_CLOEXEC)
        self.poller = select.poll()
        self.trig = None
        try:
            self.trig = os.open("/proc/pressure/memory", os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
            os.write(self.trig, PSI_TRIGGER.encode() + b"\0")
            self.poller.register(self.trig, select.POLLPRI)
        except OSError as e:
            self.log(4, f"PSI trigger unavailable ({e}); using avg10 polling only")
            self.trig = None
        self.cid = None
        self.cid_key = None
        self.scope = SCOPE_OVERRIDE or None
        self.next_discover = 0.0
        self.tripped_at = None
        self.trip_reason = ""
        self.last_kill = 0.0
        self.rebooted = False
        self.psi_events = []
        self.rates = []          # (t, refault_pages, pswpout_pages)
        self.low_since = None
        self.canary_ok_once = False
        self.canary_fails = 0
        self.stats = {"min_avail_gib": 1e9, "max_psi_full10": 0.0,
                      "max_refault_mib_s": 0.0, "max_swapout_mib_s": 0.0,
                      "max_loop_gap_s": 0.0, "psi_events": 0}
        self.snapshot = {}
        self.work = threading.Event()
        self.work_lock = threading.Lock()
        self.pending = {}
        self.assist_last = 0.0

    # ---- logging: kmsg first (never blocks on journald) ----
    def log(self, prio, msg):
        line = f"<{prio}>glm53-tp6-memguard: {msg}"[:900].encode()
        if self.kmsg is not None:
            try:
                os.write(self.kmsg, line)
                return
            except OSError:
                pass
        if DRY:
            print(line.decode(), flush=True)

    # ---- sampling (no allocation-heavy work, no fork) ----
    def sample(self):
        mem = {}
        for line in rd(self.fd_mem, 16384).split(b"\n"):
            k, _, v = line.partition(b":")
            if k in MEMKEYS:
                mem[k] = int(v.split()[0]) * 1024
        vm = {}
        for line in rd(self.fd_vm).split(b"\n"):
            k, _, v = line.partition(b" ")
            if k in (b"workingset_refault_file", b"pswpout"):
                vm[k] = int(v)
        full10 = 0.0
        for line in rd(self.fd_psi, 512).split(b"\n"):
            if line.startswith(b"full "):
                full10 = float(line.split()[1].split(b"=")[1])
        return mem, vm, full10

    # ---- container identity: cached, refreshed only while healthy ----
    def scope_for(self, cid):
        for p in (f"{CGROOT}/system.slice/docker-{cid}.scope", f"{CGROOT}/docker/{cid}"):
            if os.path.isdir(p):
                return p
        return None

    def refresh_identity(self, now, healthy):
        if SCOPE_OVERRIDE or not healthy:
            return
        try:
            st = os.stat(CIDFILE)
            key = (st.st_ino, st.st_mtime_ns)
            if key != self.cid_key:
                with open(CIDFILE, "rb") as f:
                    raw = f.read(128).strip()
                if HEX64.match(raw):
                    self.cid_key = key
                    if raw.decode() != self.cid:
                        self.cid = raw.decode()
                        self.scope = None
                        if self.tripped_at is not None and not os.path.exists(TRIPFILE):
                            self.tripped_at = None      # launcher re-armed for a new container
                            self.log(5, f"re-armed for container {self.cid[:12]}")
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if self.cid and (self.scope is None or not os.path.isdir(self.scope)):
            self.scope = self.scope_for(self.cid)

    @staticmethod
    def procs(scope):
        try:
            with open(scope + "/cgroup.procs", "rb") as f:
                return [int(x) for x in f.read().split()]
        except (OSError, ValueError):
            return []

    # ---- the kill: one write to cgroup.kill, fallbacks after ----
    def kill(self, reason):
        self.last_kill = time.monotonic()
        if self.scope is None and not SCOPE_OVERRIDE:
            self.scope = self.scope_for(self.cid) if self.cid else None
        if self.scope is None:
            return
        how = "none"
        if self.scope and not DRY:
            try:
                fd = os.open(self.scope + "/cgroup.kill", os.O_WRONLY | os.O_CLOEXEC)
                try:
                    os.write(fd, b"1")
                finally:
                    os.close(fd)
                how = "cgroup.kill"
            except OSError as e:
                how = f"cgroup.kill-errno{e.errno}"
                for pid in self.procs(self.scope):
                    try:
                        os.kill(pid, 9)
                        how = how + "+sigkill"
                    except OSError:
                        pass
        if self.tripped_at is None:
            self.tripped_at = self.last_kill
            self.trip_reason = reason
            self.post("trip", {"reason": reason, "how": how, "cid": self.cid,
                               "scope": self.scope, "at": time.time(), "dry": DRY})
        self.log(2, f"KILL ({how}) {NAME} scope={self.scope}: {reason}")

    # ---- background worker: drop_caches assist + tmpfs status/trip writes ----
    def post(self, kind, payload):
        with self.work_lock:
            self.pending[kind] = payload
        self.work.set()

    def worker(self):
        try:
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))  # this thread only
        except (OSError, AttributeError):
            pass
        while True:
            self.work.wait()
            self.work.clear()
            with self.work_lock:
                jobs, self.pending = self.pending, {}
            if "trip" in jobs:
                self.write_file(TRIPFILE, json.dumps(jobs["trip"]) + "\n")
            if "drop" in jobs and not DRY:
                try:
                    fd = os.open("/proc/sys/vm/drop_caches", os.O_WRONLY)
                    try:
                        os.write(fd, b"1")
                    finally:
                        os.close(fd)
                except OSError:
                    pass
            if "status" in jobs:
                self.write_file(STATUSFILE, json.dumps(jobs["status"]) + "\n")

    @staticmethod
    def write_file(path, text):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        except OSError:
            pass

    # ---- sshd canary: the livelock symptom itself (self-enables after first success) ----
    def canary(self):
        try:
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        except (OSError, AttributeError):
            pass
        while True:
            ok = False
            try:
                with socket.create_connection((CANARY_ADDR, 22), timeout=CANARY_TIMEOUT_S) as s:
                    s.settimeout(CANARY_TIMEOUT_S)
                    ok = s.recv(64).startswith(b"SSH-")
            except OSError:
                ok = False
            if ok:
                self.canary_ok_once = True
                self.canary_fails = 0
            elif self.canary_ok_once:
                self.canary_fails += 1
            time.sleep(CANARY_S)

    # ---- main loop ----
    def run(self):
        threading.stack_size(256 * 1024)
        threading.Thread(target=self.worker, daemon=True).start()
        if CANARY_S > 0:
            threading.Thread(target=self.canary, daemon=True).start()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.mlockall(1 | 2) != 0:           # MCL_CURRENT | MCL_FUTURE
            self.log(3, f"mlockall failed errno={ctypes.get_errno()}: guard can be paged out")
        self.log(5, f"armed: floor {HARD_AVAIL / GIB:.1f} GiB/{HARD_AVAIL_S}s, psi '{PSI_TRIGGER}' "
                    f"x{PSI_EVENTS}/{PSI_EVENT_WINDOW_S}s, avg10>={PSI_FULL_AVG10}, refault>="
                    f"{REFAULT_BPS / 2**20:.0f}MiB/s, swapout>={SWAPOUT_BPS / 2**20:.0f}MiB/s, "
                    f"self-stall>{SELF_STALL_S}s, canary {CANARY_S}s, dry={DRY}")
        last = time.monotonic()
        next_status = 0.0
        loops = 0
        while True:
            events = self.poller.poll(int(PERIOD * 1000))
            now = time.monotonic()
            gap, last = now - last, now
            loops += 1
            for fd, ev in events:
                if ev & select.POLLERR:
                    self.poller.unregister(fd)
                    self.log(4, "PSI trigger fd error; avg10 polling only")
                elif ev & select.POLLPRI:
                    self.psi_events.append(now)
                    self.stats["psi_events"] += 1
            mem, vm, full10 = self.sample()
            avail, free = mem.get(b"MemAvailable", 0), mem.get(b"MemFree", 0)
            healthy = avail > HEALTHY_AVAIL
            self.refresh_identity(now, healthy)

            self.rates.append((now, vm.get(b"workingset_refault_file", 0), vm.get(b"pswpout", 0)))
            while len(self.rates) > 2 and now - self.rates[1][0] >= RATE_S:
                self.rates.pop(0)
            t0, rf0, so0 = self.rates[0]
            span = max(now - t0, 1e-3)
            refault = (self.rates[-1][1] - rf0) * PAGE / span
            swapout = (self.rates[-1][2] - so0) * PAGE / span
            self.psi_events = [t for t in self.psi_events if now - t <= PSI_EVENT_WINDOW_S]

            st = self.stats
            st["min_avail_gib"] = min(st["min_avail_gib"], avail / GIB)
            st["max_psi_full10"] = max(st["max_psi_full10"], full10)
            if span >= RATE_S * 0.9:
                st["max_refault_mib_s"] = max(st["max_refault_mib_s"], refault / 2**20)
                st["max_swapout_mib_s"] = max(st["max_swapout_mib_s"], swapout / 2**20)
            if loops > 4:
                st["max_loop_gap_s"] = max(st["max_loop_gap_s"], gap)

            reasons = []
            if loops > 4 and gap > SELF_STALL_S:
                reasons.append(f"guard loop stalled {gap:.2f}s (mlocked RT task)")
            if avail < HARD_AVAIL:
                self.low_since = self.low_since or now
                if now - self.low_since >= HARD_AVAIL_S:
                    reasons.append(f"MemAvailable {avail / GIB:.2f} GiB < floor for {now - self.low_since:.1f}s")
            else:
                self.low_since = None
            if len(self.psi_events) >= PSI_EVENTS:
                reasons.append(f"PSI '{PSI_TRIGGER}' fired {len(self.psi_events)}x in {PSI_EVENT_WINDOW_S}s")
            if full10 >= PSI_FULL_AVG10:
                reasons.append(f"PSI memory full avg10 {full10:.1f}%")
            if span >= RATE_S * 0.9 and refault >= REFAULT_BPS:
                reasons.append(f"file refaults {refault / 2**20:.0f} MiB/s (thrashing)")
            if span >= RATE_S * 0.9 and swapout >= SWAPOUT_BPS:
                reasons.append(f"swap-out {swapout / 2**20:.0f} MiB/s")
            if self.canary_fails >= CANARY_FAILS:
                reasons.append(f"sshd banner canary failed {self.canary_fails}x")

            armed = self.scope is not None
            if reasons and armed:
                if self.tripped_at is None or (now - self.last_kill >= 1.0 and self.procs(self.scope)):
                    self.kill("; ".join(reasons))
            elif reasons and now - self.last_kill >= 10.0:
                self.last_kill = now     # rate-limit the no-container warning
                self.log(4, "pressure with no vllm container to kill: " + "; ".join(reasons))

            if (self.tripped_at is not None and REBOOT_AFTER_S > 0 and not self.rebooted
                    and now - self.tripped_at >= REBOOT_AFTER_S
                    and (avail < HARD_AVAIL or full10 >= PSI_FULL_AVG10 or gap > SELF_STALL_S)):
                self.rebooted = True
                self.log(0, f"host still wedged {now - self.tripped_at:.0f}s after kill: sysrq s,b")
                if not DRY:
                    try:
                        fd = os.open("/proc/sysrq-trigger", os.O_WRONLY)
                        os.write(fd, b"s")
                        time.sleep(2)
                        os.write(fd, b"b")
                    except OSError:
                        pass

            cache = mem.get(b"Active(file)", 0) + mem.get(b"Inactive(file)", 0) - mem.get(b"Mapped", 0)
            if (free < ASSIST_FREE and cache > ASSIST_MIN_CACHE
                    and now - self.assist_last >= ASSIST_EVERY_S):
                self.assist_last = now
                self.post("drop", True)

            if now >= next_status:
                next_status = now + 2.0
                self.post("status", {
                    "ts": time.time(), "cid": self.cid, "scope": self.scope, "armed": armed,
                    "tripped": self.tripped_at is not None, "trip_reason": self.trip_reason,
                    "avail_gib": round(avail / GIB, 2), "free_gib": round(free / GIB, 2),
                    "psi_full10": full10, "refault_mib_s": round(refault / 2**20, 1),
                    "swapout_mib_s": round(swapout / 2**20, 1), "canary_ok_once": self.canary_ok_once,
                    "canary_fails": self.canary_fails, "dry": DRY,
                    **{k: round(v, 2) for k, v in self.stats.items()}})


if __name__ == "__main__":
    Guard().run()
