"""Leader-owned sampling before any EOS or recursive-MTP branch.

Uses the already admitted TP6 NCCL group. The fixed-size decision includes an
error flag so a leader-side sampler failure still reaches every follower.
This eager path does not create a process group or claim graph compatibility.
"""
import torch
import torch.distributed as dist

from tensorfold.cuda.sampling import sample_rows


def broadcast_decision(logits, positions, sampling, *, rank, packet, broadcast, choose=sample_rows):
    """Transport-independent decision body; the CUDA wrapper owns its buffer."""
    rows = len(positions)
    error = None
    if rank == 0:
        packet.zero_()
        try:
            if logits.ndim != 2 or logits.shape[0] != rows or not 1 <= rows <= len(packet)-2:
                raise ValueError('Sampling rows exceed the admitted decision buffer')
            tokens = list(choose(logits, positions, sampling))
            if len(tokens) != rows or any(type(t) is not int or not 0 <= t < logits.shape[1] for t in tokens):
                raise ValueError('Leader sampler returned invalid token IDs')
            packet[2:2+rows] = torch.tensor(tokens, device=packet.device, dtype=torch.int64)
            packet[0], packet[1] = 1, rows
        except Exception as exc:
            packet.zero_()
            error = exc
    broadcast(packet)
    values = packet.cpu().tolist()
    if values[0] != 1:
        raise RuntimeError('TP6 leader sampling failed on this decision') from error
    if (values[1] != rows or logits.ndim != 2 or logits.shape[0] != rows
            or not 1 <= rows <= len(packet)-2
            or any(not 0 <= t < logits.shape[1] for t in values[2:2+rows])):
        raise RuntimeError('TP6 sampling decision geometry differs between ranks')
    return values[2:2+rows]


class RankZeroSampler:
    def __init__(self, group, device, max_rows):
        device = torch.device(device)
        if (not dist.is_initialized() or dist.get_world_size(group) != 6
                or dist.get_backend(group) != 'nccl' or device.type != 'cuda'
                or type(max_rows) is not int or not 1 <= max_rows <= 128):
            raise ValueError('Leader sampler requires the admitted six-rank NCCL group and CUDA device')
        # src=0 names global rank zero; this backend uses the full six-rank group.
        self.rank = dist.get_rank(group)
        if dist.get_global_rank(group, self.rank) != self.rank:
            raise ValueError('TP6 sampler requires group ranks equal to global ranks')
        self.group, self.device = group, device
        self.packet = torch.empty(max_rows+2, dtype=torch.int64, device=device)
        self.stream = torch.cuda.current_stream(device)
        self.decisions = 0

    @torch.inference_mode()
    def __call__(self, logits, positions, sampling):
        if logits.device != self.packet.device or torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('TP6 sampler requires its owning CUDA stream/device')
        result = broadcast_decision(logits, positions, sampling, rank=self.rank, packet=self.packet,
            broadcast=lambda packet: dist.broadcast(packet, src=0, group=self.group))
        self.decisions += 1
        return result
