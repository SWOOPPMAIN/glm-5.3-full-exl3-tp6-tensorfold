"""Eager adapter from the request state machine to the qualified full TP6 model.

Use one existing admitted communicator/model/cache/workspace on every rank.
This adapter deliberately does not launch workers or allocate another model.
Graphs, command distribution and API scheduling are later integration layers.
"""
import torch

from tensorfold.cuda.sampling import sample_rows
from .indexer_plan import visible_token_bound
from .model import FullModel
from .request import Extent


class FullModelBackend:
    def __init__(self, model, caches, table, workspace):
        if not isinstance(model, FullModel) or workspace.weights is not model.weights:
            raise ValueError('Request backend needs its admitted full-model workspace')
        cfg = model.weights.config
        if (len(caches) != 79 or [c.layer for c in caches] != list(range(79))
                or len({c.capacity for c in caches}) != 1
                or any((c.index_keys is not None) != (cfg.indexer_source(c.layer) == c.layer) for c in caches)):
            raise ValueError('Request backend needs all independently owned layer caches')
        self.device = workspace.hidden.device
        self.capacity = caches[0].capacity
        if (self.device.type != 'cuda' or table.device != self.device or len(table) < self.capacity
                or workspace.decoder.attention.indexer.capacity != self.capacity
                or any(c.latent.device != self.device for c in caches)):
            raise ValueError('Cache, RoPE and workspace devices/capacities disagree')
        self.model, self.caches, self.table, self.workspace = model, tuple(caches), table, workspace
        self.rows, self.logit_rows = workspace.rows, workspace.vocab.logit_rows
        self.vocab, self.eos = cfg.vocab, cfg.eos
        self.stream = torch.cuda.current_stream(self.device)

    def _stream(self):
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('The admitted request workspace has one CUDA stream owner')

    def _inputs(self, tokens, start, extent):
        self._stream()
        n = len(tokens)
        if (not isinstance(extent, Extent) or type(start) is not int or start < 0
                or not 1 <= n <= self.rows or start+n > extent.size
                or extent.base < 0 or extent.base+extent.size > self.capacity
                or any(type(t) is not int or not 0 <= t < self.vocab for t in tokens)):
            raise ValueError('Forward rows exceed the owned cache extent')
        host_positions = list(range(start, start+n))
        visible = visible_token_bound(host_positions, extent.size)
        ids = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        positions = torch.tensor(host_positions, dtype=torch.int64, device=self.device)
        bases = torch.full_like(positions, extent.base)
        slots = positions+bases
        return ids, positions, bases, slots, visible

    @torch.inference_mode()
    def target(self, tokens, start, extent):
        ids, pos, bases, slots, visible = self._inputs(tokens, start, extent)
        return self.model.target_forward(ids, pos, bases, slots, self.caches[:78], self.table,
                                         self.workspace, scope=object(), visible_tokens=visible)

    @torch.inference_mode()
    def mtp(self, tokens, hidden, start, extent):
        ids, pos, bases, slots, visible = self._inputs(tokens, start, extent)
        return self.model.mtp_forward(ids, hidden, pos, bases, slots, self.caches[78], self.table,
                                      self.workspace, scope=object(), visible_tokens=visible)

    @torch.inference_mode()
    def sample(self, hidden, positions, sampling):
        self._stream()
        if len(hidden) != len(positions) or not 1 <= len(hidden) <= self.logit_rows:
            raise ValueError('Sample rows exceed the admitted vocabulary workspace')
        # Vocabulary.project gathers in original global token order and removes
        # TP6 padding. Every rank uses the same seed and absolute positions.
        return sample_rows(self.model.logits(hidden.contiguous(), self.workspace), positions, sampling)

    def synchronize(self):
        self._stream()
        self.stream.synchronize()
