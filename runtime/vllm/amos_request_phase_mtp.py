"""Request/phase-isolated acceptance with a shared TP6 draft-depth decision.

Uses the existing measured C1..C4 costs and fixed prior. No statistics are
shared between requests. Phase markers are verified against the checkpoint
tokenizer at installation. Only future proposal length changes; target
sampling, rejection and token history remain native.
"""
from dataclasses import dataclass
import hashlib
import logging
import math
import os
from pathlib import Path

LOG = logging.getLogger('vllm.amos_request_phase_mtp')
TOKENIZER_SHA256 = '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
ASSISTANT, THINK, END_THINK, TOOL, END_TOOL = 154828, 154841, 154842, 154843, 154844
FENCES = frozenset((41002, 53913, 73022))
MAX_STATES = 128


def next_phase(phase, token):
    if token == THINK:
        return 'reasoning'
    if token in (END_THINK, END_TOOL):
        return 'content'
    if token == TOOL:
        return 'tool'
    if token in FENCES and phase in ('content', 'code'):
        return 'code' if phase == 'content' else 'content'
    return phase


def initial_phase(prompt):
    # User messages can quote markers. Only an assistant generation prefix
    # establishes an initial phase; scan backwards once, never re-tokenize.
    if prompt:
        for i in range(len(prompt)-1, -1, -1):
            if prompt[i] == ASSISTANT:
                phase = 'content'
                for token in prompt[i+1:]:
                    phase = next_phase(phase, token)
                return phase
            if prompt[i] in (154826, 154827, 154829):
                break
    return 'content'


class PhaseStats:
    def __init__(self, prior, window):
        self.success = [2*p for p in prior]
        self.opportunities = [2.0]*4
        self.window = window
        self.since_probe = 0
        self.probe_remaining = 0

    def probabilities(self):
        result, product = [], 1.0
        for success, total in zip(self.success, self.opportunities):
            product *= min(1.0, max(0.0, success/total))
            result.append(product)
        return result

    def observe(self, trials, attempted, starts_here):
        for position, accepted in trials:
            self.opportunities[position] = .95*self.opportunities[position]+1
            self.success[position] = .95*self.success[position]+int(accepted)
        # An actual full-depth proposal starting in this phase counts toward
        # its probe. Delayed shallow outputs cannot prematurely end a probe.
        probing = bool(self.probe_remaining)
        if probing and starts_here and attempted == 4:
            self.probe_remaining -= 1
            if not self.probe_remaining:
                self.since_probe = 0
        elif not probing:
            self.since_probe += 1
            if self.since_probe >= 4*self.window:
                self.probe_remaining = 4


class RequestState:
    def __init__(self, request, prior, window):
        self.request = request
        self.prior, self.window = prior, window
        self.phase = initial_phase(request.prompt_token_ids)
        self.cursor = 0
        self.histories = {}
        self.last_used = 0

    def sync(self):
        tokens = self.request.output_token_ids
        if len(tokens) < self.cursor:
            # A changed canonical history invalidates only policy statistics.
            self.phase = initial_phase(self.request.prompt_token_ids)
            self.cursor = 0
            self.histories.clear()
        for token in tokens[self.cursor:]:
            self.phase = next_phase(self.phase, token)
        self.cursor = len(tokens)

    def stats(self, phase):
        if phase not in self.histories:
            self.histories[phase] = PhaseStats(self.prior, self.window)
        return self.histories[phase]


@dataclass(frozen=True)
class Update:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float


class RequestPhaseController:
    def __init__(self, costs, prior, observation_window=16):
        if (set(costs) != {1, 2, 3, 4} or not 4 <= observation_window <= 128
                or len(prior) != 4 or any(not math.isfinite(p) or not 0 < p < 1 for p in prior)
                or any(len(v) != 4 or any(not math.isfinite(x) or not 0 < x < 10000 for x in v)
                       for v in costs.values())):
            raise ValueError('Invalid TP6 cost policy')
        self.costs = {n: tuple(v) for n, v in costs.items()}
        self.prior = tuple(prior)
        self.observation_window = observation_window
        self.max_num_spec_tokens = 4
        self.num_spec_tokens = 4
        self.states = {}
        self.pending = []
        self.clock = 0
        self.last_signature = None

    def state(self, request):
        key = request.request_id
        state = self.states.get(key)
        if state is None or state.request is not request:
            if len(self.states) >= MAX_STATES and key not in self.states:
                pending = {id(row[0]) for row in self.pending}
                eligible = [k for k, s in self.states.items() if id(s) not in pending]
                if not eligible:
                    raise RuntimeError('All bounded request states have pending observations')
                del self.states[min(eligible, key=lambda k: self.states[k].last_used)]
            state = self.states[key] = RequestState(request, self.prior, self.observation_window)
        state.sync()
        self.clock += 1
        state.last_used = self.clock
        return state

    def observe_request(self, request, attempted, accepted, tokens, stale=False):
        if (type(attempted) is not int or type(accepted) is not int
                or not 1 <= attempted <= 4 or not 0 <= accepted <= attempted
                or len(self.pending) >= 4 or len(tokens) < accepted):
            raise ValueError('Invalid request acceptance observation')
        state = self.state(request)
        phase = start = state.phase
        trials = {}
        if not stale:
            # Each position belongs to the phase BEFORE its proposal. A
            # boundary token may change the phase of later accepted positions.
            for position in range(min(attempted, accepted+1)):
                passed = position < accepted
                trials.setdefault(phase, []).append((position, passed))
                if passed:
                    phase = next_phase(phase, tokens[position])
        self.pending.append((state, attempted, accepted, start, trials))

    def observe_batch(self, *, num_drafts, num_draft_tokens, num_accepted_tokens):
        if (num_drafts != len(self.pending)
                or num_draft_tokens != sum(x[1] for x in self.pending)
                or num_accepted_tokens != sum(x[2] for x in self.pending)):
            raise ValueError('Request observations disagree with native batch counters')
        if not num_drafts:
            return None
        rows, self.pending = self.pending, []
        for state, attempted, _, start, trials in rows:
            for phase, positions in trials.items():
                state.stats(phase).observe(positions, attempted, phase == start)
        # Selection happens with the actual next scheduled batch, after native
        # request completion, admission and accepted-output history updates.
        return Update(self.num_spec_tokens, self.num_spec_tokens,
                      num_accepted_tokens/num_drafts, num_draft_tokens/num_drafts)

    def choose(self, scheduled, live):
        if self.pending:
            raise RuntimeError('Cannot select depth before batch accounting completes')
        for key, state in list(self.states.items()):
            if live.get(key) is not state.request or state.request.is_finished():
                del self.states[key]
        requests = [r for r in scheduled if not r.is_finished()]
        if not requests:
            self.last_signature = None
            return self.num_spec_tokens
        if not 1 <= len(requests) <= 4 or len({id(r) for r in requests}) != len(requests):
            raise ValueError('Request-phase policy requires a unique C1..C4 batch')
        states = [self.state(r) for r in requests]
        stats = [s.stats(s.phase) for s in states]
        signature = frozenset((id(s.request), s.phase) for s in states)
        probability = [s.probabilities() for s in stats]
        count = len(states)
        rates = [(1+sum(sum(p[:k]) for p in probability)/count)*1000/self.costs[count][k-1]
                 for k in range(1, 5)]
        previous = self.num_spec_tokens
        if any(s.probe_remaining for s in stats):
            selected = 4
        else:
            best = max(range(1, 5), key=lambda k: rates[k-1])
            selected = best if (signature != self.last_signature
                               or rates[best-1] > rates[previous-1]*1.015) else previous
        self.num_spec_tokens = selected
        self.last_signature = signature
        if selected != previous:
            LOG.info('AMOS request-phase MTP depth %d -> %d, requests=%d, phases=%s',
                     previous, selected, count, sorted({s.phase for s in states}))
        return selected


def install(scheduler, config):
    flag = os.getenv('AMOS_MTP_REQUEST_PHASE', '0')
    if flag not in ('0', '1'):
        raise ValueError('Invalid request-phase MTP setting')
    if flag == '0':
        return
    from vllm.amos_cost_aware_mtp import CostAwareController
    current = scheduler.acceptance_length_controller
    if type(current) is not CostAwareController or os.getenv('AMOS_MTP_COST_AWARE') != '1':
        raise ValueError('Request-phase MTP requires the validated TP6 cost controller')
    tokenizer = Path(config.model_config.model)/'tokenizer.json'
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Request-phase markers do not match the pinned tokenizer')
    scheduler.acceptance_length_controller = RequestPhaseController(
        current.costs, current.prior, current.observation_window)
    LOG.info('AMOS request-phase MTP installed; isolated histories, shared C1..C4 depth')


def observe_request(controller, request, attempted, accepted, tokens, stale):
    if isinstance(controller, RequestPhaseController):
        controller.observe_request(request, attempted, accepted, tokens, stale)
    else:
        from vllm.amos_cost_aware_mtp import observe_request as previous
        previous(controller, attempted, accepted)


def choose(controller, scheduled_ids, live):
    if isinstance(controller, RequestPhaseController):
        controller.choose([live[r] for r in scheduled_ids if r in live], live)
