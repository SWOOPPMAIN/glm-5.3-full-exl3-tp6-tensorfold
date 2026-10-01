"""Choose bounded MTP depth from accepted prefixes and measured TP6 step costs.

Only the scheduler's next draft length changes. The drafter, target sampler,
rejection rule, token history, and existing graph dispatcher are untouched.
"""
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path

REFERENCE_IMAGE = 'sha256:ba7322e87e3050d67ad3f0b75d874f43edd1ac927166a4600ba4c1498aa9a3a6'
LOG = logging.getLogger('vllm.amos_cost_aware_mtp')


@dataclass(frozen=True)
class Update:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float


class CostAwareController:
    def __init__(self, costs, prior, observation_window=16):
        if set(costs) != {1,2,3,4} or not 4 <= observation_window <= 128:
            raise ValueError('Cost controller requires C1..C4 and a bounded window')
        if any(len(row)!=4 or any(not math.isfinite(v) or not 0<v<10000 for v in row) for row in costs.values()):
            raise ValueError('Four finite positive costs are required at every request count')
        if len(prior)!=4 or any(not math.isfinite(p) or not 0<p<1 for p in prior):
            raise ValueError('Four bounded conditional acceptance priors are required')
        self.costs = {n:tuple(row) for n,row in costs.items()}
        self.prior = tuple(prior)
        self.observation_window = observation_window
        self.max_num_spec_tokens = 4
        self.reset()

    def reset(self):
        self.num_spec_tokens = 4
        self.success = [2*p for p in self.prior]
        self.opportunities = [2.0]*4
        self.pending = []
        self.steps = 0
        self.since_probe = 0
        self.probing = False
        self.probe_remaining = 0
        self.last_requests = None

    def observe_request(self, attempted, accepted):
        if (type(attempted) is not int or type(accepted) is not int
                or not 1<=attempted<=4 or not 0<=accepted<=attempted
                or len(self.pending)>=4):
            raise ValueError('Invalid TP6 accepted-prefix observation')
        self.pending.append((attempted,accepted))

    def probabilities(self):
        probability = 1.0
        values = []
        for success,total in zip(self.success,self.opportunities):
            probability *= min(1.0,max(0.0,success/total))
            values.append(probability)
        return values

    def rates(self, requests):
        probability = self.probabilities()
        return [(1+sum(probability[:k]))*1000/self.costs[requests][k-1] for k in range(1,5)]

    def observe_batch(self, *, num_drafts, num_draft_tokens, num_accepted_tokens):
        if (num_drafts != len(self.pending)
                or num_draft_tokens != sum(k for k,_ in self.pending)
                or num_accepted_tokens != sum(a for _,a in self.pending)):
            raise ValueError('Per-request observations disagree with native batch counters')
        if not num_drafts:
            return None
        observations,self.pending = self.pending,[]
        # An unattempted suffix is censored, not a rejection. For position i,
        # only proposals that reached i and passed its prefix are informative.
        for i in range(4):
            eligible = sum(k>i and accepted>=i for k,accepted in observations)
            accepted = sum(k>i and accepted>i for k,accepted in observations)
            if eligible:
                self.opportunities[i] = .95*self.opportunities[i]+eligible
                self.success[i] = .95*self.success[i]+accepted
        self.steps += 1
        self.since_probe += 1
        previous = self.num_spec_tokens
        probe_observed = self.probing and any(k==4 for k,_ in observations)
        if probe_observed:
            self.probe_remaining -= 1
        probe_complete = probe_observed and self.probe_remaining == 0
        select = (self.steps % self.observation_window == 0
                  or (self.last_requests is not None and num_drafts != self.last_requests)
                  or probe_complete)
        rates = self.rates(num_drafts)
        # Async scheduling can deliver shallower proposals after requesting a
        # probe. Keep requesting depth4 until it is actually observed.
        if select and (not self.probing or probe_complete):
            best = max(range(1,5),key=lambda k:rates[k-1])
            if probe_complete or rates[best-1] > rates[previous-1]*1.015:
                self.num_spec_tokens = best
        if probe_complete:
            self.probing = False
            self.since_probe = 0
        elif not self.probing and self.since_probe >= 4*self.observation_window:
            self.num_spec_tokens = 4
            self.probing = True
            self.probe_remaining = 4
        self.last_requests = num_drafts
        if self.num_spec_tokens != previous:
            LOG.info('AMOS MTP cost depth %d -> %d, requests=%d, expected rates=%s, probe=%s',
                     previous,self.num_spec_tokens,num_drafts,[round(x,2) for x in rates],self.probing)
        return Update(previous,self.num_spec_tokens,num_accepted_tokens/num_drafts,num_draft_tokens/num_drafts)


def install(scheduler, config):
    flag = os.getenv('AMOS_MTP_COST_AWARE','0')
    if flag not in ('0','1'):
        raise ValueError('AMOS_MTP_COST_AWARE must be 0 or 1')
    if flag=='0':
        return
    parallel = config.parallel_config
    model = config.model_config.hf_text_config
    prior = scheduler.acceptance_length_controller
    if (model.model_type!='glm_moe_dsa' or model.hidden_size!=6144 or model.num_hidden_layers!=78
            or parallel.tensor_parallel_size!=6 or parallel.pipeline_parallel_size!=1
            or parallel.data_parallel_size!=1 or parallel.decode_context_parallel_size!=1
            or scheduler.num_spec_tokens!=4 or scheduler.scheduler_config.max_num_seqs!=4
            or scheduler.dynamic_sd_lookup is not None or prior is None
            or os.getenv('AMOS_MTP_CALIBRATION','0')!='0'
            or os.getenv('AMOS_TP6_DYNAMIC_GRAPHS')!='1'):
        raise ValueError('Cost-aware MTP requires pinned full GLM TP6, C4, MTP4 adaptive graphs')
    policy = json.loads(Path(__file__).with_name('amos_mtp_costs.json').read_text())
    if policy['schema']!=1 or policy['reference_image']!=REFERENCE_IMAGE:
        raise ValueError('Cost profile does not match the qualified TP6 runtime')
    scheduler.acceptance_length_controller = CostAwareController(
        {int(n):values for n,values in policy['cost_ms'].items()},
        policy['prior_conditional'],prior.observation_window)
    LOG.info('AMOS MTP cost controller installed; window=%d, C1..C4, depth1..4',prior.observation_window)


def observe_request(controller, attempted, accepted):
    if isinstance(controller,CostAwareController):
        controller.observe_request(attempted,accepted)
