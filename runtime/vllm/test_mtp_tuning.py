import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import amos_mtp_tuning as tuning
import amos_request_phase_mtp as phase
from patch_mtp_tuning import patched

PROFILE=json.loads(Path(__file__).with_name('amos_mtp_costs.json').read_text())


def controller():
    return phase.RequestPhaseController({int(n):row for n,row in PROFILE['cost_ms'].items()},PROFILE['prior_conditional'],16)


def request(key='r'):
    r=NS(request_id=key,prompt_token_ids=[phase.ASSISTANT,phase.THINK],output_token_ids=[],finished=False)
    r.is_finished=lambda:r.finished
    return r


def control(mode='original',depth=0,**extra):
    return dict(revision='test-'+mode,mode=mode,depth=depth,window=16,cost_ms=None,**extra)


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'control.json'
        self.flag=patch.dict('os.environ',{'AMOS_MTP_TUNING_CONTROL':'1'})
        self.flag.start();self.addCleanup(self.flag.stop)

    def write(self,value):
        self.path.write_text(json.dumps(value))

    def choose(self,c,rows,live):
        return tuning.choose(c,rows,live,self.path)

    def test_original_matches_native_choices_and_accounting_through_phase_changes(self):
        self.write(control());a,b=controller(),controller();ra,rb=request(),request()
        for step in range(96):
            self.assertEqual(a.choose([ra],{'r':ra}),self.choose(b,[rb],{'r':rb}))
            k=a.num_spec_tokens;accepted=step%(k+1)
            tokens=[42]*accepted+[phase.END_THINK if step==8 else (73022 if step in (20,60) else 42)]
            for c,r in [(a,ra),(b,rb)]:
                c.observe_request(r,k,accepted,tokens)
                r.output_token_ids.extend(tokens)
                c.observe_batch(num_drafts=1,num_draft_tokens=k,num_accepted_tokens=accepted)
            self.assertEqual(a.state(ra).phase,b.state(rb).phase)
            for name,stats in a.state(ra).histories.items():
                other=b.state(rb).histories[name]
                self.assertEqual(vars(stats),vars(other))

    def test_fixed_depth_latches_until_every_live_request_finishes(self):
        self.write(control('fixed',1));c=controller();r=request()
        self.assertEqual(self.choose(c,[r],{'r':r}),1)
        self.write(control('fixed',4))
        # A preempted request may be live without being scheduled.
        self.assertEqual(self.choose(c,[],{'r':r}),1)
        self.assertEqual(self.choose(c,[r],{'r':r}),1)
        r.finished=True
        self.assertEqual(self.choose(c,[],{}),4)
        self.assertEqual(json.loads(self.path.with_suffix('.applied.json').read_text())['depth'],4)

    def test_invalid_policy_and_pending_accounting_do_not_mutate_control(self):
        self.write(control());c=controller();self.choose(c,[],{})
        ack=self.path.with_suffix('.applied.json').read_bytes();costs=dict(c.costs)
        bad=control('costs');bad['cost_ms']={'1':[1,2,3,float('nan')]}
        self.write(bad)
        with self.assertRaises(ValueError):self.choose(c,[],{})
        self.assertEqual(c.costs,costs);self.assertEqual(self.path.with_suffix('.applied.json').read_bytes(),ack)
        self.write(control('fixed',2));r=request();c.observe_request(r,4,0,[42])
        with self.assertRaises(RuntimeError):self.choose(c,[r],{'r':r})
        self.assertEqual(self.path.with_suffix('.applied.json').read_bytes(),ack)

    def test_candidate_costs_change_only_choice_parameters(self):
        c=controller();original=dict(c.costs);prior=c.prior
        value=control('costs');value.update(window=8,cost_ms={str(n):[10,100,200,300] for n in range(1,5)})
        self.write(value);r=request()
        self.assertEqual(self.choose(c,[r],{'r':r}),1)
        self.assertEqual(c.prior,prior);self.assertEqual(c.observation_window,8)
        r.finished=True;self.write(control());self.choose(c,[],{})
        self.assertEqual(c.costs,original);self.assertEqual(c.observation_window,16)

    def test_missing_control_and_invalid_fields_fail_before_scheduling(self):
        with self.assertRaises(FileNotFoundError):self.choose(controller(),[],{})
        for change in ({'depth':True},{'depth':5},{'window':64},{'unexpected':0}):
            value=control('fixed',2);value.update(change)
            with self.assertRaises(ValueError):tuning.validate(value)

    def test_pinned_patch_changes_only_choice_wrapper(self):
        original=Path(__file__).with_name('amos_request_phase_mtp.py').read_bytes()
        result=patched(original)
        self.assertEqual(result.count(b'tuning_choose('),1)
        with self.assertRaises(ValueError):patched(result)
        compile(result,'patched-controller','exec')


if __name__=='__main__':
    unittest.main()
