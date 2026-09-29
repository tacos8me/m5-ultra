"""CPU-only cancellation regression, exercising the actual Job.run/_attempt code."""
import ast
from pathlib import Path
import json,logging,threading,time
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
def job_class():
 source=(ROOT/'omlx/patches/deepseek_v41/og_model.py').read_text()
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Job')
 ns=dict(threading=threading,time=time,logger=logging.getLogger('test'),COPY_PREBUILD=False,
         STORE=None,NUMERICS=[None],DELTA_MIN=4096,KICKOFF=True,STATS=dict(kickoff_sent=0,kickoff_errors=0,open_retries=0),
         fe_trace=SimpleNamespace(og_stamp=lambda *a,**k:None),
         pipe_wire=SimpleNamespace(BoxBusy=type('BoxBusy',(OSError,),{}),RESUME_TRIES=5,wait_limit=lambda _:30),
         og_images=SimpleNamespace(prompt_keys=lambda t,i:t))
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'omlx/patches/deepseek_v41/og_model.py'),'exec'),ns)
 return ns['Job'],ns
class Encoder:
 def __init__(self): self.sent=0;self.closed=0;self.open_info={}
 def send_step(self,*a):self.sent+=1
 def close(self):self.closed+=1
class Cancel(unittest.TestCase):
 def test_before_start_no_open(self):
  Job,_=job_class();j=Job('fake',0,'test',[1,2]); seen=[]
  j._attempt=lambda:seen.append(1);j.cancelled=True;j.cancel_event.set();j.run()
  self.assertEqual(seen,[]);self.assertTrue(j.done.is_set())
 def test_during_retry_no_second_open(self):
  Job,_=job_class();j=Job('fake',0,'test',[1,2]);hit=threading.Event();calls=[]
  def fail():calls.append(1);hit.set();raise OSError('offline')
  j._attempt=fail;j.start();self.assertTrue(hit.wait(1));j.cancelled=True;j.cancel_event.set();j.join(.25)
  self.assertFalse(j.is_alive());self.assertEqual(calls,[1]);self.assertTrue(j.done.is_set())
 def test_during_open_no_kickoff_and_close(self):
  Job,_=job_class();j=Job('fake',0,'test',[1,2]); e=Encoder()
  def op(delta):j.encoder=e;j.cancelled=True;j.cancel_event.set();return {"delta_from": 5}
  j._open=op;j.run();self.assertEqual(e.sent,0);self.assertGreaterEqual(e.closed,1)
 def test_normal_open_still_sends(self):
  Job,_=job_class();j=Job('fake',0,'test',[1,2]);e=Encoder()
  def op(delta):j.encoder=e;return {}
  j._open=op;j.run();self.assertEqual(e.sent,1);self.assertEqual(e.closed,0);self.assertIsNone(j.error)
if __name__=='__main__':unittest.main()
