"""Causal window and SSE worker lifecycle regression checks."""
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from inference_service import AnalyzeRequest, Predictor, Session, trailing_window


class WindowTests(unittest.TestCase):
    def test_tail_padding_never_reads_future(self):
        frames = np.repeat(np.arange(100)[:,None,None],17,axis=1)
        frames = np.repeat(frames,3,axis=2).astype(np.float32)
        padded, info = trailing_window(frames,2)
        np.testing.assert_array_equal(padded[:3],frames[:3])
        np.testing.assert_array_equal(padded[3:],np.repeat(frames[2:3],61,axis=0))
        self.assertEqual(info,dict(start_frame=0,end_frame=2,observed_frames=3,padding_frames=61))
        first, info=trailing_window(frames,0)
        self.assertEqual(info['padding_frames'],63)
        np.testing.assert_array_equal(first,np.repeat(frames[:1],64,axis=0))
        later, info=trailing_window(frames,90)
        np.testing.assert_array_equal(later,frames[27:91])
        self.assertEqual(info['padding_frames'],0)
        last, info=trailing_window(frames,999)
        np.testing.assert_array_equal(last,frames[36:100])
        self.assertEqual(info['end_frame'],99)

    def test_model_path_boundary(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); (root/'best.pt').write_bytes(b'fixture')
            predictor=Predictor(root,None,None,None,None)
            self.assertEqual(predictor.model_path('best.pt'),root/'best.pt')
            for name in ('../best.pt','best.npz','missing.pt'):
                with self.assertRaises(ValueError):predictor.model_path(name)

    def test_request_validation(self):
        for invalid in ({'time':float('nan')},{'threshold':1.1},{'mode':'other'}):
            with self.assertRaises(ValueError):AnalyzeRequest(model='best.pt',source='dataset',path='test/clip',**invalid)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_coalesces_and_releases_on_cancel(self):
        class FakePredictor:
            def load(self,params):return {}
            def predict(self,loaded,params,seconds,sequence):return {'sequence':sequence,'time':seconds}
        session=Session(FakePredictor(),AnalyzeRequest(model='best.pt',source='dataset',path='test/clip',mode='window'))
        self.addAsyncCleanup(self.close,session)
        while True:
            event,data=await asyncio.wait_for(session.queue.get(),2)
            if event=='result':break
        self.assertEqual(data['sequence'],0)
        session.time=2;session.sequence=2;session.version+=1;session.wake.set()
        session.time=3;session.sequence=3;session.version+=1;session.wake.set()
        event,data=await asyncio.wait_for(session.queue.get(),2)
        self.assertEqual(event,'result');self.assertEqual(data,{'sequence':3,'time':3,'generation':0})

    async def close(self,session):
        session.close();await asyncio.gather(session.task,return_exceptions=True)
        self.assertTrue(session.task.done())

    async def test_slow_result_is_published_but_prior_seek_is_discarded(self):
        import threading
        class SlowPredictor:
            def __init__(self):self.entered=threading.Event();self.release=threading.Event()
            def load(self,params):return {}
            def predict(self,loaded,params,seconds,sequence):
                if sequence==0:
                    self.entered.set();self.release.wait(2)
                return {'sequence':sequence,'time':seconds}
        for new_seek in (False,True):
            predictor=SlowPredictor()
            session=Session(predictor,AnalyzeRequest(model='best.pt',source='dataset',path='test/clip',mode='window'))
            self.addAsyncCleanup(self.close,session)
            for _ in range(100):
                if predictor.entered.is_set():break
                await asyncio.sleep(.01)
            self.assertTrue(predictor.entered.is_set())
            session.sequence=1;session.time=1;session.version+=1
            if new_seek:session.generation=1
            session.wake.set();predictor.release.set()
            results=[]
            while not results or results[-1]['sequence']!=1:
                event,data=await asyncio.wait_for(session.queue.get(),3)
                if event=='result':results.append(data)
            self.assertEqual([r['sequence'] for r in results],[1] if new_seek else [0,1])
            session.close()

    async def test_error_is_sse_event(self):
        class InvalidPredictor:
            def load(self,params):raise ValueError('bad checkpoint')
        session=Session(InvalidPredictor(),AnalyzeRequest(model='best.pt',source='dataset',path='test/clip'))
        self.addAsyncCleanup(self.close,session)
        while True:
            event,data=await asyncio.wait_for(session.queue.get(),2)
            if event=='analysis_error':break
        self.assertEqual(data['message'],'bad checkpoint')


if __name__ == '__main__':unittest.main()
