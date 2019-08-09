""" Tools for working with async queues and tasks.

These are mostly failed experiments, too much complexity. Futures based
techniques compose better and are only slightly more expensive in terms of
overheads. I'm keeping these for now, but probably they will be deleted.
"""
import asyncio
import queue
import logging
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from odc.ppt import EOS_MARKER


log = logging.getLogger(__name__)


async def async_q2q_map(func, q_in, q_out,
                        eos_marker=EOS_MARKER,
                        eos_passthrough=True,
                        **kwargs):
    """Like `map` but operating on values from/to queues.

       Roughly equivalent to:

       > while not end of stream:
       >    q_out.put(func(q_in.get(), **kwargs))

       Processing stops when `eos_marker` object is observed on input, by
       default `eos_marker` is passed through to output queue, but you can
       disable that.

       Calls `task_done()` method on input queue after result was copied to output queue.

       Assumption is that mapping function doesn't raise exceptions, instead it
       should return some sort of error object. If calling `func` does result
       in an exception it will be caught and logged but otherwise ignored.

       It is safe to have multiple consumers/producers reading/writing from the
       queues, although you might want to disable eos pass-through in those
       cases.

       func : Callable

       q_in: Input asyncio.Queue
       q_out: Output asyncio.Queue

       eos_marker: Value that indicates end of stream

       eos_passthrough: If True copy eos_marker to output queue before
                        terminating, if False then don't

    """
    while True:
        x = await q_in.get()

        if x is eos_marker:
            if eos_passthrough:
                await q_out.put(x)
            q_in.task_done()
            return

        err, result = (None, None)
        try:
            result = await func(x, **kwargs)
        except Exception as e:
            err = str(e)
            log.error("Uncaught exception: %s", err)

        if err is None:
            await q_out.put(result)

        q_in.task_done()


async def gen2q_async(func,
                      q_out,
                      nconcurrent,
                      eos_marker=EOS_MARKER,
                      eos_passthrough=True,
                      loop=None):
    """ Run upto `nconcurrent` generator functions, pump values from generator function into `q_out`
        To indicate that no more data is available func should return special value `eos_marker`

          [func(0)] \
          [func(1)]  >--> q_out
          [func(2)] /

        - func is expected not to raise exceptions
    """

    async def worker(idx):
        n = 0
        while True:
            try:
                x = await func(idx)
            except Exception as e:
                log.error("Uncaught exception: %s", str(e))
                return n

            if x is eos_marker:
                return n
            n += 1
            await q_out.put(x)
        return n

    ff = [asyncio.ensure_future(worker(i), loop=loop)
          for i in range(nconcurrent)]

    n_total = 0
    for f in ff:
        n_total += (await f)

    if eos_passthrough:
        await q_out.put(eos_marker)

    return n_total


async def aq2sq_pump(src, dst,
                     eos_marker=EOS_MARKER,
                     eos_passthrough=True,
                     dt=0.01):
    """ Pump from async Queue to synchronous queue.

        dt -- how much to sleep when dst is full
    """

    def safe_put(x, dst):
        try:
            dst.put_nowait(x)
        except queue.Full:
            return False

        return True

    async def push_to_dst(x, dst, dt):
        while not safe_put(x, dst):
            await asyncio.sleep(dt)

    while True:
        x = await src.get()

        if x is eos_marker:
            if eos_passthrough:
                await push_to_dst(x, dst, dt)

            src.task_done()
            break

        await push_to_dst(x, dst, dt)
        src.task_done()


async def q2q_nmap(func,
                   q_in,
                   q_out,
                   nconcurrent,
                   eos_marker=EOS_MARKER,
                   eos_passthrough=True,
                   dt=0.01,
                   loop=None):
    """Pump data from synchronous queue to another synchronous queue via a worker
       pool of async `func`s. Allow upto `nconcurrent` concurrent `func` tasks
       at a time.

                / [func] \
         q_in ->  [func]  >--> q_out
                \ [func] /


        - Order is not preserved.
        - func is expected not to raise exceptions
    """
    def safe_get(src):
        try:
            x = src.get_nowait()
            return (x, True)
        except queue.Empty:
            return (None, False)

    def safe_put(x, dst):
        try:
            dst.put_nowait(x)
        except queue.Full:
            return False

        return True

    async def push_to_dst(x, dst, dt):
        while not safe_put(x, dst):
            await asyncio.sleep(dt)

    async def intake_loop(src, dst, dt):
        while True:
            x, ok = safe_get(src)
            if not ok:
                await asyncio.sleep(dt)
            elif x is eos_marker:
                src.task_done()
                break
            else:
                await dst.put(x)
                src.task_done()

        for _ in range(nconcurrent):
            await dst.put(eos_marker)

        await dst.join()

    async def output_loop(src, dst, dt):
        while True:
            x = await src.get()

            if x is eos_marker:
                src.task_done()
                break

            await push_to_dst(x, dst, dt)
            src.task_done()

    aq_in = asyncio.Queue(nconcurrent*2)
    aq_out = asyncio.Queue(aq_in.maxsize)

    #                 / [func] \
    # q_in -> aq_in ->  [func]  >--> aq_out -> q_out
    #                 \ [func] /

    # Launch async worker pool: aq_in ->[func]-> aq_out
    for _ in range(nconcurrent):
        asyncio.ensure_future(async_q2q_map(func, aq_in, aq_out,
                                            eos_marker=eos_marker,
                                            eos_passthrough=False),
                              loop=loop)

    # Pump from aq_out -> q_out (async to sync interface)
    asyncio.ensure_future(output_loop(aq_out, q_out, dt), loop=loop)

    # Pump from q_in -> aq_in (sync to async interface)
    await intake_loop(q_in, aq_in, dt)

    # by this time all input items have been mapped through func and are in aq_out

    # terminate output pump
    await aq_out.put(eos_marker)  # tell output_loop to stop
    await aq_out.join()           # wait for ack, all valid data is in `q_out` now

    # finally push through eos_marker unless asked not too
    if eos_passthrough:
        await push_to_dst(eos_marker, q_out, dt)


################################################################################
# tests below
################################################################################


def test_q2q_map():
    async def proc(x):
        await asyncio.sleep(0.01)
        return (x, x)

    loop = asyncio.new_event_loop()

    def run(**kwargs):
        q1 = asyncio.Queue(10)
        q2 = asyncio.Queue(10)

        for i in range(4):
            q1.put_nowait(i)
        q1.put_nowait(EOS_MARKER)

        async def run_test(**kwargs):
            await async_q2q_map(proc, q1, q2, **kwargs)
            await q1.join()

            xx = []
            while not q2.empty():
                xx.append(q2.get_nowait())
            return xx

        return loop.run_until_complete(run_test(**kwargs))

    expect = [(i, i) for i in range(4)]
    assert run() == expect + [EOS_MARKER]
    assert run(eos_passthrough=False) == expect

    loop.close()


def test_q2qnmap():
    import random

    async def proc(x, state, delay=0.1):
        state.active += 1

        delay = random.uniform(0, delay)
        await asyncio.sleep(delay)

        state.max_active = max(state.active, state.max_active)
        state.active -= 1
        return (x, x)

    def run_producer(n, q, eos_marker):
        for i in range(n):
            q.put(i)
        q.put(eos_marker)
        q.join()

    def run_consumer(q, eos_marker):
        xx = []
        while True:
            x = q.get()
            q.task_done()
            xx.append(x)
            if x is eos_marker:
                break

        return xx

    wk_pool = ThreadPoolExecutor(max_workers=2)
    src = queue.Queue(3)
    dst = queue.Queue(3)

    # first do self test of consumer/producer
    N = 100

    wk_pool.submit(run_producer, N, src, EOS_MARKER)
    xx = wk_pool.submit(run_consumer, src, EOS_MARKER)
    xx = xx.result()

    assert len(xx) == N + 1
    assert len(set(xx) - set(range(N)) - set([EOS_MARKER])) == 0
    assert src.qsize() == 0

    loop = asyncio.new_event_loop()

    def run(N, nconcurrent, delay, eos_passthrough=True):
        async def run_test(func, N, nconcurrent):
            wk_pool.submit(run_producer, N, src, EOS_MARKER)
            xx = wk_pool.submit(run_consumer, dst, EOS_MARKER)
            await q2q_nmap(func, src, dst, nconcurrent, eos_passthrough=eos_passthrough)

            if eos_passthrough is False:
                dst.put(EOS_MARKER)

            return xx.result()

        state = SimpleNamespace(active=0, max_active=0)
        func = lambda x: proc(x, delay=delay, state=state)
        return state, loop.run_until_complete(run_test(func, N, nconcurrent))

    expect = set([(x, x) for x in range(N)] + [EOS_MARKER])

    st, xx = run(N, 20, 0.1)
    assert len(xx) == N + 1
    assert 1 < st.max_active <= 20
    assert set(xx) == expect

    st, xx = run(N, 4, 0.01)
    assert len(xx) == N + 1
    assert 1 < st.max_active <= 4
    assert set(xx) == expect

    st, xx = run(N, 4, 0.01, eos_passthrough=False)
    assert len(xx) == N + 1
    assert 1 < st.max_active <= 4
    assert set(xx) == expect


def test_gen2q():

    async def gen_func(idx, state):
        if state.count >= state.max_count:
            return EOS_MARKER

        cc = state.count
        state.count += 1

        await asyncio.sleep(state.dt)
        return cc

    async def sink(q):
        xx = []
        while True:
            x = await q.get()
            if x is EOS_MARKER:
                return xx
            xx.append(x)
        return xx

    async def run_async(nconcurrent, max_count=100, dt=0.1):
        state = SimpleNamespace(count=0,
                                max_count=max_count,
                                dt=dt)
        gen = lambda idx: gen_func(idx, state)

        q = asyncio.Queue(maxsize=10)
        g2q = asyncio.ensure_future(gen2q_async(gen, q, nconcurrent))
        xx = await sink(q)
        return g2q.result(), xx

    loop = asyncio.new_event_loop()

    def run(*args, **kwargs):
        return loop.run_until_complete(run_async(*args, **kwargs))

    n, xx = run(10, max_count=100, dt=0.1)
    assert len(xx) == n
    assert len(xx) == 100
    assert set(xx) == set(range(100))
