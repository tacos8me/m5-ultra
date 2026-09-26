import numpy as np
from omlx.patches.deepseek_v41 import hot_rows as m
from test_deepseek_v41_engram_io import table

def test_cache():
 c=m.HotRows(32768)
 rows=np.arange(80,dtype=np.int64); raw=np.arange(80*32,dtype=np.uint16).reshape(80,32)
 c.remember(rows,np.ones(80,dtype=np.int64),[(raw,'BF16')])
 hit,data=c.lookup(rows);assert hit.all();np.testing.assert_array_equal(data[0][0],raw)
 assert c.nbytes<=c.budget
 assert not c.lookup(np.array([-1, -2], dtype=np.int64))[0].any()
 # Every colliding ID must miss; frequency admission replaces only its own slot.
 other=rows+len(c.tags)
 hit,_=c.lookup(other);assert not hit.any()
 new=raw+77
 c.remember(other,np.full(80,9),[(new,'BF16')])
 hit,data=c.lookup(other);assert hit.sum()>40;np.testing.assert_array_equal(data[0][0],new[hit])
 saved=new[hit].copy()
 old_hit,old_data=c.lookup(rows);assert not (old_hit & hit).any()
 np.testing.assert_array_equal(old_data[0][0],raw[old_hit])
 # Copies returned by lookup survive slot replacement and close.
 c.clear();np.testing.assert_array_equal(data[0][0],saved);assert c.nbytes==0

def test_mixed_frequency_and_budget():
 c=m.HotRows(4096)
 rng=np.random.default_rng(3)
 for _ in range(300):
  rows,counts=np.unique(rng.integers(0,1000,size=90),return_counts=True)
  data=[(np.repeat(rows[:,None],4,axis=1).astype(np.int32),'U32')]
  c.remember(rows,counts,data)
  probe=np.arange(1000);hit,got=c.lookup(probe)
  np.testing.assert_array_equal(got[0][0],np.repeat(probe[hit,None],4,axis=1))
  assert c.nbytes<=4096
 # Concurrent caller copies cannot alias slots.
 c.remember(np.array([0]),np.array([1000000]),[(np.zeros((1,4),np.int32),'U32')])
 assert c.freq.max()<=65535



def test_storage_concurrent_hits_and_misses(table, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from omlx.patches.deepseek_v41 import storage
    monkeypatch.setattr(storage, "HOT_ROWS_BYTES", 1 << 20)
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    def check(seed):
        ids = np.random.default_rng(seed).integers(0, 5000, 1000)
        result = embed.gather(ids, warm_rows=bool(seed % 2))
        for (actual, dtype), (expected, expected_dtype) in zip(result, embed._read_rows(ids)):
            assert dtype == expected_dtype
            np.testing.assert_array_equal(actual, expected)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(check, range(12)))
        assert 0 < embed._hot_rows.nbytes <= 1 << 20
        assert embed._hot_rows.hits > 0
        assert "_hot_rows" not in embed and not embed.children()
    finally:
        embed.close()
    assert embed._hot_rows.nbytes == 0
