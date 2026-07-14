import java.util.*;

public class VectorDB {
    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private int nextId = 1;
    public final int dims;

    public VectorDB(int d) {
        this.dims = d;
        this.kdt = new KDTree(d);
        this.hnsw = new HNSW(16, 200);
    }

    public synchronized int insert(String meta, String cat, float[] emb, DistFn dist) {
        VectorItem v = new VectorItem(nextId++, meta, cat, emb);
        store.put(v.id, v);
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);
        return v.id;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        List<VectorItem> rem = new ArrayList<>(store.values());
        kdt.rebuild(rem);
        return true;
    }

    public static class Hit {
        public int id;
        public String meta;
        public String cat;
        public float[] emb;
        public float dist;

        public Hit(int id, String meta, String cat, float[] emb, float dist) {
            this.id = id;
            this.meta = meta;
            this.cat = cat;
            this.emb = emb;
            this.dist = dist;
        }
    }

    public static class SearchOut {
        public List<Hit> hits = new ArrayList<>();
        public long us;
        public String algo;
        public String metric;
    }

    public synchronized SearchOut search(float[] q, int k, String metric, String algo) {
        DistFn dfn = DistanceMetrics.getDistFn(metric);
        long t0 = System.nanoTime();

        List<Pair> raw;
        if ("bruteforce".equals(algo)) {
            raw = bf.knn(q, k, dfn);
        } else if ("kdtree".equals(algo)) {
            raw = kdt.knn(q, k, dfn);
        } else {
            raw = hnsw.knn(q, k, 50, dfn);
        }

        long us = (System.nanoTime() - t0) / 1000;

        SearchOut out = new SearchOut();
        out.us = us;
        out.algo = algo;
        out.metric = metric;
        for (Pair p : raw) {
            VectorItem item = store.get(p.id);
            if (item != null) {
                out.hits.add(new Hit(item.id, item.metadata, item.category, item.emb, p.dist));
            }
        }
        return out;
    }

    public static class BenchOut {
        public long bfUs, kdUs, hnswUs;
        public int n;

        public BenchOut(long bfUs, long kdUs, long hnswUs, int n) {
            this.bfUs = bfUs;
            this.kdUs = kdUs;
            this.hnswUs = hnswUs;
            this.n = n;
        }
    }

    public synchronized BenchOut benchmark(float[] q, int k, String metric) {
        DistFn dfn = DistanceMetrics.getDistFn(metric);

        long tBf = System.nanoTime();
        bf.knn(q, k, dfn);
        long bfUs = (System.nanoTime() - tBf) / 1000;

        long tKd = System.nanoTime();
        kdt.knn(q, k, dfn);
        long kdUs = (System.nanoTime() - tKd) / 1000;

        long tHnsw = System.nanoTime();
        hnsw.knn(q, k, 50, dfn);
        long hnswUs = (System.nanoTime() - tHnsw) / 1000;

        return new BenchOut(bfUs, kdUs, hnswUs, store.size());
    }

    public synchronized List<VectorItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized GraphInfo hnswInfo() {
        return hnsw.getInfo();
    }

    public synchronized int size() {
        return store.size();
    }
}
