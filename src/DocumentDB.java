import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DocumentDB {
    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private int nextId = 1;
    private int dims = 0;

    public synchronized int insert(String title, String text, float[] emb) {
        if (dims == 0) dims = emb.length;
        DocItem item = new DocItem(nextId++, title, text, emb);
        store.put(item.id, item);
        VectorItem vi = new VectorItem(item.id, title, "doc", emb);
        hnsw.insert(vi, DistanceMetrics::cosine);
        bf.insert(vi);
        return item.id;
    }

    public static class DocHit {
        public float dist;
        public DocItem item;

        public DocHit(float dist, DocItem item) {
            this.dist = dist;
            this.item = item;
        }
    }

    public synchronized List<DocHit> search(float[] q, int k, float maxDist) {
        if (store.isEmpty()) return new ArrayList<>();
        List<Pair> raw = (store.size() < 10)
                ? bf.knn(q, k, DistanceMetrics::cosine)
                : hnsw.knn(q, k, 50, DistanceMetrics::cosine);
        List<DocHit> out = new ArrayList<>();
        for (Pair p : raw) {
            DocItem item = store.get(p.id);
            if (item != null && p.dist <= maxDist) {
                out.add(new DocHit(p.dist, item));
            }
        }
        return out;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        hnsw.remove(id);
        bf.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized int size() {
        return store.size();
    }

    public synchronized int getDims() {
        return dims;
    }
}
