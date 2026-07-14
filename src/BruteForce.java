import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class BruteForce {
    public final List<VectorItem> items = new ArrayList<>();

    public synchronized void insert(VectorItem v) {
        items.add(v);
    }

    public synchronized List<Pair> knn(float[] q, int k, DistFn dist) {
        List<Pair> r = new ArrayList<>(items.size());
        for (VectorItem v : items) {
            r.add(new Pair(dist.apply(q, v.emb), v.id));
        }
        r.sort(Comparator.comparingDouble(p -> p.dist));
        if (r.size() > k) {
            return new ArrayList<>(r.subList(0, k));
        }
        return r;
    }

    public synchronized void remove(int id) {
        items.removeIf(v -> v.id == id);
    }
}
