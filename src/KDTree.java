import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class KDTree {
    private KDNode root = null;
    private final int dims;

    public KDTree(int d) {
        this.dims = d;
    }

    private KDNode ins(KDNode n, VectorItem v, int d) {
        if (n == null) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n.item.emb[ax]) {
            n.left = ins(n.left, v, d + 1);
        } else {
            n.right = ins(n.right, v, d + 1);
        }
        return n;
    }

    public synchronized void insert(VectorItem v) {
        root = ins(root, v, 0);
    }

    private void knn(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<Pair> heap) {
        if (n == null) return;
        float dn = dist.apply(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek().dist) {
            heap.offer(new Pair(dn, n.item.id));
            if (heap.size() > k) {
                heap.poll();
            }
        }
        int ax = d % dims;
        float diff = q[ax] - n.item.emb[ax];
        KDNode closer = diff < 0 ? n.left : n.right;
        KDNode farther = diff < 0 ? n.right : n.left;
        knn(closer, q, k, d + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist) {
            knn(farther, q, k, d + 1, dist, heap);
        }
    }

    public synchronized List<Pair> knn(float[] q, int k, DistFn dist) {
        PriorityQueue<Pair> heap = new PriorityQueue<>((a, b) -> Float.compare(b.dist, a.dist));
        knn(root, q, k, 0, dist, heap);
        List<Pair> r = new ArrayList<>();
        while (!heap.isEmpty()) {
            r.add(heap.poll());
        }
        r.sort(Comparator.comparingDouble(p -> p.dist));
        return r;
    }

    public synchronized void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) {
            insert(v);
        }
    }
}
