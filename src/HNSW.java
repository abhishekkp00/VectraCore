import java.util.*;

public class HNSW {
    private final Map<Integer, HNSWNode> G = new HashMap<>();
    private final int M;
    private final int M0;
    private final int ef_build;
    private final double mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.ef_build = efBuild;
        this.mL = 1.0 / Math.log(m);
    }

    private int randLevel() {
        double u = rng.nextDouble();
        if (u == 0.0) u = 1e-9;
        return (int) Math.floor(-Math.log(u) * mL);
    }

    private List<Pair> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
        Set<Integer> vis = new HashSet<>();
        PriorityQueue<Pair> cands = new PriorityQueue<>(Comparator.comparingDouble(p -> p.dist));
        PriorityQueue<Pair> found = new PriorityQueue<>((a, b) -> Float.compare(b.dist, a.dist));

        HNSWNode epNode = G.get(ep);
        float d0 = dist.apply(q, epNode.item.emb);
        vis.add(ep);
        cands.offer(new Pair(d0, ep));
        found.offer(new Pair(d0, ep));

        while (!cands.isEmpty()) {
            Pair curr = cands.poll();
            float cd = curr.dist;
            int cid = curr.id;

            if (found.size() >= ef && cd > found.peek().dist) {
                break;
            }

            HNSWNode cidNode = G.get(cid);
            if (cidNode == null || lyr >= cidNode.nbrs.size()) continue;

            for (int nid : cidNode.nbrs.get(lyr)) {
                if (vis.contains(nid) || !G.containsKey(nid)) continue;
                vis.add(nid);

                HNSWNode nidNode = G.get(nid);
                if (nidNode == null) continue;

                float nd = dist.apply(q, nidNode.item.emb);
                if (found.size() < ef || nd < found.peek().dist) {
                    Pair p = new Pair(nd, nid);
                    cands.offer(p);
                    found.offer(p);
                    if (found.size() > ef) {
                        found.poll();
                    }
                }
            }
        }

        List<Pair> res = new ArrayList<>();
        while (!found.isEmpty()) {
            res.add(found.poll());
        }
        res.sort(Comparator.comparingDouble(p -> p.dist));
        return res;
    }

    private List<Integer> selectNbrs(List<Pair> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        int limit = Math.min(cands.size(), maxM);
        for (int i = 0; i < limit; i++) {
            r.add(cands.get(i).id);
        }
        return r;
    }

    public synchronized void insert(VectorItem item, DistFn dist) {
        int id = item.id;
        int lvl = randLevel();
        G.put(id, new HNSWNode(item, lvl));

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            HNSWNode epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<Pair> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<Pair> W = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);

            HNSWNode currNode = G.get(id);
            currNode.nbrs.set(lc, new ArrayList<>(sel));

            for (int nid : sel) {
                HNSWNode nidNode = G.get(nid);
                if (nidNode == null) continue;
                while (nidNode.nbrs.size() <= lc) {
                    nidNode.nbrs.add(new ArrayList<>());
                }
                List<Integer> conn = nidNode.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<Pair> ds = new ArrayList<>();
                    for (int c : conn) {
                        HNSWNode cNode = G.get(c);
                        if (cNode != null) {
                            ds.add(new Pair(dist.apply(nidNode.item.emb, cNode.item.emb), c));
                        }
                    }
                    ds.sort(Comparator.comparingDouble(p -> p.dist));
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) {
                        conn.add(ds.get(i).id);
                    }
                }
            }
            if (!W.isEmpty()) ep = W.get(0).id;
        }

        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public synchronized List<Pair> knn(float[] q, int k, int ef, DistFn dist) {
        if (entryPt == -1) return new ArrayList<>();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            HNSWNode epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<Pair> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
        }
        List<Pair> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        if (W.size() > k) {
            return new ArrayList<>(W.subList(0, k));
        }
        return W;
    }

    public synchronized void remove(int id) {
        if (!G.containsKey(id)) return;
        for (HNSWNode nd : G.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.remove((Integer) id);
            }
        }
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : G.keySet()) {
                if (nid != id) {
                    entryPt = nid;
                    break;
                }
            }
        }
        G.remove(id);
    }

    public synchronized GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = G.size();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) {
            gi.nodesPerLayer.add(0);
            gi.edgesPerLayer.add(0);
        }
        for (Map.Entry<Integer, HNSWNode> entry : G.entrySet()) {
            int id = entry.getKey();
            HNSWNode nd = entry.getValue();
            gi.nodes.add(new NV(id, nd.item.metadata, nd.item.category, nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            gi.edges.add(new EV(id, nid, lc));
                        }
                    }
                }
            }
        }
        return gi;
    }

    public synchronized int size() {
        return G.size();
    }
}
