public class KDNode {
    public VectorItem item;
    public KDNode left = null;
    public KDNode right = null;

    public KDNode(VectorItem v) {
        this.item = v;
    }
}
