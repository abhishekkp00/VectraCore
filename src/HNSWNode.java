import java.util.ArrayList;
import java.util.List;

public class HNSWNode {
    public VectorItem item;
    public int maxLyr;
    public List<List<Integer>> nbrs;

    public HNSWNode(VectorItem item, int maxLyr) {
        this.item = item;
        this.maxLyr = maxLyr;
        this.nbrs = new ArrayList<>(maxLyr + 1);
        for (int i = 0; i <= maxLyr; i++) {
            this.nbrs.add(new ArrayList<>());
        }
    }
}
