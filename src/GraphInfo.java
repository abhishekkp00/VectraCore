import java.util.ArrayList;
import java.util.List;

public class GraphInfo {
    public int topLayer;
    public int nodeCount;
    public List<Integer> nodesPerLayer = new ArrayList<>();
    public List<Integer> edgesPerLayer = new ArrayList<>();
    public List<NV> nodes = new ArrayList<>();
    public List<EV> edges = new ArrayList<>();
}
