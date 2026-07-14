import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.Executors;

public class Main {
    public static final int DIMS = 16;

    private static VectorDB db;
    private static DocumentDB docDB;
    private static OllamaClient ollama;

    public static void main(String[] args) {
        db = new VectorDB(DIMS);
        docDB = new DocumentDB();
        String envHost = System.getenv("OLLAMA_HOST");
        String envPort = System.getenv("OLLAMA_PORT");
        String host = (envHost != null && !envHost.isEmpty()) ? envHost : "127.0.0.1";
        int portVal = 11434;
        if (envPort != null && !envPort.isEmpty()) {
            try {
                portVal = Integer.parseInt(envPort);
            } catch (NumberFormatException ignored) {}
        }
        ollama = new OllamaClient(host, portVal);

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectraCore Engine (Java) ===");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        try {
            int port = 8080;
            HttpServer server = null;
            while (port < 8090) {
                try {
                    server = HttpServer.create(new InetSocketAddress(port), 0);
                    break;
                } catch (IOException e) {
                    port++;
                }
            }
            if (server == null) {
                throw new IOException("No free ports found in range 8080-8089");
            }

            System.out.println("Server running at: http://localhost:" + port);

            // Register routes matching C++ server endpoints
            server.createContext("/", new StaticHandler());
            server.createContext("/search", new SearchHandler());
            server.createContext("/insert", new InsertHandler());
            server.createContext("/delete/", new DeleteHandler());
            server.createContext("/items", new ItemsHandler());
            server.createContext("/benchmark", new BenchmarkHandler());
            server.createContext("/hnsw-info", new HnswInfoHandler());
            server.createContext("/doc/insert", new DocInsertHandler());
            server.createContext("/doc/delete/", new DocDeleteHandler());
            server.createContext("/doc/list", new DocListHandler());
            server.createContext("/doc/search", new DocSearchHandler());
            server.createContext("/doc/ask", new DocAskHandler());
            server.createContext("/status", new StatusHandler());
            server.createContext("/stats", new StatsHandler());

            server.setExecutor(Executors.newFixedThreadPool(8));
            server.start();

        } catch (IOException e) {
            System.err.println("Failed to start HTTP server: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void loadDemo(VectorDB db) {
        DistFn dist = DistanceMetrics.getDistFn("cosine");
        db.insert("Linked List: nodes connected by pointers", "cs",
                new float[]{0.90f, 0.85f, 0.72f, 0.68f, 0.12f, 0.08f, 0.15f, 0.10f, 0.05f, 0.08f, 0.06f, 0.09f, 0.07f, 0.11f, 0.08f, 0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
                new float[]{0.88f, 0.82f, 0.78f, 0.74f, 0.15f, 0.10f, 0.08f, 0.12f, 0.06f, 0.07f, 0.08f, 0.05f, 0.09f, 0.06f, 0.07f, 0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
                new float[]{0.82f, 0.76f, 0.88f, 0.80f, 0.20f, 0.18f, 0.12f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.08f, 0.09f, 0.06f, 0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
                new float[]{0.85f, 0.80f, 0.75f, 0.82f, 0.18f, 0.14f, 0.10f, 0.08f, 0.06f, 0.09f, 0.07f, 0.06f, 0.10f, 0.08f, 0.09f, 0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
                new float[]{0.87f, 0.78f, 0.70f, 0.76f, 0.13f, 0.11f, 0.09f, 0.14f, 0.08f, 0.07f, 0.06f, 0.08f, 0.07f, 0.10f, 0.08f, 0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
                new float[]{0.12f, 0.15f, 0.18f, 0.10f, 0.91f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.07f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
                new float[]{0.20f, 0.18f, 0.15f, 0.12f, 0.88f, 0.90f, 0.82f, 0.76f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f, 0.07f, 0.08f, 0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
                new float[]{0.15f, 0.12f, 0.20f, 0.18f, 0.84f, 0.80f, 0.88f, 0.82f, 0.07f, 0.08f, 0.06f, 0.10f, 0.09f, 0.06f, 0.09f, 0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
                new float[]{0.22f, 0.16f, 0.14f, 0.20f, 0.80f, 0.85f, 0.76f, 0.90f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.10f, 0.07f, 0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
                new float[]{0.18f, 0.20f, 0.16f, 0.14f, 0.86f, 0.78f, 0.84f, 0.80f, 0.06f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.10f, 0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                new float[]{0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.90f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.09f, 0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
                new float[]{0.06f, 0.08f, 0.07f, 0.09f, 0.09f, 0.06f, 0.08f, 0.07f, 0.86f, 0.90f, 0.82f, 0.76f, 0.07f, 0.09f, 0.06f, 0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                new float[]{0.09f, 0.07f, 0.06f, 0.08f, 0.08f, 0.09f, 0.07f, 0.06f, 0.82f, 0.78f, 0.90f, 0.84f, 0.09f, 0.07f, 0.08f, 0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                new float[]{0.07f, 0.09f, 0.08f, 0.06f, 0.06f, 0.07f, 0.09f, 0.08f, 0.78f, 0.82f, 0.86f, 0.90f, 0.06f, 0.08f, 0.07f, 0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
                new float[]{0.06f, 0.07f, 0.10f, 0.09f, 0.10f, 0.06f, 0.07f, 0.10f, 0.85f, 0.80f, 0.76f, 0.82f, 0.09f, 0.07f, 0.10f, 0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
                new float[]{0.09f, 0.07f, 0.08f, 0.10f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.09f, 0.06f, 0.91f, 0.85f, 0.78f, 0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
                new float[]{0.07f, 0.09f, 0.06f, 0.08f, 0.09f, 0.07f, 0.10f, 0.08f, 0.07f, 0.09f, 0.08f, 0.07f, 0.87f, 0.89f, 0.82f, 0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                new float[]{0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.09f, 0.06f, 0.07f, 0.08f, 0.83f, 0.80f, 0.88f, 0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
                new float[]{0.25f, 0.20f, 0.22f, 0.18f, 0.22f, 0.18f, 0.20f, 0.15f, 0.06f, 0.08f, 0.07f, 0.09f, 0.80f, 0.84f, 0.78f, 0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                new float[]{0.06f, 0.08f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.07f, 0.10f, 0.08f, 0.06f, 0.07f, 0.85f, 0.82f, 0.86f, 0.80f}, dist);
    }

    private static void setCORS(HttpExchange exchange) {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void sendResponse(HttpExchange exchange, int status, String body, String contentType) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        sendResponse(exchange, status, bytes, contentType);
    }

    private static void sendResponse(HttpExchange exchange, int status, byte[] bytes, String contentType) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    static class StaticHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            File file = new File("web/index.html");
            if (!file.exists()) {
                file = new File("index.html");
            }
            if (!file.exists()) {
                sendResponse(exchange, 404, "File not found", "text/plain");
                return;
            }
            byte[] bytes = Files.readAllBytes(file.toPath());
            sendResponse(exchange, 200, bytes, "text/html");
        }
    }

    static class SearchHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            Map<String, String> queryParams = JsonUtils.parseQueryParams(exchange.getRequestURI().getQuery());
            float[] q = JsonUtils.parseVec(queryParams.get("v"));
            if (q.length != DIMS) {
                sendResponse(exchange, 400, "{\"error\":\"need " + DIMS + "D vector\"}", "application/json");
                return;
            }

            int k = 5;
            try {
                if (queryParams.containsKey("k")) {
                    k = Integer.parseInt(queryParams.get("k"));
                }
            } catch (NumberFormatException ignored) {}

            String metric = queryParams.getOrDefault("metric", "cosine");
            if (metric.isEmpty()) metric = "cosine";
            String algo = queryParams.getOrDefault("algo", "hnsw");
            if (algo.isEmpty()) algo = "hnsw";

            VectorDB.SearchOut out = db.search(q, k, metric, algo);

            StringBuilder ss = new StringBuilder();
            ss.append("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0) ss.append(",");
                VectorDB.Hit h = out.hits.get(i);
                ss.append("{\"id\":").append(h.id)
                  .append(",\"metadata\":").append(JsonUtils.jS(h.meta))
                  .append(",\"category\":").append(JsonUtils.jS(h.cat))
                  .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                  .append(",\"embedding\":").append(JsonUtils.jVec(h.emb)).append("}");
            }
            ss.append("],\"latencyUs\":").append(out.us)
              .append(",\"algo\":").append(JsonUtils.jS(out.algo))
              .append(",\"metric\":").append(JsonUtils.jS(out.metric)).append("}");

            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class InsertHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String meta = JsonUtils.extractStr(body, "metadata");
            String cat = JsonUtils.extractStr(body, "category");
            float[] emb = new float[0];

            int p = body.indexOf("\"embedding\"");
            if (p != -1) {
                int start = body.indexOf("[", p);
                int end = body.indexOf("]", start);
                if (start != -1 && end != -1) {
                    emb = JsonUtils.parseVec(body.substring(start + 1, end));
                }
            }

            if (meta.isEmpty() || emb.length != DIMS) {
                sendResponse(exchange, 400, "{\"error\":\"invalid body\"}", "application/json");
                return;
            }

            int id = db.insert(meta, cat, emb, DistanceMetrics.getDistFn("cosine"));
            sendResponse(exchange, 200, "{\"id\":" + id + "}", "application/json");
        }
    }

    static class DeleteHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String path = exchange.getRequestURI().getPath();
            String idStr = path.substring("/delete/".length());
            boolean ok = false;
            try {
                int id = Integer.parseInt(idStr);
                ok = db.remove(id);
            } catch (NumberFormatException ignored) {}

            sendResponse(exchange, 200, "{\"ok\":" + ok + "}", "application/json");
        }
    }

    static class ItemsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            List<VectorItem> items = db.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) ss.append(",");
                VectorItem v = items.get(i);
                ss.append("{\"id\":").append(v.id)
                  .append(",\"metadata\":").append(JsonUtils.jS(v.metadata))
                  .append(",\"category\":").append(JsonUtils.jS(v.category))
                  .append(",\"embedding\":").append(JsonUtils.jVec(v.emb)).append("}");
            }
            ss.append("]");
            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class BenchmarkHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            Map<String, String> queryParams = JsonUtils.parseQueryParams(exchange.getRequestURI().getQuery());
            float[] q = JsonUtils.parseVec(queryParams.get("v"));
            if (q.length != DIMS) {
                sendResponse(exchange, 400, "{\"error\":\"need " + DIMS + "D vector\"}", "application/json");
                return;
            }

            int k = 5;
            try {
                if (queryParams.containsKey("k")) {
                    k = Integer.parseInt(queryParams.get("k"));
                }
            } catch (NumberFormatException ignored) {}

            String metric = queryParams.getOrDefault("metric", "cosine");
            if (metric.isEmpty()) metric = "cosine";

            VectorDB.BenchOut b = db.benchmark(q, k, metric);
            String resp = "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                    + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}";
            sendResponse(exchange, 200, resp, "application/json");
        }
    }

    static class HnswInfoHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            GraphInfo gi = db.hnswInfo();
            StringBuilder ss = new StringBuilder();
            ss.append("{\"topLayer\":").append(gi.topLayer)
              .append(",\"nodeCount\":").append(gi.nodeCount)
              .append(",\"nodesPerLayer\":[");
            for (int i = 0; i < gi.nodesPerLayer.size(); i++) {
                if (i > 0) ss.append(",");
                ss.append(gi.nodesPerLayer.get(i));
            }
            ss.append("],\"edgesPerLayer\":[");
            for (int i = 0; i < gi.edgesPerLayer.size(); i++) {
                if (i > 0) ss.append(",");
                ss.append(gi.edgesPerLayer.get(i));
            }
            ss.append("],\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0) ss.append(",");
                NV n = gi.nodes.get(i);
                ss.append("{\"id\":").append(n.id)
                  .append(",\"metadata\":").append(JsonUtils.jS(n.metadata))
                  .append(",\"category\":").append(JsonUtils.jS(n.category))
                  .append(",\"maxLyr\":").append(n.maxLyr).append("}");
            }
            ss.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0) ss.append(",");
                EV e = gi.edges.get(i);
                ss.append("{\"src\":").append(e.src)
                  .append(",\"dst\":").append(e.dst)
                  .append(",\"lyr\":").append(e.lyr).append("}");
            }
            ss.append("]}");

            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class DocInsertHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String title = JsonUtils.extractStr(body, "title");
            String text = JsonUtils.extractStr(body, "text");

            if (title.isEmpty() || text.isEmpty()) {
                sendResponse(exchange, 400, "{\"error\":\"need title and text\"}", "application/json");
                return;
            }

            // Word-based chunking
            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();

            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    String err = "{\"error\":\"Ollama unavailable. "
                            + "Install from https://ollama.com then run: "
                            + "ollama pull nomic-embed-text && ollama pull llama3.2\"}";
                    sendResponse(exchange, 500, err, "application/json");
                    return;
                }
                String chunkTitle = (chunks.size() > 1)
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }

            StringBuilder ss = new StringBuilder();
            ss.append("{\"ids\":[");
            for (int i = 0; i < ids.size(); i++) {
                if (i > 0) ss.append(",");
                ss.append(ids.get(i));
            }
            ss.append("],\"chunks\":").append(chunks.size())
              .append(",\"dims\":").append(docDB.getDims()).append("}");

            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class DocDeleteHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String path = exchange.getRequestURI().getPath();
            String idStr = path.substring("/doc/delete/".length());
            boolean ok = false;
            try {
                int id = Integer.parseInt(idStr);
                ok = docDB.remove(id);
            } catch (NumberFormatException ignored) {}

            sendResponse(exchange, 200, "{\"ok\":" + ok + "}", "application/json");
        }
    }

    static class DocListHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            List<DocItem> docs = docDB.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0) ss.append(",");
                DocItem d = docs.get(i);
                String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                int wordsCount = d.text.trim().split("\\s+").length;
                ss.append("{\"id\":").append(d.id)
                  .append(",\"title\":").append(JsonUtils.jS(d.title))
                  .append(",\"preview\":").append(JsonUtils.jS(preview))
                  .append(",\"words\":").append(wordsCount).append("}");
            }
            ss.append("]");

            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class DocSearchHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String question = JsonUtils.extractStr(body, "question");
            int k = JsonUtils.extractInt(body, "k", 3);

            if (question.isEmpty()) {
                sendResponse(exchange, 400, "{\"error\":\"need question\"}", "application/json");
                return;
            }

            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                sendResponse(exchange, 500, "{\"error\":\"Ollama unavailable\"}", "application/json");
                return;
            }

            List<DocumentDB.DocHit> hits = docDB.search(qEmb, k, 0.7f);

            StringBuilder ss = new StringBuilder();
            ss.append("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(",");
                DocumentDB.DocHit hit = hits.get(i);
                ss.append("{\"id\":").append(hit.item.id)
                  .append(",\"title\":").append(JsonUtils.jS(hit.item.title))
                  .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hit.dist)).append("}");
            }
            ss.append("]}");

            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class DocAskHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String question = JsonUtils.extractStr(body, "question");
            int k = JsonUtils.extractInt(body, "k", 3);

            if (question.isEmpty()) {
                sendResponse(exchange, 400, "{\"error\":\"need question\"}", "application/json");
                return;
            }

            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                sendResponse(exchange, 500, "{\"error\":\"Ollama unavailable\"}", "application/json");
                return;
            }

            List<DocumentDB.DocHit> hits = docDB.search(qEmb, k, 0.7f);

            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                DocumentDB.DocHit hit = hits.get(i);
                ctx.append("[").append(i + 1).append("] ").append(hit.item.title).append(":\n")
                   .append(hit.item.text).append("\n\n");
            }

            String prompt = "You are a helpful assistant. Answer the user's question directly. "
                    + "Use the provided context if it contains relevant information. "
                    + "If it doesn't, just use your own general knowledge. "
                    + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                    + "Just answer the question naturally.\n\n"
                    + "Context:\n" + ctx.toString()
                    + "Question: " + question + "\n\n"
                    + "Answer:";

            String answer = ollama.generate(prompt);

            StringBuilder ss = new StringBuilder();
            ss.append("{\"answer\":").append(JsonUtils.jS(answer))
              .append(",\"model\":").append(JsonUtils.jS(ollama.genModel))
              .append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(",");
                DocumentDB.DocHit hit = hits.get(i);
                ss.append("{\"id\":").append(hit.item.id)
                  .append(",\"title\":").append(JsonUtils.jS(hit.item.title))
                  .append(",\"text\":").append(JsonUtils.jS(hit.item.text))
                  .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hit.dist)).append("}");
            }
            ss.append("],\"docCount\":").append(docDB.size()).append("}");

            sendResponse(exchange, 200, ss.toString(), "application/json");
        }
    }

    static class StatusHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            boolean up = ollama.isAvailable();
            String resp = "{\"ollamaAvailable\":" + (up ? "true" : "false")
                    + ",\"embedModel\":" + JsonUtils.jS(ollama.embedModel)
                    + ",\"genModel\":" + JsonUtils.jS(ollama.genModel)
                    + ",\"docCount\":" + docDB.size()
                    + ",\"docDims\":" + docDB.getDims()
                    + ",\"demoDims\":" + DIMS
                    + ",\"demoCount\":" + db.size() + "}";

            sendResponse(exchange, 200, resp, "application/json");
        }
    }

    static class StatsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCORS(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            String resp = "{\"count\":" + db.size()
                    + ",\"dims\":" + DIMS
                    + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                    + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";

            sendResponse(exchange, 200, resp, "application/json");
        }
    }

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        if (text == null || text.trim().isEmpty()) return new ArrayList<>();
        String[] words = text.split("\\s+");
        if (words.length == 0) return new ArrayList<>();
        if (words.length <= chunkWords) return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunk.append(" ");
                chunk.append(words[j]);
            }
            chunks.add(chunk.toString());
            if (end == words.length) break;
        }
        return chunks;
    }
}
