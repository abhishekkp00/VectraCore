import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class OllamaClient {
    private final String baseUrl;
    private final HttpClient client;

    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public OllamaClient(String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();
    }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                detectModels(resp.body());
                return true;
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    private void detectModels(String body) {
        if (body == null) return;

        // Detect embed model
        if (body.contains("nomic-embed-text")) {
            embedModel = "nomic-embed-text";
        } else if (body.contains("all-minilm")) {
            embedModel = "all-minilm";
        }

        // Detect generation model
        if (body.contains("llama3.2")) {
            genModel = "llama3.2";
        } else if (body.contains("qwen2.5:3b")) {
            genModel = "qwen2.5:3b";
        } else if (body.contains("qwen2.5-coder")) {
            genModel = "qwen2.5-coder";
        } else if (body.contains("gemma4")) {
            genModel = "gemma4";
        } else if (body.contains("tinyllama")) {
            genModel = "tinyllama";
        }
    }

    public float[] embed(String text) {
        try {
            String jsonBody = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + escapeJson(text) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/embeddings"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .timeout(Duration.ofSeconds(30))
                    .build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return new float[0];
            return JsonUtils.parseEmbeddingArray(resp.body());
        } catch (Exception e) {
            return new float[0];
        }
    }

    public String generate(String prompt) {
        try {
            String jsonBody = "{\"model\":\"" + genModel + "\",\"prompt\":\"" + escapeJson(prompt) + "\",\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .timeout(Duration.ofSeconds(180))
                    .build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
            return JsonUtils.extractStr(resp.body(), "response");
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        return sb.toString();
    }
}
