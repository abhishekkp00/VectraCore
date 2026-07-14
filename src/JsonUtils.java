import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;

public class JsonUtils {
    public static String jS(String s) {
        if (s == null) return "\"\"";
        StringBuilder o = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> o.append("\\\"");
                case '\\' -> o.append("\\\\");
                case '\n' -> o.append("\\n");
                case '\r' -> o.append("\\r");
                case '\t' -> o.append("\\t");
                default -> o.append(c);
            }
        }
        return o.append('"').toString();
    }

    public static String jVec(float[] v) {
        if (v == null) return "[]";
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) ss.append(",");
            ss.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return ss.append("]").toString();
    }

    public static float[] parseVec(String s) {
        if (s == null || s.trim().isEmpty()) return new float[0];
        String[] parts = s.split(",");
        List<Float> list = new ArrayList<>();
        for (String p : parts) {
            try {
                list.add(Float.parseFloat(p.trim()));
            } catch (NumberFormatException ignored) {}
        }
        float[] res = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public static String extractStr(String json, String key) {
        String search = "\"" + key + "\"";
        int p = json.indexOf(search);
        if (p == -1) return "";
        p = json.indexOf(":", p) + 1;
        while (p < json.length() && (json.charAt(p) == ' ' || json.charAt(p) == '\t')) {
            p++;
        }
        if (p >= json.length() || json.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < json.length()) {
            char c = json.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < json.length()) {
                p++;
                char next = json.charAt(p);
                switch (next) {
                    case '"' -> result.append('"');
                    case '\\' -> result.append('\\');
                    case 'n' -> result.append('\n');
                    case 'r' -> result.append('\r');
                    case 't' -> result.append('\t');
                    default -> result.append(next);
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    public static int extractInt(String json, String key, int defValue) {
        String search = "\"" + key + "\"";
        int p = json.indexOf(search);
        if (p == -1) return defValue;
        p = json.indexOf(":", p) + 1;
        while (p < json.length() && (json.charAt(p) == ' ' || json.charAt(p) == '\t')) {
            p++;
        }
        int end = p;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-' || json.charAt(end) == '+')) {
            end++;
        }
        try {
            return Integer.parseInt(json.substring(p, end).trim());
        } catch (Exception e) {
            return defValue;
        }
    }

    public static Map<String, String> parseQueryParams(String query) {
        Map<String, String> params = new HashMap<>();
        if (query == null || query.isEmpty()) return params;
        for (String param : query.split("&")) {
            String[] pair = param.split("=", 2);
            if (pair.length > 0) {
                String key = URLDecoder.decode(pair[0], StandardCharsets.UTF_8);
                String value = pair.length > 1 ? URLDecoder.decode(pair[1], StandardCharsets.UTF_8) : "";
                params.put(key, value);
            }
        }
        return params;
    }

    public static float[] parseEmbeddingArray(String json) {
        int start = json.indexOf("\"embedding\"");
        if (start == -1) return new float[0];
        int bracketStart = json.indexOf("[", start);
        if (bracketStart == -1) return new float[0];
        int bracketEnd = json.indexOf("]", bracketStart);
        if (bracketEnd == -1) return new float[0];
        String arrayContent = json.substring(bracketStart + 1, bracketEnd);
        if (arrayContent.trim().isEmpty()) return new float[0];
        String[] parts = arrayContent.split(",");
        float[] res = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            res[i] = Float.parseFloat(parts[i].trim());
        }
        return res;
    }
}
