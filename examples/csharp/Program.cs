using System.Net.Http.Json;
using System.Text.Json;

var baseUrl = Environment.GetEnvironmentVariable("CLASSIFIER_URL") ?? "http://localhost:8000";
var apiKey = Environment.GetEnvironmentVariable("API_KEY")
    ?? throw new InvalidOperationException("Set API_KEY before running this example.");
using var http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromMinutes(5) };
http.DefaultRequestHeaders.Add("X-API-Key", apiKey);
var client = new ClassifierClient(http);

Guid? conversationId = null;
int version = 0;
Console.WriteLine("請輸入問題：");
while (Console.ReadLine() is { } message && !string.IsNullOrWhiteSpace(message))
{
    // Keep this SAME request object (and ID) when retrying a timeout/network error.
    var request = new ClassificationRequest(Guid.NewGuid(), conversationId, version, message);
    var response = await client.ClassifyAsync(request);
    conversationId = response.ConversationId;
    version = response.Version;
    if (response.IsFinal)
    {
        Console.WriteLine(response.Domain == "UNKNOWN"
            ? "無法確定系統，請交由人工分流。"
            : $"轉交：{response.Domain}");
        break;
    }
    Console.WriteLine(response.Clarification?.Question);
}

public sealed class ClassifierClient(HttpClient http)
{
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true
    };

    public async Task<ClassificationResponse> ClassifyAsync(
        ClassificationRequest request, CancellationToken cancellationToken = default)
    {
        using var response = await http.PostAsJsonAsync("/api/v1/classify", request, Json, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            // 429/502/503/504 may be retried using the SAME request after backoff.
            // 409 requires reconciliation; 404/410 require a new conversation.
            // Never convert a technical error into a successful UNKNOWN result.
            throw new HttpRequestException(body, null, response.StatusCode);
        }
        return await response.Content.ReadFromJsonAsync<ClassificationResponse>(Json, cancellationToken)
            ?? throw new JsonException("Empty classifier response");
    }
}

public sealed record ClassificationRequest(Guid RequestId, Guid? ConversationId, int ExpectedVersion, string Message);
public sealed record Clarification(string Id, string Question);
public sealed record ClassificationResponse(
    Guid RequestId, Guid ConversationId, int Version, string Status, string? Domain,
    int ClarificationCount, Clarification? Clarification, string ReasonCode, bool IsFinal, string ConfigVersion);
