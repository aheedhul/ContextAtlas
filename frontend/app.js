const state = {
    topK: 3,
    apiEndpoint: localStorage.getItem("contextatlas_api_endpoint") || "http://localhost:8000",
};

const elements = {
    queryForm: document.getElementById("queryForm"),
    submitButton: document.getElementById("submitButton"),
    clearButton: document.getElementById("clearButton"),
    queryInput: document.getElementById("queryInput"),
    topKSlider: document.getElementById("topKSlider"),
    topKValue: document.getElementById("topKValue"),
    chipTopK: document.getElementById("chipTopK"),
    answerContent: document.getElementById("answerContent"),
    answerCard: document.getElementById("answerCard"),
    contextsList: document.getElementById("contextsList"),
    responseTime: document.getElementById("responseTime"),
    connectionState: document.getElementById("connectionState"),
    toast: document.getElementById("toast"),
    evidenceCard: document.getElementById("evidenceCard"),
};

const contextTemplate = document.getElementById("contextTemplate");

function setConnectionState(status, message) {
    const colorMap = {
        ready: "var(--success)",
        pending: "var(--accent)",
        error: "var(--danger)",
    };

    elements.connectionState.querySelector("span:nth-child(2)").textContent = message;
    elements.connectionState.style.color = colorMap[status] || "var(--text-muted)";
}

function setLoading(isLoading) {
    elements.submitButton.disabled = isLoading;
    elements.submitButton.classList.toggle("loading", isLoading);
}

function showToast(message, tone = "info") {
    const toneMap = {
        info: "rgba(20, 34, 61, 0.95)",
        success: "rgba(86, 242, 195, 0.95)",
        warning: "rgba(247, 178, 103, 0.95)",
        danger: "rgba(255, 107, 107, 0.92)",
    };

    elements.toast.style.background = toneMap[tone] || toneMap.info;
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    setTimeout(() => elements.toast.classList.remove("show"), 4200);
}

function sanitize(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderAnswer(answer) {
    elements.answerContent.innerHTML = "";

    if (!answer) {
        const placeholder = document.createElement("p");
        placeholder.classList.add("placeholder");
        placeholder.textContent = "No answer available.";
        elements.answerContent.appendChild(placeholder);
        return;
    }

    const paragraphs = answer.split(/\n{2,}/);
    paragraphs.forEach((chunk) => {
        if (!chunk.trim()) return;
        const paragraph = document.createElement("p");
        const citationPattern = /(\(See Pages:[^)]+\))/i;
        const safeChunk = sanitize(chunk).replace(/\n/g, "<br>");
        paragraph.innerHTML = safeChunk.replace(citationPattern, '<span class="citation">$1</span>');
        elements.answerContent.appendChild(paragraph);
    });
}

function parseContext(raw, index) {
    const context = {
        label: `Snippet ${index + 1}`,
        page: "Page N/A",
        text: raw,
    };

    const pageMatch = raw.match(/\[Page\s([^\]]+)\]/i);
    if (pageMatch) {
        context.page = `Page ${pageMatch[1]}`;
    }

    const trimmed = raw.replace(/\[Context[^\]]*\]\s*/i, "").replace(/\[Page[^\]]*\]\s*/i, "").trim();
    context.text = trimmed;
    return context;
}

function renderContexts(contexts) {
    elements.contextsList.innerHTML = "";

    if (!Array.isArray(contexts) || contexts.length === 0) {
    const placeholder = document.createElement("div");
    placeholder.classList.add("placeholder", "warning");
    placeholder.setAttribute("role", "alert");
    placeholder.innerHTML = "<strong>Information not found</strong><p>Information not found in the supplied corpus.</p>";
        elements.contextsList.appendChild(placeholder);
        elements.evidenceCard.classList.add("warning");
        return;
    }

    elements.evidenceCard.classList.remove("warning");

    contexts.forEach((contextString, index) => {
        const parsed = parseContext(contextString, index);
        const fragment = contextTemplate.content.cloneNode(true);

        fragment.querySelector(".snippet-badge").textContent = parsed.label;
        fragment.querySelector(".snippet-page").textContent = parsed.page;
        fragment.querySelector(".snippet-text").textContent = parsed.text;

        elements.contextsList.appendChild(fragment);
    });
}

function updateTimestamp() {
    const now = new Date();
    const formatter = new Intl.DateTimeFormat([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
    elements.responseTime.textContent = `Generated at ${formatter.format(now)} (local time)`;
}

async function submitQuery(event) {
    event.preventDefault();
    const query = elements.queryInput.value.trim();

    if (!query) {
        showToast("Please enter a question first.", "warning");
        return;
    }

    const endpoint = state.apiEndpoint.replace(/\/$/, "");
    const url = `${endpoint}/query`;

    setLoading(true);
    setConnectionState("pending", "Retrieving contexts...");

    const requestStart = performance.now();

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query, top_k: state.topK }),
        });

    const latency = performance.now() - requestStart;

        if (!response.ok) {
            const errorBody = await response.text();
            let message = `API returned ${response.status}`;
            if (errorBody) {
                try {
                    const parsed = JSON.parse(errorBody);
                    message = parsed.detail || parsed.message || JSON.stringify(parsed);
                } catch {
                    message = errorBody;
                }
            }
            throw new Error(message);
        }

        const payload = await response.json();

        renderAnswer(payload.answer);
        renderContexts(payload.contexts);
        updateTimestamp();
        const latencyMs = Math.max(0, Math.round(latency));
        setConnectionState("ready", latencyMs ? `Ready · ${latencyMs} ms` : "Ready · FastAPI online");
        showToast("Response generated successfully.", "success");

        if (payload.answer && /I do not have enough information/i.test(payload.answer)) {
            elements.answerCard.classList.add("neutral");
        } else {
            elements.answerCard.classList.remove("neutral");
        }
    } catch (error) {
        console.error(error);
        const friendlyMessage = (error && error.message ? error.message : "Unknown error").trim();
        const displayMessage = friendlyMessage || "Unexpected failure contacting the API.";
        renderAnswer(`Request failed: ${displayMessage}`);
        renderContexts([]);
        const shortMessage = displayMessage.length > 48
            ? `${displayMessage.slice(0, 45)}…`
            : displayMessage;
        setConnectionState("error", shortMessage || "Connection issue");
        showToast(`Request failed: ${displayMessage}`, "danger");
    } finally {
        setLoading(false);
    }
}

function clearResults() {
    elements.queryInput.value = "";
    elements.answerContent.innerHTML = '<p class="placeholder">Submit a question to generate a grounded answer with citations.</p>';
    elements.contextsList.innerHTML = '<div class="placeholder warning" role="alert"><strong>No contexts yet.</strong><p>The supporting paragraphs will appear here after your first query.</p></div>';
    elements.responseTime.textContent = "Awaiting first query...";
    elements.answerCard.classList.remove("neutral");
    elements.evidenceCard.classList.remove("warning");
    setConnectionState("pending", "Awaiting first query");
}

function handleSliderChange(event) {
    const value = Number(event.target.value);
    state.topK = value;
    elements.topKValue.textContent = value;
    elements.chipTopK.textContent = value;
}

function init() {
    try {
        const params = new URLSearchParams(window.location.search);
        const apiParam = params.get("api");
        if (apiParam) {
            state.apiEndpoint = apiParam;
            localStorage.setItem("contextatlas_api_endpoint", apiParam);
        }
    } catch (error) {
        console.warn("Unable to parse api query parameter", error);
    }

    elements.topKValue.textContent = state.topK;
    elements.chipTopK.textContent = state.topK;
    setConnectionState("pending", "Awaiting first query");

    elements.queryForm.addEventListener("submit", submitQuery);
    elements.clearButton.addEventListener("click", clearResults);
    elements.topKSlider.addEventListener("input", handleSliderChange);

    // No default value; placeholder guides the user to type.
}

document.addEventListener("DOMContentLoaded", init);
