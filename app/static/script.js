const form = document.getElementById("analysis-form");
const textInput = document.getElementById("text-input");
const analyzeBtn = document.getElementById("analyze-btn");
const charCount = document.getElementById("char-count");
const formMessage = document.getElementById("form-message");
const resultCard = document.getElementById("result-card");
const historyList = document.getElementById("history-list");

const statsFields = {
    total: document.getElementById("stat-total"),
    positive: document.getElementById("stat-positive"),
    negative: document.getElementById("stat-negative"),
    neutral: document.getElementById("stat-neutral"),
};

const sentimentLabel = {
    positive: "positif",
    negative: "negatif",
    neutral: "neutre",
};

textInput.addEventListener("input", () => {
    charCount.textContent = `${textInput.value.length} / 1000`;
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = textInput.value.trim();

    if (text.length < 5 || text.length > 1000) {
        setMessage("Le texte doit contenir entre 5 et 1000 caracteres.", true);
        return;
    }

    analyzeBtn.disabled = true;
    setMessage("Analyse en cours...", false);

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Erreur pendant l'analyse.");
        }

        renderResult(data);
        setMessage("Analyse enregistree.", false);
        textInput.value = "";
        charCount.textContent = "0 / 1000";
        await Promise.all([loadHistory(), loadStats()]);
    } catch (error) {
        setMessage(error.message, true);
    } finally {
        analyzeBtn.disabled = false;
    }
});

async function loadHistory() {
    const response = await fetch("/history");
    const data = await response.json();
    if (!response.ok) {
        throw new Error("Impossible de charger l'historique.");
    }

    if (data.length === 0) {
        historyList.innerHTML = '<p class="empty-history">Aucune analyse pour le moment.</p>';
        return;
    }

    historyList.innerHTML = data
        .map(
            (item) => `
                <article class="history-item">
                    <div class="history-top">
                        <span class="badge ${item.sentiment}">${sentimentLabel[item.sentiment] || item.sentiment}</span>
                        <span class="history-meta">${formatDate(item.created_at)}</span>
                    </div>
                    <p class="history-text">${escapeHtml(item.text)}</p>
                    <div class="keyword-list">
                        ${item.keywords.map((keyword) => `<span class="keyword-chip">${escapeHtml(keyword)}</span>`).join("")}
                    </div>
                    <div class="history-bottom">
                        <span class="history-meta">Confiance: ${(item.confidence * 100).toFixed(1)}%</span>
                        <button class="history-delete" data-id="${item.id}">Supprimer</button>
                    </div>
                </article>
            `
        )
        .join("");

    historyList.querySelectorAll(".history-delete").forEach((button) => {
        button.addEventListener("click", async () => {
            try {
                await deleteAnalysis(button.dataset.id);
            } catch (error) {
                setMessage(error.message, true);
            }
        });
    });
}

async function loadStats() {
    const response = await fetch("/stats");
    const data = await response.json();
    if (!response.ok) {
        throw new Error("Impossible de charger les statistiques.");
    }

    Object.entries(statsFields).forEach(([key, element]) => {
        element.textContent = data[key];
    });
}

async function deleteAnalysis(id) {
    const response = await fetch(`/history/${id}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail || "Suppression impossible.");
    }
    setMessage(data.message, false);
    await Promise.all([loadHistory(), loadStats()]);
}

function renderResult(item) {
    const keywordsMarkup = item.keywords.length
        ? item.keywords.map((keyword) => `<span class="keyword-chip">${escapeHtml(keyword)}</span>`).join("")
        : "<span class=\"history-meta\">Aucun mot-cle pertinent detecte.</span>";

    resultCard.classList.remove("empty");
    resultCard.innerHTML = `
        <div>
            <span class="badge ${item.sentiment}">${sentimentLabel[item.sentiment] || item.sentiment}</span>
        </div>
        <div>
            <h3>Confiance</h3>
            <p>${(item.confidence * 100).toFixed(1)}%</p>
        </div>
        <div>
            <h3>Mots-cles</h3>
            <div class="keyword-list">${keywordsMarkup}</div>
        </div>
        <div>
            <h3>Texte analyse</h3>
            <p class="history-text">${escapeHtml(item.text)}</p>
        </div>
    `;
}

function setMessage(message, isError) {
    formMessage.textContent = message;
    formMessage.style.color = isError ? "var(--negative)" : "var(--muted)";
}

function formatDate(value) {
    return new Intl.DateTimeFormat("fr-CA", {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(new Date(value));
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

Promise.all([loadHistory(), loadStats()]).catch((error) => {
    setMessage(error.message, true);
});
