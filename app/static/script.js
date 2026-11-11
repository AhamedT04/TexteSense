const form = document.getElementById("analysis-form");
const textInput = document.getElementById("text-input");
const analyzeBtn = document.getElementById("analyze-btn");
const charCount = document.getElementById("char-count");
const formMessage = document.getElementById("form-message");
const resultCard = document.getElementById("result-card");
const historyList = document.getElementById("history-list");
const historyPagination = document.getElementById("history-pagination");
const historyPrev = document.getElementById("history-prev");
const historyNext = document.getElementById("history-next");
const historyPageInfo = document.getElementById("history-page-info");

const statsFields = {
    total: document.getElementById("stat-total"),
    positive: document.getElementById("stat-positive"),
    negative: document.getElementById("stat-negative"),
    neutral: document.getElementById("stat-neutral"),
    with_feedback: document.getElementById("stat-feedback"),
    corrected: document.getElementById("stat-corrected"),
};

const SENTIMENTS = ["positive", "neutral", "negative"];
const SENTIMENT_LABEL = {
    positive: "Positif",
    negative: "Négatif",
    neutral: "Neutre",
};

const PAGE_SIZE = 10;
let currentPage = 0;
let lastResult = null; // dernière analyse affichée dans la carte « Résultat »

// ---------------------------------------------------------------------------
// Compteur de caractères
// ---------------------------------------------------------------------------

textInput.addEventListener("input", () => {
    const len = textInput.value.length;
    charCount.textContent = `${len.toLocaleString("fr-FR")} / 1 000`;
    charCount.classList.toggle("near-limit", len > 900);
});

// ---------------------------------------------------------------------------
// Soumission du formulaire
// ---------------------------------------------------------------------------

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = textInput.value.trim();

    if (text.length < 5 || text.length > 1000) {
        setMessage("Le texte doit contenir entre 5 et 1 000 caractères.", true);
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyse…";
    setMessage("", false);

    try {
        const data = await requestJson("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        renderResult(data);
        setMessage("Analyse enregistrée.", false);
        textInput.value = "";
        charCount.textContent = "0 / 1 000";
        currentPage = 0;
        await Promise.all([loadHistory(), loadStats()]);
    } catch (error) {
        setMessage(error.message, true);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyser";
    }
});

// ---------------------------------------------------------------------------
// Historique paginé
// ---------------------------------------------------------------------------

async function loadHistory() {
    const skip = currentPage * PAGE_SIZE;
    const data = await requestJson(`/history?skip=${skip}&limit=${PAGE_SIZE}`);
    renderHistory(data);
    updatePagination(data.length);
}

function renderHistory(items) {
    if (items.length === 0 && currentPage === 0) {
        historyList.innerHTML = '<p class="empty-history">Aucune analyse pour le moment.</p>';
        historyPagination.hidden = true;
        return;
    }

    historyList.innerHTML = items
        .map(
            (item) => `
            <article class="history-item">
                <div class="history-top">
                    <span class="badge ${item.sentiment}">${SENTIMENT_LABEL[item.sentiment] ?? item.sentiment}</span>
                    ${item.low_confidence ? lowConfidenceBadge() : ""}
                    <span class="history-meta">${formatDate(item.created_at)}</span>
                </div>
                <p class="history-text">${escapeHtml(item.text)}</p>
                <div class="keyword-list">
                    ${item.keywords.map((k) => `<span class="keyword-chip">${escapeHtml(k)}</span>`).join("")}
                </div>
                ${feedbackMarkup(item)}
                <div class="history-bottom">
                    <span class="history-meta">Confiance : ${(item.confidence * 100).toFixed(1)} %</span>
                    <button class="history-delete" data-id="${item.id}" aria-label="Supprimer l'analyse">Supprimer</button>
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

function updatePagination(fetchedCount) {
    const hasPrev = currentPage > 0;
    const hasNext = fetchedCount === PAGE_SIZE;

    historyPagination.hidden = !hasPrev && !hasNext;
    historyPrev.disabled = !hasPrev;
    historyNext.disabled = !hasNext;
    historyPageInfo.textContent = `Page ${currentPage + 1}`;
}

historyPrev.addEventListener("click", async () => {
    if (currentPage > 0) {
        currentPage--;
        await loadHistory();
    }
});

historyNext.addEventListener("click", async () => {
    currentPage++;
    await loadHistory();
});

// ---------------------------------------------------------------------------
// Statistiques
// ---------------------------------------------------------------------------

async function loadStats() {
    const data = await requestJson("/stats");
    Object.entries(statsFields).forEach(([key, element]) => {
        if (element) element.textContent = (data[key] ?? 0).toLocaleString("fr-FR");
    });
}

// ---------------------------------------------------------------------------
// Suppression
// ---------------------------------------------------------------------------

async function deleteAnalysis(id) {
    const data = await requestJson(`/history/${id}`, { method: "DELETE" });
    setMessage(data.message, false);
    if (lastResult && String(lastResult.id) === String(id)) {
        lastResult = null;
        resultCard.classList.add("empty");
        resultCard.innerHTML = "<p>Soumettez un texte pour voir l'analyse.</p>";
    }
    if (currentPage > 0) currentPage = 0;
    await Promise.all([loadHistory(), loadStats()]);
}

// ---------------------------------------------------------------------------
// Retour utilisateur (confirmer ou corriger une prédiction)
// ---------------------------------------------------------------------------

function feedbackMarkup(item) {
    if (item.user_label) {
        const verdict = item.user_label === item.sentiment ? "confirmé" : "corrigé";
        return `
            <div class="feedback" data-id="${item.id}">
                <span class="feedback-text">Votre avis (${verdict}) :</span>
                <span class="badge small ${item.user_label}">${SENTIMENT_LABEL[item.user_label]}</span>
                <button type="button" class="feedback-link" data-action="clear">Modifier</button>
            </div>`;
    }
    const alternatives = SENTIMENTS.filter((s) => s !== item.sentiment)
        .map(
            (s) =>
                `<button type="button" class="feedback-btn ${s}" data-action="set" data-label="${s}">${SENTIMENT_LABEL[s]}</button>`
        )
        .join("");
    return `
        <div class="feedback" data-id="${item.id}">
            <span class="feedback-text">Ce résultat est-il juste ?</span>
            <button type="button" class="feedback-btn" data-action="set" data-label="${item.sentiment}">Oui</button>
            <span class="feedback-text">Sinon, c'est plutôt :</span>
            ${alternatives}
        </div>`;
}

async function handleFeedbackClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const container = button.closest(".feedback");
    const id = container.dataset.id;

    container.querySelectorAll("button").forEach((b) => (b.disabled = true));
    try {
        const updated =
            button.dataset.action === "clear"
                ? await requestJson(`/analysis/${id}/feedback`, { method: "DELETE" })
                : await requestJson(`/analysis/${id}/feedback`, {
                      method: "PUT",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ label: button.dataset.label }),
                  });
        if (lastResult && lastResult.id === updated.id) renderResult(updated);
        setMessage(updated.user_label ? "Merci, votre avis est enregistré." : "Avis retiré.", false);
        await Promise.all([loadHistory(), loadStats()]);
    } catch (error) {
        setMessage(error.message, true);
        container.querySelectorAll("button").forEach((b) => (b.disabled = false));
    }
}

resultCard.addEventListener("click", handleFeedbackClick);
historyList.addEventListener("click", handleFeedbackClick);

// ---------------------------------------------------------------------------
// Rendu du résultat
// ---------------------------------------------------------------------------

function renderResult(item) {
    lastResult = item;
    const keywordsMarkup = item.keywords.length
        ? item.keywords.map((k) => `<span class="keyword-chip">${escapeHtml(k)}</span>`).join("")
        : '<span class="history-meta">Aucun mot-clé pertinent détecté.</span>';

    resultCard.classList.remove("empty");
    resultCard.innerHTML = `
        <div>
            <span class="badge ${item.sentiment}">${SENTIMENT_LABEL[item.sentiment] ?? item.sentiment}</span>
            ${item.low_confidence ? lowConfidenceBadge() : ""}
        </div>
        <div>
            <h3>Confiance</h3>
            <p>${(item.confidence * 100).toFixed(1)} %</p>
        </div>
        <div>
            <h3>Mots-clés</h3>
            <div class="keyword-list">${keywordsMarkup}</div>
        </div>
        <div>
            <h3>Texte analysé</h3>
            <p class="history-text">${escapeHtml(item.text)}</p>
        </div>
        ${feedbackMarkup(item)}
        ${item.model_name ? `<p class="history-meta model-name">Modèle : ${escapeHtml(shortModelName(item.model_name))}</p>` : ""}
    `;
}

// ---------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    let data = null;
    try {
        data = await response.json();
    } catch {
        // réponse sans corps JSON
    }
    if (!response.ok) {
        const detail = data && data.detail;
        throw new Error(typeof detail === "string" ? detail : "La requête a échoué. Réessayez.");
    }
    return data;
}

function lowConfidenceBadge() {
    return '<span class="badge low-confidence" title="Confiance faible — résultat indicatif">Confiance faible</span>';
}

function shortModelName(name) {
    // Un chemin local est réduit à son dernier dossier.
    const parts = name.split(/[\\/]/).filter(Boolean);
    return name.includes(":") || name.startsWith("/") ? parts[parts.length - 1] : name;
}

function setMessage(message, isError) {
    formMessage.textContent = message;
    formMessage.style.color = isError ? "var(--negative)" : "var(--muted)";
}

function formatDate(value) {
    return new Intl.DateTimeFormat("fr-FR", {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(new Date(value));
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

Promise.all([loadHistory(), loadStats()]).catch((error) => {
    setMessage(error.message, true);
});
