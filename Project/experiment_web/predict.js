"use strict";

const byId = (id) => document.getElementById(id);
const models = { XGBClassifier: "XGBoost", CatBoostClassifier: "CatBoost", RandomForestClassifier: "Random Forest" };
const units = {
    heart_rate: "bpm", body_temperature: "°C", ambient_temperature: "°C",
    training_duration: "min", hydration_level: "%", humidity: "%",
};
const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
const requestedRun = Number(new URLSearchParams(location.search).get("run"));
let options = [];
let loaded = null;
let inputVersion = 0;

function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
}

function label(name) {
    const text = name.replaceAll("_", " ");
    return text[0].toUpperCase() + text.slice(1);
}

async function request(path, body) {
    const response = await fetch(path, body === undefined ? undefined : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The server could not complete the request.");
    return data;
}

function showOptions(preferred) {
    const selected = byId("model-run");
    const choices = options.filter((run) => run.classes.length === Number(byId("target").value));
    choices.sort((first, second) => Number(second.evaluation.endsWith("_test"))
        - Number(first.evaluation.endsWith("_test")) || second.combined_score - first.combined_score
        || second.id - first.id);
    selected.replaceChildren(...choices.map((run) => {
        const [, , features, variant, calibration] = run.name.split("/");
        const text = [
            `#${run.id}`, models[run.model] ?? run.model, label(features ?? run.name),
            label(variant ?? "baseline"), calibration, run.evaluation.endsWith("_test") ? "Test" : "CV",
        ].filter(Boolean).join(" · ");
        return new Option(text, run.id);
    }));
    if (choices.some((run) => run.id === preferred)) selected.value = String(preferred);
    byId("load-model").disabled = !choices.length;
    byId("model-status").textContent = choices.length
        ? "Loading fits the saved settings and calibration on the development data."
        : "No saved models for these risk categories.";
}

function resetPrediction() {
    inputVersion += 1;
    byId("prediction-status").textContent = "Enter session data and select Predict.";
    byId("prediction-result").replaceChildren();
}

function renderFields() {
    byId("feature-inputs").replaceChildren(...loaded.fields.map((field) => {
        const name = label(field.name);
        const unit = units[field.name];
        const wrapper = element("label", unit ? `${name} (${unit})` : name);
        const input = element("input");
        input.type = "number";
        input.step = "any";
        input.name = field.name;
        input.placeholder = number.format(field.median);
        const hint = element("small", `Development range ${number.format(field.minimum)}–${number.format(field.maximum)}`);
        wrapper.append(input, hint);
        return wrapper;
    }));
}

function renderDecision(model, result) {
    const panel = element("div");
    panel.className = "decision-card";
    const binary = model.classes.length === 2;
    const banner = element("strong", binary
        ? result.prediction === model.classes[1] ? "High Risk flag triggered" : "High Risk flag not triggered"
        : result.prediction);
    banner.className = "prediction-class";
    panel.append(banner);

    if (binary) {
        const risk = result.probabilities[model.classes[1]];
        const threshold = model.cutoffs[0];
        const triggered = result.prediction === model.classes[1];
        panel.append(
            element("p", `Estimated High Risk probability: ${percent.format(risk)}`),
            element("p", `High Risk flag threshold: ${percent.format(threshold)}`),
            element("p", triggered
                ? "The estimated probability reached the threshold, so the model raised a flag."
                : "The estimated probability stayed below the threshold, so the model did not raise a flag."),
        );
    } else {
        const score = model.classes.reduce((total, name, index) =>
            total + index * result.probabilities[name], 0);
        const [low, high] = model.cutoffs;
        const rules = element("ul");
        const range = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
        for (const rule of [
            `Healthy below ${range.format(low)}`,
            `Low Risk from ${range.format(low)} up to ${range.format(high)}`,
            `Injured at ${range.format(high)} or higher`,
        ]) rules.append(element("li", rule));
        panel.append(element("p", `Risk score: ${range.format(score)} out of 2. The score weights the three probabilities by 0, 1, and 2.`), rules);
    }
    return panel;
}

async function loadSelectedModel() {
    const button = byId("load-model");
    const target = byId("target");
    const run = byId("model-run");
    loaded = null;
    byId("prediction-workspace").hidden = true;
    resetPrediction();
    if (!run.value) return;
    button.disabled = target.disabled = run.disabled = true;
    byId("model-status").textContent = `Loading run #${run.value}…`;
    try {
        loaded = await request("/api/predict/load", { run_id: Number(run.value) });
        renderFields();
        resetPrediction();
        byId("prediction-workspace").hidden = false;
        byId("model-status").textContent = `Loaded #${loaded.id} · ${models[loaded.model] ?? loaded.model} · ${loaded.classes.join(" / ")}`;
    } catch (error) {
        byId("model-status").textContent = error.message;
    } finally {
        button.disabled = target.disabled = run.disabled = false;
    }
}

byId("target").addEventListener("change", () => {
    showOptions();
    return loadSelectedModel();
});
byId("model-run").addEventListener("change", loadSelectedModel);
byId("model-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadSelectedModel();
});

byId("use-example").addEventListener("click", () => {
    for (const field of loaded.fields) {
        byId("prediction-form").elements.namedItem(field.name).value = field.median;
    }
    resetPrediction();
});
byId("prediction-form").addEventListener("reset", resetPrediction);
byId("prediction-form").addEventListener("input", resetPrediction);
byId("prediction-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = byId("predict-button");
    const status = byId("prediction-status");
    const model = loaded;
    const version = inputVersion;
    const features = Object.fromEntries(loaded.fields.flatMap(({ name }) => {
        const input = form.elements.namedItem(name);
        return input.value === "" ? [] : [[name, input.valueAsNumber]];
    }));
    byId("prediction-result").replaceChildren();
    button.disabled = true;
    status.textContent = "Calculating…";
    try {
        const result = await request("/api/predict", { run_id: model.id, features });
        if (model !== loaded || version !== inputVersion) return;
        const decision = renderDecision(model, result);
        const breakdown = element("details");
        breakdown.className = "probability-details";
        breakdown.open = model.classes.length === 3;
        breakdown.append(element("summary", "All class probabilities"));
        const list = element("div");
        list.className = "probability-list";
        for (const [name, probability] of Object.entries(result.probabilities)) {
            const row = element("div");
            row.className = "probability-row";
            const bar = element("progress");
            bar.max = 1;
            bar.value = probability;
            bar.setAttribute("aria-label", `${name} probability`);
            row.append(element("span", name), element("strong", percent.format(probability)), bar);
            list.append(row);
        }
        breakdown.append(list);
        status.textContent = `Model run #${result.run_id}`;
        const notes = result.imputed_features.length
            ? element("p", `Training medians used for ${result.imputed_features.map(label).join(", ")}.`) : null;
        byId("prediction-result").replaceChildren(decision, breakdown, ...(notes ? [notes] : []));
    } catch (error) {
        if (model === loaded && version === inputVersion) status.textContent = error.message;
    } finally {
        button.disabled = false;
    }
});

async function initialize() {
    try {
        options = await request("/api/predict/models");
        const preferred = options.find((run) => run.id === requestedRun);
        if (preferred) byId("target").value = String(preferred.classes.length);
        showOptions(preferred?.id);
    } catch (error) {
        byId("model-status").textContent = `Unable to load saved runs. ${error.message}`;
    }
}

initialize();
