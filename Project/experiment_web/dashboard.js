"use strict";

const byId = (id) => document.getElementById(id);
const labels = {
    cv: "Notebook workflow CV", test: "Notebook workflow test",
    study_cv3: "3-fold screen", study_cv5: "5-fold finalists", study_test: "Session holdout",
    study_athlete_cv5: "Athlete-disjoint CV",
    binary_cv3: "3-fold screen", binary_cv5: "5-fold finalists", binary_test: "Session holdout",
    binary_athlete_cv5: "Athlete-disjoint CV",
    study_sweep_cv3: "Model sweep · 3-fold CV",
    binary_sweep_cv3: "XGBoost sweep · 3-fold CV",
};
const modelNames = {
    RandomForestClassifier: "Random Forest", CatBoostClassifier: "CatBoost", XGBClassifier: "XGBoost",
};
const scoreFields = {
    combined: "Combined", accuracy: "Accuracy", f1: "Macro F1",
    roc: "ROC AUC", pr: "PR AUC", brier: "Brier",
};
const filters = ["search", "target", "stage", "sort"];
const dateFormat = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const state = { runs: [], loaded: false, error: "" };

function element(tag, value, className) {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = value;
    if (className) node.className = className;
    return node;
}

function format(value, digits = 4) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function valueText(value) {
    return value === undefined ? "unrecorded" : JSON.stringify(value);
}

function normalize(run) {
    const metrics = run.metrics;
    const data = metrics.data ?? {};
    const classes = data.target_classes ?? Object.keys(metrics.confusion_matrix ?? {});
    const target = run.evaluated_on.startsWith("binary_") || classes.length === 2 ? "binary" : "multiclass";
    const probabilities = metrics.probability_metrics?.["Macro average"];
    const model = modelNames[run.model] ?? run.model;
    const variant = run.name.split("/")[3];
    const configuration = data.feature_set
        ? [data.feature_set, variant, Number.isFinite(data.calibration_fraction)
            ? `${Math.round(data.calibration_fraction * 100)}% calibration`
            : null].filter(Boolean).join(" · ").replaceAll("_", " ")
        : run.name;
    return {
        ...run, target, model, data, configuration,
        stage: labels[run.evaluated_on] ?? run.evaluated_on,
        scores: {
            combined: metrics.combined_score, accuracy: metrics.accuracy,
            f1: metrics.threshold_metrics?.["Macro average"]?.F1,
            roc: probabilities?.["ROC AUC"], pr: probabilities?.["PR AUC"], brier: metrics.calibrated_brier,
        },
    };
}

function updateSummary() {
    byId("run-count").textContent = state.runs.length;
    const models = new Set(state.runs.map((run) => run.model)).size;
    byId("model-count").textContent = `${models} ${models === 1 ? "model" : "models"}`;
    const cvRuns = state.runs.filter((run) =>
        /(?:^|_)cv\d*$/.test(run.evaluated_on) && Number.isFinite(run.scores.combined));
    for (const target of ["binary", "multiclass"]) {
        const best = cvRuns.filter((run) => run.target === target)
            .sort((a, b) => b.scores.combined - a.scores.combined || b.id - a.id)[0];
        byId(`${target}-best`).textContent = format(best?.scores.combined);
        byId(`${target}-label`).textContent = best
            ? `${best.model} / Run #${best.id} / ${best.stage}` : "No CV runs";
    }
}

function renderRuns() {
    const query = byId("search").value.trim().toLowerCase();
    const target = byId("target").value;
    const stage = byId("stage").value;
    const sort = byId("sort").value;
    const visible = state.runs.filter((run) => {
        const searchable = [run.id, `#${run.id}`, run.name, run.model, run.notes, ...(run.data.features ?? [])]
            .join(" ").toLowerCase();
        return (!target || target === run.target) && (!stage || stage === run.stage) && searchable.includes(query);
    });
    visible.sort((a, b) => {
        if (sort === "id") return b.id - a.id;
        const ascending = sort === "brier";
        const missing = ascending ? Infinity : -Infinity;
        const difference = (a.scores[sort] ?? missing) - (b.scores[sort] ?? missing);
        return (ascending ? difference : -difference) || b.id - a.id;
    });
    byId("status").textContent = state.error || (state.loaded
        ? `${visible.length} of ${state.runs.length} runs` : "Loading history…");
    const rows = visible.map((run) => {
        const row = element("tr");
        const id = element("td");
        const button = element("button", `#${run.id}`, "run-button");
        button.type = "button";
        button.setAttribute("aria-label", `Open run ${run.id}, ${run.model}`);
        button.addEventListener("click", () => openRun(run));
        id.append(button);
        const config = element("td", run.model);
        config.append(element("small", run.configuration));
        const stage = element("td", run.stage, "stage");
        stage.append(element("small", run.target === "binary" ? "Binary" : "Multiclass"));
        row.append(id, config, stage);
        for (const key of Object.keys(scoreFields)) {
            row.append(element("td", format(run.scores[key]), key === "combined" ? "score" : ""));
        }
        return row;
    });
    if (!rows.length && state.loaded) {
        const cell = element("td", state.runs.length ? "No runs match these filters." : "No saved runs yet.");
        cell.colSpan = 9;
        const row = element("tr");
        row.append(cell);
        rows.push(row);
    }
    byId("runs").replaceChildren(...rows);
}

function differenceList(run, baseline) {
    if (!baseline) return ["Choose a baseline to compare recorded settings."];
    const changes = [];
    if (run.model !== baseline.model) changes.push(`Model: ${baseline.model} → ${run.model}`);
    for (const key of ["feature_set", "feature_count", "calibration_fraction", "holdout_fraction", "cv_folds", "source", "imputation"]) {
        if (run.data[key] !== baseline.data[key]) {
            changes.push(`${key.replaceAll("_", " ")}: ${valueText(baseline.data[key])} → ${valueText(run.data[key])}`);
        }
    }
    const features = run.data.features;
    const previous = baseline.data.features;
    if (Array.isArray(features) && Array.isArray(previous)) {
        const added = features.filter((name) => !previous.includes(name));
        const removed = previous.filter((name) => !features.includes(name));
        if (added.length) changes.push(`Features added: ${added.join(", ")}`);
        if (removed.length) changes.push(`Features removed: ${removed.join(", ")}`);
    } else {
        changes.push("Feature names are unavailable for one or both runs.");
    }
    for (const key of [...new Set([...Object.keys(run.params), ...Object.keys(baseline.params)])].sort()) {
        if (valueText(run.params[key]) !== valueText(baseline.params[key])) {
            changes.push(`${key}: ${valueText(baseline.params[key])} → ${valueText(run.params[key])}`);
        }
    }
    return changes.length ? changes : ["No differences in the recorded settings."];
}

function section(title, ...content) {
    const node = element("section", undefined, "detail-section");
    node.append(element("h3", title), ...content);
    return node;
}

function scoreList(run) {
    const list = element("dl");
    for (const [key, label] of Object.entries(scoreFields)) {
        const entry = element("div");
        entry.append(element("dt", label), element("dd", format(run.scores[key])));
        list.append(entry);
    }
    return list;
}

function tableDetails(title, record, formatCell = (value) => Number.isInteger(value) ? value : format(value), rowLabel = "Class") {
    const node = element("details");
    node.append(element("summary", title));
    const rows = record instanceof Map ? [...record] : Object.entries(record);
    const columns = Object.keys(rows[0]?.[1] ?? {});
    const table = element("table");
    const head = table.createTHead().insertRow();
    for (const name of [rowLabel, ...columns]) {
        const heading = element("th", name);
        heading.scope = "col";
        head.append(heading);
    }
    const body = table.createTBody();
    for (const [name, values] of rows) {
        const row = body.insertRow();
        const heading = element("th", name);
        heading.scope = "row";
        row.append(heading);
        for (const key of columns) {
            const value = values[key];
            row.append(element("td", formatCell(value, key)));
        }
    }
    const scroll = element("div", undefined, "table-scroll");
    scroll.append(table);
    node.append(scroll);
    return node;
}

function featureImportance(run) {
    const scores = Object.entries(run.metrics.feature_importance ?? {});
    if (!scores.length) return element("p", "Feature importance was not saved for this run.");
    scores.sort(([, first], [, second]) => second - first);
    const table = tableDetails(
        "All features",
        new Map(scores.map(([name, value]) => [name, { Importance: value }])),
        (value) => format(value, 5),
        "Feature",
    );
    table.open = true;
    return table;
}

function rankFeatureSets(runs, stage) {
    const groups = new Map();
    const sweep = stage.endsWith("_sweep_cv3");
    // Runs arrive newest first, so renamed or repeated evaluations cannot add weight to a setting.
    for (const run of runs) {
        const set = run.data.feature_set ?? run.data.features?.join(", ");
        if (run.evaluated_on !== stage || !set || !Number.isFinite(run.scores.combined)) continue;
        if (!groups.has(set)) groups.set(set, new Map());
        const settings = groups.get(set);
        const key = sweep ? `${run.model}/${run.name.split("/")[3]}` : run.model;
        if (!settings.has(key)) settings.set(key, run.scores.combined);
    }
    return [...groups].map(([name, scores]) => ({
        name: name.replaceAll("_", " "),
        models: scores.size,
        mean: [...scores.values()].reduce((sum, score) => sum + score, 0) / scores.size,
    })).sort((a, b) => b.mean - a.mean || a.name.localeCompare(b.name));
}

function leadingImportanceRun(runs, stage) {
    const available = runs.filter((run) => Number.isFinite(run.scores.combined)
        && Object.keys(run.metrics.feature_importance ?? {}).length);
    const matching = available.filter((run) => run.evaluated_on === stage);
    // Fall back to a single evaluation type so CV and test scores aren't ranked together.
    const fallbackStages = ["binary_cv5", "study_cv5", "cv", "binary_test", "study_test", "test"];
    const fallback = fallbackStages.find((name) => available.some((run) => run.evaluated_on === name))
        ?? available[0]?.evaluated_on;
    const pool = matching.length ? matching : available.filter((run) => run.evaluated_on === fallback);
    return pool.sort((a, b) => b.scores.combined - a.scores.combined || b.id - a.id)[0];
}

function renderFeatureAnalysis() {
    const target = byId("analysis-target").value;
    const stage = byId("analysis-stage");
    const runs = state.runs.filter((run) => run.target === target);
    const available = [...new Set(runs.filter((run) => run.data.feature_set || run.data.features?.length)
        .map((run) => run.evaluated_on))];
    const previous = stage.value;
    const preferred = target === "binary" ? "binary_cv3" : "study_cv3";
    stage.replaceChildren(...available.map((value) => new Option(labels[value] ?? value, value)));
    stage.disabled = !available.length;
    stage.value = available.includes(previous) ? previous : available.includes(preferred) ? preferred : available[0] ?? "";

    const ranking = rankFeatureSets(runs, stage.value);
    byId("analysis-status").textContent = ranking.length
        ? `Leading feature set: ${ranking[0].name} (${format(ranking[0].mean)} combined)`
        : "Feature-set scores are unavailable for this target.";
    const featureSets = byId("feature-sets");
    featureSets.replaceChildren(element("h3", "Feature-set scores"));
    if (ranking.length) {
        const results = new Map(ranking.map((entry) => [entry.name, {
            "Mean combined": entry.mean,
            [stage.value.endsWith("_sweep_cv3") ? "Settings" : "Models"]: entry.models,
        }]));
        const table = tableDetails("Ranked feature sets", results,
            (value, key) => key === "Mean combined" ? format(value) : value, "Feature set");
        table.open = true;
        featureSets.append(
            element("p", stage.value.endsWith("_sweep_cv3")
                ? "Average across XGBoost settings for each feature set. The settings count shows coverage."
                : "Average of the latest run per model and feature set for this evaluation. Model coverage and settings can vary."),
            table,
        );
    }

    const individual = byId("feature-ranking");
    individual.replaceChildren(element("h3", "Individual feature importance"));
    const source = leadingImportanceRun(runs, stage.value);
    if (!source) {
        individual.append(element("p", "No feature importance has been saved for this target yet."));
        return;
    }
    const link = element("button", `#${source.id} · ${source.model} · ${source.stage}`, "run-button");
    link.type = "button";
    link.addEventListener("click", () => openRun(source));
    individual.append(
        element("p", "Highest-scoring run with saved importance in the evaluation shown below. Importance describes the fitted model's use of a feature."),
        link,
        featureImportance(source),
    );
}

function errorCounts(confusion) {
    const classes = Object.keys(confusion);
    const total = classes.reduce((sum, actual) =>
        sum + classes.reduce((count, predicted) => count + confusion[actual][predicted], 0), 0);
    return Object.fromEntries(classes.map((label) => {
        const tp = confusion[label][label];
        const fp = classes.reduce((sum, actual) => sum + confusion[actual][label], 0) - tp;
        const fn = classes.reduce((sum, predicted) => sum + confusion[label][predicted], 0) - tp;
        const tn = total - tp - fp - fn;
        return [label, {
            TP: tp, TN: tn, FP: fp, FN: fn,
            "False-positive rate": fp + tn ? fp / (fp + tn) : null,
            "False-negative rate": fn + tp ? fn / (fn + tp) : null,
        }];
    }));
}

function openRun(run) {
    const otherRuns = state.runs.filter((item) => item.id !== run.id && item.target === run.target);
    const baseline = otherRuns.find((item) =>
        item.id < run.id && item.model === run.model && item.evaluated_on === run.evaluated_on);
    const select = element("select");
    select.append(new Option("Select a run", ""));
    for (const item of otherRuns) {
        select.append(new Option(`#${item.id} · ${item.model} · ${item.stage} · ${item.configuration}`, item.id));
    }
    select.value = baseline?.id ?? "";
    const compare = element("label", "Compare with");
    compare.append(select);
    const changes = element("ul");
    const scoreChange = element("p");
    const updateChanges = () => {
        const selected = otherRuns.find((item) => item.id === Number(select.value));
        changes.replaceChildren(...differenceList(run, selected).map((text) => element("li", text)));
        scoreChange.hidden = !selected || selected.evaluated_on !== run.evaluated_on;
        if (!scoreChange.hidden) {
            const delta = run.scores.combined - selected.scores.combined;
            scoreChange.textContent = `Combined score change: ${delta > 0 ? "+" : ""}${format(delta)}`;
        }
    };
    select.addEventListener("change", updateChanges);
    updateChanges();

    const featureNames = run.data.features?.join(", ") || "Feature names were not recorded for this run.";
    const settings = element("details");
    settings.append(element("summary", "Model settings and calibration"), element("pre", JSON.stringify({
        model: run.params,
        data: run.data,
        calibration: run.metrics.calibration_by_fold ?? run.metrics.calibration,
        git_commit: run.git_commit,
    }, null, 2)));
    byId("detail-title").textContent = `#${run.id} · ${run.model}`;
    byId("detail-meta").textContent = `${run.stage} · ${run.target} · ${dateFormat.format(new Date(run.timestamp))}`;
    const confusion = run.metrics.confusion_matrix ?? {};
    const errors = tableDetails("TP, TN, FP, FN and error rates", errorCounts(confusion), (value, key) => {
        if (!key.endsWith("rate")) return value;
        return value === null ? "—" : `${format(value * 100, 2)}%`;
    });
    errors.open = true;
    byId("detail-body").replaceChildren(
        section("Scores", scoreList(run)),
        section("Changes", compare, scoreChange, changes),
        section("Prediction errors",
            element("p", "Each row treats that class as positive. False-positive rate is FP ÷ (FP + TN). False-negative rate is FN ÷ (FN + TP)."),
            errors),
        section("Features", element("p", featureNames)),
        section("Feature importance", featureImportance(run)),
        section("Run notes", element("p", run.notes || "No notes recorded.")),
        section("Run details", settings,
            tableDetails("Threshold metrics", run.metrics.threshold_metrics ?? {}),
            tableDetails("Probability metrics", run.metrics.probability_metrics ?? {}),
            tableDetails("Confusion matrix", confusion)),
    );
    if (run.data.source === "RawData.csv" && run.data.features?.length && run.data.target_classes?.length) {
        const predictLink = element("a", "Use this run on the Predict page");
        predictLink.href = `/predict?run=${run.id}`;
        byId("detail-body").prepend(predictLink);
    }
    const dialog = byId("details");
    dialog.scrollTop = 0;
    dialog.showModal();
}

async function refreshRuns() {
    const button = byId("refresh");
    button.disabled = true;
    byId("status").textContent = "Loading history…";
    try {
        const response = await fetch("/api/runs");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        state.runs = (await response.json()).map(normalize);
        state.loaded = true;
        state.error = "";
        const stage = byId("stage");
        const selected = stage.value;
        const stages = [...new Set(state.runs.map((run) => run.stage))].sort();
        stage.replaceChildren(new Option("All evaluations", ""), ...stages.map((name) => new Option(name, name)));
        if (stages.includes(selected)) stage.value = selected;
        updateSummary();
        renderFeatureAnalysis();
    } catch (error) {
        state.error = `Unable to load runs (${error.message}).`;
    } finally {
        button.disabled = false;
        renderRuns();
    }
}

for (const id of filters) byId(id).addEventListener(id === "search" ? "input" : "change", renderRuns);
for (const id of ["analysis-target", "analysis-stage"]) byId(id).addEventListener("change", renderFeatureAnalysis);
byId("refresh").addEventListener("click", refreshRuns);
refreshRuns();
