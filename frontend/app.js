/**
 * Frontend logic for Privacy-Preserving Federated Smart Healthcare
 */

let accuracyChart = null;
let lossChart = null;
let chartLabels = [];
let accuracyData = [];
let lossData = [];

// DOM Element references
const backendStatusDot = document.getElementById("backendStatusDot");
const backendStatusText = document.getElementById("backendStatusText");
const headerRoundNum = document.getElementById("headerRoundNum");
const agentGrid = document.getElementById("agentGrid");
const logConsole = document.getElementById("logConsole");
const btnClearLogs = document.getElementById("btnClearLogs");

const inputRounds = document.getElementById("inputRounds");
const btnRunRound = document.getElementById("btnRunRound");
const btnReset = document.getElementById("btnReset");
const toggleDP = document.getElementById("toggleDP");
const sliderEpsilon = document.getElementById("sliderEpsilon");
const epsilonVal = document.getElementById("epsilonVal");
const toggleSecAgg = document.getElementById("toggleSecAgg");

const statAcc = document.getElementById("statAcc");
const statLoss = document.getElementById("statLoss");

// Prediction elements
const predictionForm = document.getElementById("predictionForm");
const btnPrefillSample = document.getElementById("btnPrefillSample");
const predictionResult = document.getElementById("predictionResult");
const resRiskBadge = document.getElementById("resRiskBadge");
const resProb = document.getElementById("resProb");
const resRound = document.getElementById("resRound");
const resPrivacyNotice = document.getElementById("resPrivacyNotice");

// Initialize Charts
function initCharts() {
    const ctxAcc = document.getElementById("accuracyChart").getContext("2d");
    const ctxLoss = document.getElementById("lossChart").getContext("2d");

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                grid: { color: "rgba(51, 65, 85, 0.4)" },
                ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } }
            },
            y: {
                grid: { color: "rgba(51, 65, 85, 0.4)" },
                ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } }
            }
        }
    };

    accuracyChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Global Accuracy',
                data: accuracyData,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                tension: 0.35,
                fill: true,
                pointBackgroundColor: '#10b981'
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, min: 0, max: 1 }
            }
        }
    });

    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Global Loss',
                data: lossData,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                borderWidth: 2,
                tension: 0.35,
                fill: true,
                pointBackgroundColor: '#f59e0b'
            }]
        },
        options: commonOptions
    });
}

function appendLog(message) {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement("div");
    
    // Style highlighting based on tags
    if (message.includes("[PRIVACY_DP]")) {
        entry.className = "text-teal-400 font-mono";
    } else if (message.includes("[SECAGG]")) {
        entry.className = "text-indigo-400 font-mono";
    } else if (message.includes("[FED_AGGREGATOR]") || message.includes("[GLOBAL_EVAL]")) {
        entry.className = "text-emerald-400 font-bold font-mono";
    } else if (message.includes("Initiated")) {
        entry.className = "text-sky-300 font-semibold font-mono";
    } else {
        entry.className = "text-slate-300 font-mono";
    }
    
    entry.textContent = `[${time}] ${message}`;
    logConsole.appendChild(entry);
    logConsole.scrollTop = logConsole.scrollHeight;
}

// Fetch Agent Metadata & Status
async function loadAgents() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/agents`);
        if (!res.ok) throw new Error("Failed to load agent metadata");
        const agents = await res.json();
        renderAgents(agents);
        setBackendOnline(true);
    } catch (err) {
        console.error(err);
        setBackendOnline(false);
    }
}

function setBackendOnline(online) {
    if (online) {
        backendStatusDot.className = "h-2.5 w-2.5 rounded-full bg-emerald-500";
        backendStatusText.textContent = "Federated Core Online";
    } else {
        backendStatusDot.className = "h-2.5 w-2.5 rounded-full bg-rose-500 animate-ping";
        backendStatusText.textContent = "Backend Disconnected";
    }
}

function renderAgents(agents) {
    agentGrid.innerHTML = "";
    agents.forEach(agent => {
        const card = document.createElement("div");
        card.id = `card-${agent.agent_id}`;
        card.className = "agent-card bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col justify-between";
        
        let statusColor = "bg-slate-700 text-slate-300";
        if (agent.status.includes("Training")) {
            statusColor = "bg-amber-950 text-amber-300 border border-amber-800 animate-pulse";
        } else if (agent.status.includes("Complete") || agent.status.includes("Aggregated")) {
            statusColor = "bg-emerald-950 text-emerald-300 border border-emerald-800";
        }

        card.innerHTML = `
            <div>
                <div class="flex items-center justify-between mb-2">
                    <h4 class="font-bold text-sm text-slate-200 tracking-wide">${agent.name}</h4>
                    <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full ${statusColor}">${agent.status}</span>
                </div>
                <p class="text-xs text-slate-400 mb-3">${agent.bias_description}</p>
                
                <div class="grid grid-cols-2 gap-2 text-xs font-mono bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80 mb-3">
                    <div>
                        <span class="text-slate-500 block text-[10px]">PATIENT SAMPLES</span>
                        <span class="text-teal-400 font-bold">${agent.sample_count} records</span>
                    </div>
                    <div>
                        <span class="text-slate-500 block text-[10px]">MEAN AGE / MALE</span>
                        <span class="text-slate-200">${agent.mean_age}y / ${(agent.male_ratio * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                        <span class="text-slate-500 block text-[10px]">LOCAL ACCURACY</span>
                        <span class="text-emerald-400 font-bold">${(agent.last_accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                        <span class="text-slate-500 block text-[10px]">LOCAL LOSS</span>
                        <span class="text-amber-400 font-bold">${agent.last_loss.toFixed(3)}</span>
                    </div>
                </div>
            </div>
            
            <div class="text-[11px] text-slate-500 flex items-center justify-between pt-2 border-t border-slate-800">
                <span>Healthy: ${agent.target_distribution.healthy}</span>
                <span>Disease: ${agent.target_distribution.disease}</span>
            </div>
        `;
        agentGrid.appendChild(card);
    });
}

// Run Federated Communication Round
async function runFederatedRound() {
    const rounds = parseInt(inputRounds.value) || 1;
    const useDP = toggleDP.checked;
    const epsilon = parseFloat(sliderEpsilon.value) || 1.0;
    const useSecAgg = toggleSecAgg.checked;

    btnRunRound.disabled = true;
    btnRunRound.innerHTML = `<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Aggregating...`;

    try {
        const payload = {
            rounds: rounds,
            use_dp: useDP,
            epsilon: epsilon,
            use_secagg: useSecAgg
        };

        const res = await fetch(`${API_BASE_URL}/api/train-round`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Failed to execute federated training round");
        const data = await res.json();

        // Update header & metrics
        headerRoundNum.textContent = data.current_round;
        statAcc.textContent = `${(data.global_accuracy * 100).toFixed(2)}%`;
        statLoss.textContent = data.global_loss.toFixed(4);

        // Update logs
        data.logs.forEach(msg => appendLog(msg));

        // Re-render agents with fresh metrics
        await loadAgents();

        // Update Charts
        updateCharts(data.history);

    } catch (err) {
        console.error(err);
        appendLog(`[ERROR] Communication failure: ${err.message}`);
    } finally {
        btnRunRound.disabled = false;
        btnRunRound.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> Run Next Round`;
    }
}

function updateCharts(history) {
    chartLabels = history.map(h => `R${h.round}`);
    accuracyData = history.map(h => h.accuracy);
    lossData = history.map(h => h.loss);

    accuracyChart.data.labels = chartLabels;
    accuracyChart.data.datasets[0].data = accuracyData;
    accuracyChart.update();

    lossChart.data.labels = chartLabels;
    lossChart.data.datasets[0].data = lossData;
    lossChart.update();
}

// Reset Server State
async function resetServer() {
    if (!confirm("Are you sure you want to reset the global model and all agents to Round 0?")) return;
    try {
        const res = await fetch(`${API_BASE_URL}/api/reset`, { method: "POST" });
        if (!res.ok) throw new Error("Reset failed");
        
        headerRoundNum.textContent = "0";
        statAcc.textContent = "0.00%";
        statLoss.textContent = "0.0000";
        
        chartLabels = [];
        accuracyData = [];
        lossData = [];
        accuracyChart.data.labels = [];
        accuracyChart.data.datasets[0].data = [];
        accuracyChart.update();
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update();

        predictionResult.classList.add("hidden");
        appendLog("[SYSTEM] Successfully reset global federated environment.");
        await loadAgents();
    } catch (err) {
        console.error(err);
        appendLog(`[ERROR] Reset failure: ${err.message}`);
    }
}

// Prefill Patient Predictor with Realistic Clinical Profile
function prefillSamplePatient() {
    document.getElementById("inputAge").value = 62;
    document.getElementById("inputSex").value = 1;
    document.getElementById("inputCp").value = 3; // Asymptomatic
    document.getElementById("inputTrestbps").value = 145;
    document.getElementById("inputChol").value = 282;
    document.getElementById("inputFbs").value = 1;
    document.getElementById("inputRestecg").value = 2;
    document.getElementById("inputThalach").value = 142;
    document.getElementById("inputExang").value = 1;
    document.getElementById("inputOldpeak").value = 2.8;
    document.getElementById("inputSlope").value = 1;
    document.getElementById("inputCa").value = 2;
    document.getElementById("inputThal").value = 2;
}

// Patient Prediction via Current Aggregated Global Model
async function handlePrediction(e) {
    e.preventDefault();
    
    const payload = {
        age: parseFloat(document.getElementById("inputAge").value),
        sex: parseInt(document.getElementById("inputSex").value),
        cp: parseInt(document.getElementById("inputCp").value),
        trestbps: parseFloat(document.getElementById("inputTrestbps").value),
        chol: parseFloat(document.getElementById("inputChol").value),
        fbs: parseInt(document.getElementById("inputFbs").value),
        restecg: parseInt(document.getElementById("inputRestecg").value),
        thalach: parseFloat(document.getElementById("inputThalach").value),
        exang: parseInt(document.getElementById("inputExang").value),
        oldpeak: parseFloat(document.getElementById("inputOldpeak").value),
        slope: parseInt(document.getElementById("inputSlope").value),
        ca: parseInt(document.getElementById("inputCa").value),
        thal: parseInt(document.getElementById("inputThal").value)
    };

    try {
        const res = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Inference request failed");
        const pred = await res.json();

        predictionResult.classList.remove("hidden");
        const pct = (pred.probability * 100).toFixed(1);
        resProb.textContent = `${pct}%`;
        resRound.textContent = pred.model_round;

        if (pred.has_disease) {
            resRiskBadge.className = "inline-block mt-1 text-sm font-bold px-3 py-1 rounded-full bg-rose-950 text-rose-300 border border-rose-800";
            resRiskBadge.textContent = pred.risk_level;
            resProb.className = "text-2xl font-mono font-bold text-rose-400";
        } else {
            resRiskBadge.className = "inline-block mt-1 text-sm font-bold px-3 py-1 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800";
            resRiskBadge.textContent = pred.risk_level;
            resProb.className = "text-2xl font-mono font-bold text-emerald-400";
        }

        appendLog(`[PREDICTOR] Inference executed on patient sample -> Disease Risk: ${pct}% (${pred.risk_level})`);

    } catch (err) {
        console.error(err);
        appendLog(`[ERROR] Prediction failed: ${err.message}`);
    }
}

// Event Listeners
sliderEpsilon.addEventListener("input", (e) => {
    epsilonVal.textContent = parseFloat(e.target.value).toFixed(2);
});

btnRunRound.addEventListener("click", runFederatedRound);
btnReset.addEventListener("click", resetServer);
btnClearLogs.addEventListener("click", () => {
    logConsole.innerHTML = "";
    appendLog("[CONSOLE] Log display cleared.");
});
btnPrefillSample.addEventListener("click", prefillSamplePatient);
predictionForm.addEventListener("submit", handlePrediction);

// Initialization on DOM Load
window.addEventListener("DOMContentLoaded", () => {
    initCharts();
    loadAgents();
    setInterval(async () => {
        try {
            const h = await fetch(`${API_BASE_URL}/health`);
            setBackendOnline(h.ok);
        } catch {
            setBackendOnline(false);
        }
    }, 10000);
});

