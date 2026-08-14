import DelphiONNX from "https://episphere.github.io/delphi-onnx/delphiSDK.js"

const NUM_DAYS_IN_A_YEAR = 365.25;
let sdk = null;
let isModelReady = false;
let hasAcknowledgedNotice = false;

function updatePredictButtonState() {
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) predictBtn.disabled = !isModelReady || !hasAcknowledgedNotice;
}

function acknowledgeResearchNotice() {
    hasAcknowledgedNotice = true;

    const acknowledgmentButton = document.getElementById('acknowledge-notice-btn');
    const acknowledgmentHelp = document.getElementById('acknowledgment-help');
    if (acknowledgmentButton) {
        acknowledgmentButton.disabled = true;
        acknowledgmentButton.textContent = 'Research notice acknowledged';
        acknowledgmentButton.classList.remove('bg-gray-800', 'hover:bg-gray-900');
        acknowledgmentButton.classList.add('bg-green-700', 'cursor-default');
    }
    if (acknowledgmentHelp) {
        acknowledgmentHelp.textContent = isModelReady
            ? 'Acknowledged. Get Prediction is now available.'
            : 'Acknowledged. Get Prediction will become available when the model is ready.';
    }

    updatePredictButtonState();
}

// Define your full event list
const rawEventsList = [
    { "event": "Male", "age": 0 },
    { "event": "B01 Varicella [chickenpox]", "age": 2 },
    { "event": "L20 Atopic dermatitis", "age": 3 },
    { "event": "No event", "age": 5 },
    { "event": "No event", "age": 10 },
    { "event": "No event", "age": 15 },
    { "event": "No event", "age": 20 },
    { "event": "G43 Migraine", "age": 20 },
    { "event": "E73 Lactose intolerance", "age": 21 },
    { "event": "B27 Infectious mononucleosis", "age": 22 },
    { "event": "No event", "age": 25 },
    { "event": "J11 Influenza, virus not identified", "age": 28 },
    { "event": "No event", "age": 30 },
    { "event": "No event", "age": 35 },
    { "event": "No event", "age": 40 },
    { "event": "Smoking low", "age": 41 },
    { "event": "BMI mid", "age": 41 },
    { "event": "Alcohol low", "age": 41 },
    { "event": "No event", "age": 42 }
];

// Create disease dropdown function
window.createDiseaseDropdown = function(selectedEvent) {
    const select = document.createElement('select');
    // give disease selects a class that allows them to expand to fill the row
    select.className = 'event-code-select event-disease';
    select.required = true;

    // Add default empty option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select Disease --';
    select.appendChild(defaultOption);

    // Prefer SDK vocabulary if available, but also include non-ICD events from the local rawEventsList
    let optionsSource = [];
    const localEvents = rawEventsList.map(e => e.event).filter(ev => ev && ev !== 'Male' && ev !== 'Female');

    if (sdk && sdk.tokenIdToName) {
        // SDK labels (ICD-like tokens) first
        const sdkEvents = Object.values(sdk.tokenIdToName).filter(event => !!event);

        // Merge: SDK events first, then any local events that are not already present
        const merged = [...sdkEvents];
        localEvents.forEach(ev => { if (!merged.includes(ev)) merged.push(ev); });
        // Filter to show ICD-like entries and also keep 'No event' and the non-ICD locals
        optionsSource = merged.filter(event => event && (event === 'No event' || event.match(/^[A-Z][0-9]/) || localEvents.includes(event)));
    } else {
        // SDK not present: use local events only
        optionsSource = localEvents;
    }

    // Deduplicate while preserving order
    const seen = new Set();
    optionsSource.forEach(event => {
        if (seen.has(event)) return;
        seen.add(event);
        const option = document.createElement('option');
        option.value = event;
        option.textContent = event;
        if (event === selectedEvent) option.selected = true;
        select.appendChild(option);
    });

    console.debug('createDiseaseDropdown: populated', { sourceCount: optionsSource.length, selectedEvent });

    return select;
}

// Initialize the form with the first valid event
async function initializeForm() {
    // Initialize SDK if not already initialized so dropdowns use SDK labels
    if (!sdk) {
        sdk = new DelphiONNX({ seed: 100 });
        try {
            await sdk.initialize();
            isModelReady = true;
            console.debug('initializeForm: SDK initialized');
        } catch (err) {
            isModelReady = false;
            console.error('initializeForm: SDK initialization failed', err);
        }
    }

    // Populate events-container with all events from rawEventsList (include sex events)
    const eventsContainer = document.getElementById('events-container');
    if (eventsContainer) {
        eventsContainer.innerHTML = '';
        let idx = 0;
        rawEventsList.forEach(event => {
            if (!event) return;

            const isSex = event.event === 'Male' || event.event === 'Female';

            // create group
            const newGroup = document.createElement('div');
            newGroup.className = 'event-input-group bg-gray-50 rounded-lg p-3 shadow-sm';

            if (isSex) {
                // For sex tokens we don't ask for age — do not display age, include hidden age input for internal use
                newGroup.innerHTML = `
                    <label class="font-medium text-gray-600">Sex:</label>
                    <span id="sex-placeholder-${idx}"></span>
                    <input type="hidden" class="event-age" value="0">
                    ${idx > 0 ? '<button type="button" onclick="this.closest(\'.event-input-group\').remove()" class="remove-btn text-sm text-red-600 hover:text-red-800 transition duration-150">Remove</button>' : ''}
                `;
            } else {
                newGroup.innerHTML = `
                    <label for="age-${idx}" class="font-medium text-gray-600">Event Age:</label>
                    <input type="number" class="event-age" id="age-${idx}" value="${event.age}" min="0" max="120">
                    <div class="event-row">
                        <label for="code-${idx}" class="font-medium text-gray-600">Disease Event:</label>
                        <span id="code-placeholder-${idx}"></span>
                        ${idx > 0 ? '<button type="button" onclick="this.closest(\'.event-input-group\').remove()" class="remove-btn text-sm text-red-600 hover:text-red-800 transition duration-150">Remove</button>' : ''}
                    </div>
                `;
            }

            eventsContainer.appendChild(newGroup);

            if (isSex) {
                const placeholder = newGroup.querySelector(`#sex-placeholder-${idx}`);
                const sexSelect = document.createElement('select');
                sexSelect.className = 'event-sex-select';
                sexSelect.required = true;

                const defaultOpt = document.createElement('option');
                defaultOpt.value = '';
                defaultOpt.textContent = '-- Select Sex --';
                sexSelect.appendChild(defaultOpt);

                const maleOpt = document.createElement('option');
                maleOpt.value = 'Male';
                maleOpt.textContent = 'Male';
                if (event.event === 'Male') maleOpt.selected = true;
                sexSelect.appendChild(maleOpt);

                const femaleOpt = document.createElement('option');
                femaleOpt.value = 'Female';
                femaleOpt.textContent = 'Female';
                if (event.event === 'Female') femaleOpt.selected = true;
                sexSelect.appendChild(femaleOpt);

                try {
                    placeholder.replaceWith(sexSelect);
                } catch (e) {
                    const ageInput = newGroup.querySelector('.event-age');
                    if (ageInput) ageInput.insertAdjacentElement('afterend', sexSelect);
                    else newGroup.appendChild(sexSelect);
                }
            } else {
                const placeholder = newGroup.querySelector(`#code-placeholder-${idx}`);
                const dropdown = window.createDiseaseDropdown(event.event);
                if (dropdown) {
                    try {
                        placeholder.replaceWith(dropdown);
                    } catch (e) {
                        const ageInput = newGroup.querySelector('.event-age');
                        if (ageInput) ageInput.insertAdjacentElement('afterend', dropdown);
                        else newGroup.appendChild(dropdown);
                    }
                }
            }

            idx++;
        });

        // Enable buttons now that inputs exist
        const addBtn = document.getElementById('add-btn');
        updatePredictButtonState();
        if (hasAcknowledgedNotice) {
            const acknowledgmentHelp = document.getElementById('acknowledgment-help');
            if (acknowledgmentHelp) {
                acknowledgmentHelp.textContent = isModelReady
                    ? 'Acknowledged. Get Prediction is now available.'
                    : 'Acknowledged, but Get Prediction is unavailable because the model could not be loaded.';
            }
        }
        if (addBtn) addBtn.disabled = false;
    }
}

// After SDK is initialized, refresh all existing dropdowns to use SDK vocabulary
function refreshDropdownsUsingSdk() {
    if (!sdk || !sdk.tokenIdToName) return;
    const selects = document.querySelectorAll('.event-code-select');
    selects.forEach(select => {
        const currentValue = select.value;
        const newSelect = window.createDiseaseDropdown(currentValue);
        if (newSelect) select.replaceWith(newSelect);
    });
}

async function runDelphiPrediction() {
    const resultElement = document.getElementById('prediction-result');
    if (!resultElement) return;

    if (!hasAcknowledgedNotice) {
        resultElement.textContent = 'Please acknowledge the research notice before generating a demonstration trajectory.';
        document.getElementById('acknowledge-notice-btn')?.focus();
        return;
    }

    try {
        if (!sdk) {
            sdk = new DelphiONNX({ seed: 100 });
            await sdk.initialize();
            isModelReady = true;
            updatePredictButtonState();
        }

        // SDK is ready: refresh dropdowns to use SDK labels if available
        try { refreshDropdownsUsingSdk(); } catch (e) { console.warn('Failed to refresh dropdowns with SDK labels:', e); }

        resultElement.textContent = 'Calculating full trajectory... Please wait.';

        // Read events from the page (allows user edits)
        const events = [];
        const eventGroups = document.querySelectorAll('.event-input-group');
        let maxAge = 0;

        eventGroups.forEach(group => {
            const ageInput = group.querySelector('.event-age');
            let codeSelect = group.querySelector('.event-code-select');
            if (!codeSelect) codeSelect = group.querySelector('.event-sex-select');

            const eventAge = ageInput ? parseInt(ageInput.value) : NaN;
            const eventName = codeSelect ? codeSelect.value : '';

            if (isNaN(eventAge) || eventName === '') {
                return; // skip invalid rows
            }

            events.push({ age: eventAge, eventName });
            if (eventAge > maxAge) maxAge = eventAge;
        });

        if (events.length === 0) {
            resultElement.textContent = 'Error: Please add at least one valid past health event.';
            return;
        }

        // Sort by age
        events.sort((a,b) => a.age - b.age);

        const icdEvents = events.map(e => e.eventName);
        const eventAges = events.map(e => e.age);

        let tokenHistory, ageHistoryDays;
        try {
            // const startToken = sdk.utils.ageTokenName || sdk.utils.tokenIdToName['1'] || null;
            // if (!startToken) {
            //     throw new Error('Starting age token not found in SDK vocabulary.');
            // }
            // tokenHistory = sdk.getTokensFromEvents([startToken, ...icdEvents]);
            // ageHistoryDays = sdk.convertAgeToDays([0, ...eventAges]);
            tokenHistory = sdk.getTokensFromEvents(icdEvents)
            ageHistoryDays = sdk.convertAgeToDays(eventAges)
        } catch (e) {
            console.error('Input processing failed:', e);
            resultElement.textContent = `Input Error: ${e.message}. Check console for details.`;
            return;
        }

        // Run trajectory generation
        const prediction = await sdk.generateTrajectory(tokenHistory, ageHistoryDays, {
            maxAge: 85 * NUM_DAYS_IN_A_YEAR
        });

        // Render results as a table similar to delphi.js
        const inputLength = tokenHistory.length;
        const predictedTokens = prediction.tokenIds;
        const predictedAgesDays = prediction.age;
        const terminationTokenId = 1269;

        let html = `
            <table class="trajectory-table">
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Predicted Age (Years)</th>
                        <th>Event (ICD-10)</th>
                        <th>Event Description</th>
                    </tr>
                </thead>
                <tbody>
        `;

        let stepCounter = 1;
        for (let i = inputLength; i < predictedTokens.length; i++) {
            const token = predictedTokens[i];
            const ageDays = predictedAgesDays[i];
            const ageYears = sdk.convertAgeToYears(ageDays, 1);
            const eventName = sdk.getEventsFromTokens(token);

            if (token === 0) continue;

            let icdCode = 'N/A';
            let description = 'N/A';
            if (eventName) {
                if (eventName === 'No event') {
                    icdCode = '-';
                    description = 'No event';
                } else if (token === terminationTokenId || eventName === 'Death') {
                    // Render death/termination token specially
                    icdCode = '-';
                    description = 'Death';
                } else {
                    const parts = eventName.split(' ');
                    icdCode = parts[0] || 'N/A';
                    description = parts.slice(1).join(' ') || 'N/A';
                }
            }

            const isDeath = (token === terminationTokenId || description === 'Death');
            html += `
                <tr>
                    <td>${stepCounter++}</td>
                    <td class="${isDeath ? 'text-red-600 font-bold' : ''}">${ageYears}</td>
                    <td class="${isDeath ? 'text-red-600 font-bold' : ''}">${icdCode}</td>
                    <td class="${isDeath ? 'text-red-600 font-bold' : ''}">${description}</td>
                </tr>
            `;

            if (token === terminationTokenId) break;
        }

        if (stepCounter === 1) {
            resultElement.textContent = `No new events were predicted up to the maximum simulation age.`;
        } else {
            html += '</tbody></table>';
            resultElement.innerHTML = html;
        }

    } catch (error) {
        console.error('Prediction failed:', error);
        const errorMessage = error.message || 'An unknown error occurred.';
        resultElement.textContent = `Prediction Error: ${errorMessage}. Please check your inputs or the console for details.`;
    }
}

// Initialize the form, but do not generate a prediction before acknowledgment.
document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('acknowledge-notice-btn')
        ?.addEventListener('click', acknowledgeResearchNotice, { once: true });
    await initializeForm();
    const resultElement = document.getElementById('prediction-result');
    if (resultElement && isModelReady) {
        resultElement.textContent = 'Model ready. Acknowledge the research notice, then enter or review the history and select Get Prediction.';
    } else if (resultElement) {
        resultElement.textContent = 'The model could not be loaded. Please check your internet connection and try again.';
    }
});

// Expose functions to global scope so inline handlers in index.html work
window.runDelphiPrediction = runDelphiPrediction;

// Add event input dynamically (used by the "Add another event" button)
window.addEventInput = function() {
    const eventsContainer = document.getElementById('events-container');
    if (!eventsContainer) return;
    const index = eventsContainer.children.length;

    const newGroup = document.createElement('div');
    newGroup.className = 'event-input-group bg-gray-50 rounded-lg p-3 shadow-sm';

    newGroup.innerHTML = `
        <label for="age-${index}" class="font-medium text-gray-600">Event Age:</label>
        <input type="number" class="event-age" id="age-${index}" value="" min="0" max="120">
        <div class="event-row">
            <label for="code-${index}" class="font-medium text-gray-600">Disease Event:</label>
            <span id="code-placeholder-${index}"></span>
            <button type="button" onclick="this.closest('.event-input-group').remove()" class="remove-btn text-sm text-red-600 hover:text-red-800 transition duration-150">Remove</button>
        </div>
    `;

    eventsContainer.appendChild(newGroup);

    const placeholder = newGroup.querySelector(`#code-placeholder-${index}`);
    const dropdown = window.createDiseaseDropdown('');
    if (dropdown) {
        try {
            placeholder.replaceWith(dropdown);
        } catch (e) {
            const ageInput = newGroup.querySelector('.event-age');
            if (ageInput) ageInput.insertAdjacentElement('afterend', dropdown);
            else newGroup.appendChild(dropdown);
        }
    }
}
