// --- 1. CONFIGURATION & GLOBAL STATE ---
const MODEL_URL = 'https://raw.githubusercontent.com/episphere/delphi-onnx/main/delphi.onnx';
const LABELS_URL = 'https://raw.githubusercontent.com/episphere/delphi-onnx/main/delphi_labels_chapters_colours_icd.json';

let delphiModel = null;
let allLabels = []; // To store the parsed JSON labels for populating the dropdowns
let isInitialized = false;

// --- 2. INITIALIZATION FUNCTION ---

/**
 * Loads the model and labels, then populates the initial dropdown.
 */
async function initializeDelphi() {
    const resultElement = document.getElementById('prediction-result');
    resultElement.textContent = 'Loading model and disease labels... Please wait.';

    try {
        // A. Load Labels JSON for populating the user-friendly dropdown
        const response = await fetch(LABELS_URL);
        const labelsData = await response.json();
        // Filter out non-ICD-10 entries (like 'SEX_M', 'SEX_F', 'age')
        allLabels = labelsData.filter(item => item.name.includes(' '));

        // B. Initialize the DelphiONNX model (The DelphiSDK.js file must be loaded)
        delphiModel = new DelphiONNX(MODEL_URL, LABELS_URL);
        console.log('DelphiONNX instance created. Starting model load...');
        await delphiModel.initialize();
        isInitialized = true;

        console.log('Model and labels loaded successfully.');

        // C. Populate Dropdowns for ALL existing event groups (e.g., the default one in HTML)
        document.querySelectorAll('.event-input-group').forEach(group => {
            const codeInput = group.querySelector('.event-code');
            const initialCode = codeInput ? codeInput.value : '';
            // Replace the initial hidden text input with a <select> dropdown
            if (codeInput) {
                codeInput.replaceWith(createDiseaseDropdown(initialCode));
            }
        });

        // D. Update UI
        resultElement.textContent = 'Model ready! Enter patient history and click "Get Prediction".';
        // Enable buttons once the model is fully loaded
        document.querySelectorAll('button').forEach(btn => btn.disabled = false);

    } catch (error) {
        console.error('Error during initialization:', error);
        resultElement.textContent = `Initialization failed. Error: ${error.message}`;
    }
}

/**
 * Creates and returns the disease dropdown element populated with ICD-10 data.
 * @param {string} selectedIcd The ICD-10 code to pre-select.
 * @returns {HTMLElement} The created <select> element.
 */
function createDiseaseDropdown(selectedIcd) {
    const select = document.createElement('select');
    select.className = 'event-code-select'; // Class used for identifying the input during prediction
    select.required = true;

    // Default 'Please Select' option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select Disease --';
    select.appendChild(defaultOption);

    // Populate with actual disease data
    allLabels.forEach(item => {
        // Skip non-disease/control tokens
        if (!item.name.includes(' ')) return;

        const option = document.createElement('option');
        // The value is the ICD-10 code (e.g., 'I10')
        option.value = item.name.split(' ')[0];
        // The visible text is the Chapter + full name (e.g., 'I. Infectious Diseases | I10 Essential hypertension')
        option.textContent = item['ICD-10 Chapter (short)'] + ' | ' + item.name;

        if (option.value === selectedIcd) {
            option.selected = true;
        }
        select.appendChild(option);
    });

    return select;
}

// --- 3. EVENT MANAGEMENT FUNCTION ---

/**
 * Adds a new event input row with the disease dropdown. This is called by the "Add another event" button.
 */
function addEventInput() {
    if (!isInitialized) return; // Prevent adding events if labels haven't loaded

    const eventsContainer = document.getElementById('events-container');
    const index = eventsContainer.children.length; // New ID index

    const newGroup = document.createElement('div');
    newGroup.className = 'event-input-group';
    newGroup.innerHTML = `
        <label for="age-${index}">Event Age:</label>
        <input type="number" class="event-age" id="age-${index}" value="" min="1" max="100">

        <label for="code-${index}">Disease Event:</label>
        <span id="code-placeholder-${index}"></span> <button type="button" onclick="this.closest('.event-input-group').remove()">Remove</button>
    `;

    eventsContainer.appendChild(newGroup);

    // Insert the populated dropdown into the placeholder span
    const placeholder = newGroup.querySelector(`#code-placeholder-${index}`);
    placeholder.replaceWith(createDiseaseDropdown('')); // Create empty dropdown
}

// --- 4. PREDICTION LOGIC FUNCTION ---

/**
 * Gathers inputs and runs the Delphi prediction. This is called by the "Get Prediction" button.
 */
async function runDelphiPrediction() {
    const resultElement = document.getElementById('prediction-result');
    if (!delphiModel || !isInitialized) {
        resultElement.textContent = 'Model is not ready. Please wait for initialization.';
        return;
    }

    resultElement.textContent = 'Calculating prediction...';

    try {
        // A. Gather Inputs
        const currentAge = parseInt(document.getElementById('current-age').value);
        if (isNaN(currentAge) || currentAge < 1) {
            resultElement.textContent = 'Error: Please enter a valid Current Age.';
            return;
        }

        const events = [];
        const eventGroups = document.querySelectorAll('.event-input-group');

        eventGroups.forEach(group => {
            const ageInput = group.querySelector('.event-age');
            const codeSelect = group.querySelector('.event-code-select');

            const eventAge = parseInt(ageInput.value);
            // Get the selected value, which is the ICD-10 code (e.g., 'I10')
            const icdCode = codeSelect ? codeSelect.value : '';

            // Only push valid events
            if (icdCode && !isNaN(eventAge) && eventAge > 0 && eventAge <= currentAge) {
                events.push({ age: eventAge, icd: icdCode });
            }
        });

        // B. Run Prediction
        const prediction = await delphiModel.generateTrajectory({
            current_age: currentAge,
            history: events
        });

        // C. Display Output
        const predictionAge = Math.round(prediction.predicted_age);
        const nextEvent = prediction.predicted_event; // Full description
        const nextICD = prediction.predicted_icd;     // ICD-10 code
        const score = prediction.score.toFixed(4);

        resultElement.innerHTML = `
            **Predicted Next Event:** \nEvent: ${nextEvent} (ICD-10: ${nextICD})
            \nPredicted Age: ${predictionAge} years old
            \nConfidence Score: ${score}
        `;

    } catch (error) {
        console.error('Prediction failed:', error);
        resultElement.textContent = `Prediction Error: ${error.message}.`;
    }
}

// --- 5. APPLICATION START ---

// Start the model initialization when the page content is fully loaded
document.addEventListener('DOMContentLoaded', initializeDelphi);
