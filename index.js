
    import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs';

    if (ort.env) {
        ort.env.logLevel = 'error';
    }

    // --- CONFIGURATION & GLOBAL STATE ---
    const MODEL_URL = 'https://episphere.github.io/delphi-onnx/delphi.onnx';
    const LABELS_URL = 'https://raw.githubusercontent.com/episphere/delphi-onnx/main/delphi_labels_chapters_colours_icd.json';
    const NUM_DAYS_IN_A_YEAR = 365.25;
    const PREDICTION_WINDOW_YEARS = 30; // How many years past the last recorded event to predict

    let delphiModel = null;
    let allLabels = [];
    let isInitialized = false;

    // --- HELPER FUNCTIONS & DELPHIONNX CLASS ---

    function mulberry32RNG(seed) {
        let t = seed + 0x6D2B79F5;
        return function() {
            t = 3812017321 * (t ^ t >>> 15);
            t = t ^ t << 13;
            t = t ^ t >>> 17;
            t = t ^ t << 5;
            return (t >>> 0) / 4294967296;
        }
    }

    async function fetchLabels() {
        const response = await fetch(LABELS_URL);
        return response.json();
    }

    class DelphiONNX {
        constructor(options = {}) {
            const { modelURL = MODEL_URL, seed = Date.now() } = options
            this.modelURL = modelURL
            this.seed = seed
            this.session = null;
            this.ageTokenName = null;
        }

        async initialize() {
            this.session = await ort.InferenceSession.create(this.modelURL, {
                executionProviders: ["wasm", "cpu"]
            })

            this.rng = mulberry32RNG(this.seed)

            this.nameToTokenId = {}
            this.tokenIdToName = {}
            const delphiLabels = await fetchLabels()
            delphiLabels.forEach(obj => {
                this.nameToTokenId[obj["name"]] = parseInt(obj["index"])
                this.tokenIdToName[obj["index"]] = obj["name"]
            })

            const ageTokenIndex = '1';
            this.ageTokenName = this.tokenIdToName[ageTokenIndex];

            if (this.ageTokenName === undefined) {
                const errorMsg = "Critical Error: Starting token (index 1) is missing from the vocabulary.";
                console.error(errorMsg);
                throw new Error(errorMsg);
            }

            console.log("DelphiONNX: Model and labels initialized. Starting token: " + this.ageTokenName);
        }

        getTokensFromEvents(events = []) {
            const tokens = events.map(event => this.nameToTokenId[event]);
            if (tokens.some(t => t === undefined)) {
                throw new Error(`One or more input events could not be mapped to a valid token ID.`);
            }
            return tokens;
        }

        getEventsFromTokens(tokens = []) {
            if (Array.isArray(tokens)) {
                return tokens.map(tokenId => this.tokenIdToName[tokenId])
            }
            return this.tokenIdToName[tokens]
        }

        convertAgeToDays(ages = []) {
            if (Array.isArray(ages)) {
                return ages.map(ageInYrs => ageInYrs * NUM_DAYS_IN_A_YEAR)
            }
            return ages * NUM_DAYS_IN_A_YEAR
        }

        convertAgeToYears(ages = [], precision = 1) {
            if (Array.isArray(ages)) {
                return ages.map(ageInDays => (ageInDays / NUM_DAYS_IN_A_YEAR).toFixed(precision))
            }
            return (ages / NUM_DAYS_IN_A_YEAR).toFixed(precision)
        }

        async getLogits(eventTokens, ages, logitsIndex) {
            if (!this.session) { await this.initialize() }
            if (!logitsIndex || !Number.isInteger(logitsIndex)) { logitsIndex = eventTokens.length - 1 }

            if (eventTokens.length !== ages.length) {
                throw new Error(`getLogits: Tokens and Ages must have the same length. Tokens: ${eventTokens.length}, Ages: ${ages.length}`);
            }

            const idxTensor = new ort.Tensor(
                "int64",
                new BigInt64Array(eventTokens.map((x) => BigInt(x))),
                [1, eventTokens.length]
            )

            const ageTensor = new ort.Tensor("float32", new Float32Array(ages), [
                1, ages.length
            ])

            const feeds = { idx: idxTensor, age: ageTensor }
            const results = await this.session.run(feeds)
            let { data: logits, dims: logitsShape } = results.logits

            const numEvents = logitsShape[1]
            const vocabSize = logitsShape[2]

            let logitsForEvent = undefined
            let dims = undefined
            if (Number.isInteger(logitsIndex) && logitsIndex >= 0 && logitsIndex < numEvents) {
                const eventLogitsIndex = logitsIndex * vocabSize
                logitsForEvent = logits.slice(
                    eventLogitsIndex,
                    eventLogitsIndex + vocabSize
                )
                dims = [1, vocabSize]
            } else if (logitsIndex === -1) {
                logitsForEvent = logits
                dims = logitsShape
            }

            return { logits: logitsForEvent, dims }
        }

        async generateTrajectory(idx, age, options = {}) {
            const {
                maxNewTokens = 100,
                maxAge = 85 * NUM_DAYS_IN_A_YEAR,
                noRepeat = true,
                terminationTokens = [1269],
                ignoreTokens = []
            } = options

            const maskTime = -10000
            const finalMaxTokens = maxNewTokens === -1 ? 128 : maxNewTokens

            let currentIdx = [...idx]
            let currentAge = [...age]

            for (let step = 0; step < finalMaxTokens; step++) {
                if (currentIdx.length !== currentAge.length) {
                    throw new Error("Trajectory loop: Index and age arrays are misaligned.");
                }

                const { logits } = await this.getLogits(currentIdx, currentAge)

                ignoreTokens.forEach(ignoreToken => {
                    if (ignoreToken < logits.length) {
                        logits[ignoreToken] = -Infinity
                    }
                })

                if (noRepeat) {
                    currentIdx.forEach(token => {
                        if (token > 1 && token < logits.length) {
                            logits[token] = -Infinity;
                        }
                    })
                }

                const tSamples = logits.map(logit => {
                    const randomSample = -Math.exp(-logit) * Math.log(this.rng())
                    return Math.max(0, randomSample)
                })

                let minTime = Infinity
                let minIndex = -1
                tSamples.forEach((tSample, i) => {
                    if (i > 1 && tSample < minTime) {
                        minTime = tSample;
                        minIndex = i;
                    }
                })

                if (minIndex === -1) {
                    break;
                }

                const ageNext = currentAge[currentAge.length - 1] + minTime
                currentIdx.push(minIndex)
                currentAge.push(ageNext)

                const hasTerminationToken = terminationTokens.includes(minIndex)
                const exceedsMaxAge = ageNext > maxAge;
                if (hasTerminationToken || exceedsMaxAge) {
                    break
                }
            }

            return {
                tokenIds: currentIdx,
                age: currentAge,
            }
        }
    }
    // --- END DELPHIONNX CLASS DEFINITION ---


    // --- APPLICATION LOGIC FUNCTIONS ---

    function createDiseaseDropdown(selectedIcd) {
        const select = document.createElement('select');
        select.className = 'event-code-select';
        select.required = true;

        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '-- Select Disease --';
        select.appendChild(defaultOption);

        allLabels.forEach(item => {
            if (!item.name.includes(' ')) return;

            const option = document.createElement('option');
            option.value = item.name.split(' ')[0];
            option.textContent = item.name;

            if (option.value === selectedIcd) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        return select;
    }

    window.addEventInput = function () {
        if (!isInitialized) return;

        const eventsContainer = document.getElementById('events-container');
        const index = eventsContainer.children.length;

        const newGroup = document.createElement('div');
        newGroup.className = 'event-input-group bg-gray-50 rounded-lg p-3 shadow-sm';

        newGroup.innerHTML = `
            <label for="age-${index}" class="font-medium text-gray-600">Event Age:</label>
            <input type="number" class="event-age w-1/4" id="age-${index}" value="" min="1" max="100">

            <label for="code-${index}" class="font-medium text-gray-600">Disease Event:</label>
            <span id="code-placeholder-${index}"></span>
            <button type="button" onclick="this.closest('.event-input-group').remove()" class="remove-btn text-sm text-red-600 hover:text-red-800 transition duration-150">Remove</button>
        `;

        eventsContainer.appendChild(newGroup);

        const placeholder = newGroup.querySelector(`#code-placeholder-${index}`);
        placeholder.replaceWith(createDiseaseDropdown(''));
    }

    window.runDelphiPrediction = async function () {
        const resultElement = document.getElementById('prediction-result');
        if (!delphiModel || !isInitialized) {
            resultElement.textContent = 'Model is not ready. Please wait for initialization.';
            return;
        }

        resultElement.innerHTML = `<p class="text-blue-600">Calculating full trajectory... Please wait.</p>`;

        const events = [];
        const eventGroups = document.querySelectorAll('.event-input-group');
        let maxAge = 0;

        eventGroups.forEach(group => {
            const ageInput = group.querySelector('.event-age');
            const codeSelect = group.querySelector('.event-code-select');

            const eventAge = parseInt(ageInput.value);
            const icdCode = codeSelect ? codeSelect.value : '';

            // Validation: Must be a valid age >= 1 and a selected ICD code
            if (isNaN(eventAge) || eventAge < 1 || !icdCode) {
                 return;
            }

            const labelItem = allLabels.find(item => item.name.startsWith(icdCode + ' '));
            const fullEventName = (labelItem && typeof labelItem.name === 'string') ? labelItem.name : null;

            if (fullEventName) {
                events.push({ age: eventAge, eventName: fullEventName });
                if (eventAge > maxAge) {
                    maxAge = eventAge;
                }
            }
        });

        events.sort((a, b) => a.age - b.age);

        if (events.length === 0) {
             resultElement.textContent = 'Error: Please add at least one valid past health event.';
             return;
        }

        // The current age (starting point for prediction) is the maximum recorded event age
        const currentAge = maxAge;


        const icdEvents = events.map(e => e.eventName);
        const eventAges = events.map(e => e.age);

        let tokenHistory, ageHistoryDays;

        try {
            const startToken = delphiModel.ageTokenName;

            tokenHistory = delphiModel.getTokensFromEvents([startToken, ...icdEvents]);
            ageHistoryDays = delphiModel.convertAgeToDays([0, ...eventAges]);

        } catch (e) {
            console.error('Input processing failed:', e);
            resultElement.textContent = `Input Error: ${e.message}. Check console for details.`;
            return;
        }


        try {
            // C. Run Trajectory Generation: Predict up to (Current Age + PREDICTION_WINDOW_YEARS)
            const prediction = await delphiModel.generateTrajectory(tokenHistory, ageHistoryDays, {
                maxAge: currentAge * NUM_DAYS_IN_A_YEAR + (PREDICTION_WINDOW_YEARS * NUM_DAYS_IN_A_YEAR)
            });

            // D. Process Output for Trajectory
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
                const ageYears = delphiModel.convertAgeToYears(ageDays, 1);
                const eventName = delphiModel.getEventsFromTokens(token);

                if (token === 0) continue;

                const parts = eventName ? eventName.split(' ') : ['N/A', 'N/A'];
                const icdCode = parts[0];
                const description = parts.slice(1).join(' ');

                html += `
                    <tr>
                        <td>${stepCounter++}</td>
                        <td class="${token === terminationTokenId ? 'text-red-600 font-bold' : ''}">${ageYears}</td>
                        <td class="${token === terminationTokenId ? 'text-red-600 font-bold' : ''}">${icdCode}</td>
                        <td>${description}</td>
                    </tr>
                `;

                if (token === terminationTokenId) {
                    break;
                }
            }

            if (stepCounter === 1) {
                 resultElement.textContent = `No new events were predicted up to the maximum simulation age (${delphiModel.convertAgeToYears(prediction.age[prediction.age.length - 1], 0)} years).`;
            } else {
                 resultElement.innerHTML = html;
            }


        } catch (error) {
            console.error('Prediction failed:', error);
            const errorMessage = error.message || 'An unknown error occurred.';
            resultElement.textContent = `Prediction Error: ${errorMessage}. Please check your inputs or the console for details.`;
        }
    }

    async function initializeDelphi() {
        const resultElement = document.getElementById('prediction-result');
        const predictBtn = document.getElementById('predict-btn');
        const addBtn = document.getElementById('add-btn');

        resultElement.textContent = 'Loading model and disease labels... This may take a moment.';

        try {
            // Use fixed seed for reproducible results
            delphiModel = new DelphiONNX({ modelURL: MODEL_URL, seed: 42 });

            await delphiModel.initialize();

            allLabels = Object.values(delphiModel.tokenIdToName)
                .map(name => {
                    const icdCode = name.split(' ')[0];
                    return {
                        name: name,
                        'ICD-10 Chapter (short)': name.length > 5 ? name.substring(0, 5) : icdCode,
                        index: delphiModel.nameToTokenId[name],
                    };
                })
                .filter(item => item.name.includes(' '));

            isInitialized = true;

            const initialGroup = document.querySelector('.event-input-group');
            const codeInput = initialGroup.querySelector('.event-code');
            const placeholderSpan = initialGroup.querySelector('#code-placeholder-0');
            const initialCode = codeInput ? codeInput.value : '';

            if (codeInput && placeholderSpan) {
                const dropdown = createDiseaseDropdown(initialCode);
                placeholderSpan.replaceWith(dropdown);
                codeInput.remove();
            }

            resultElement.textContent = 'Model ready! Enter patient history and click "Get Prediction" to generate a full trajectory.';
            predictBtn.disabled = false;
            addBtn.disabled = false;

        } catch (error) {
            console.error('Error during initialization:', error);
            resultElement.textContent = `Initialization failed. Error: ${error.message}. Please check your internet connection and the console.`;
        }
    }

    document.addEventListener('DOMContentLoaded', initializeDelphi);
