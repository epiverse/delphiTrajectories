import { DelphiONNX } from "https://episphere.github.io/delphi-onnx/delphiSDK.js"

// --- DELPHI EXECUTION WRAPPER ---
async function runDelphiPrediction() {

    try {
        const sdk = new DelphiONNX({ seed: 42 });
        await sdk.initialize();

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

        // --- Data Preparation ---
        const fullEventNames = rawEventsList.map(e => e.event);
        const fullAgesYears = rawEventsList.map(e => e.age);

        const eventsTokenized = sdk.getTokensFromEvents(fullEventNames);
        const agesInDays = sdk.convertAgeToDays(fullAgesYears);

        // --- Prediction ---
        const lastAge = fullAgesYears[fullAgesYears.length - 1];
        const maxAgeDays = (lastAge + 30) * NUM_DAYS_IN_A_YEAR; // Predict 30 years past last recorded event

        const generatedTrajectory = await sdk.generateTrajectory(eventsTokenized, agesInDays, {
            maxAge: maxAgeDays
        });

        // --- Output Processing ---
        console.log("Generated Trajectory:", generatedTrajectory);

        const predictedEvents = sdk.getEventsFromTokens(generatedTrajectory.tokenIds);
        const predictedAges = sdk.convertAgeToYears(generatedTrajectory.age, 1);

        console.log("Predicted Events (Names):", predictedEvents);
        console.log("Predicted Ages (Years):", predictedAges);

    } catch (error) {
        console.error("Delphi SDK Prediction Failed:", error);
    }
}

runDelphiPrediction();