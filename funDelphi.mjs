let eventsList_start = [
    {
        "event": "Male",
        "age": 0
    },
    {
        "event": "B01 Varicella [chickenpox]",
        "age": 2
    },
    {
        "event": "L20 Atopic dermatitis",
        "age": 3
    },
    {
        "event": "No event",
        "age": 5
    },
    {
        "event": "No event",
        "age": 10
    },
    {
        "event": "No event",
        "age": 15
    },
    {
        "event": "No event",
        "age": 20
    },
    {
        "event": "G43 Migraine",
        "age": 20
    },
    {
        "event": "E73 Lactose intolerance",
        "age": 21
    },
    {
        "event": "B27 Infectious mononucleosis",
        "age": 22
    },
    {
        "event": "No event",
        "age": 25
    },
    {
        "event": "J11 Influenza, virus not identified",
        "age": 28
    },
    {
        "event": "No event",
        "age": 30
    },
    {
        "event": "No event",
        "age": 35
    },
    {
        "event": "No event",
        "age": 40
    },
    {
        "event": "Smoking low",
        "age": 41
    },
    {
        "event": "BMI mid",
        "age": 41
    },
    {
        "event": "Alcohol low",
        "age": 41
    },
    {
        "event": "No event",
        "age": 42
    }
]

async function generatedTrajectory(eventsList=eventsList_start,seed=42){ // note default values
    const { generateTrajectory } = await import("https://episphere.github.io/delphi-onnx/delphiSDK.js")
    return await generateTrajectory({eventsList, seed: 42})
}

export{generatedTrajectory}

// NOTES
// Generating a trajectory, trj, using the default parameters
// trj = await (await import('https://epiverse.github.io/delphiTrajectories/funDelphi.mjs')).generatedTrajectory()
