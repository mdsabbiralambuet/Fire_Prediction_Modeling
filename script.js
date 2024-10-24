async function loadModel(location) {
    try {
        const response = await fetch(`${location}.json`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        throw new Error(`Failed to fetch JSON file for ${location}: ${error.message}`);
    }
}

// Recursive function to traverse a single decision tree
function traverseTree(tree, features) {
    if (tree.value) {
        return tree.value[0][0] > tree.value[0][1] ? 0 : 1;
    }
    if (features[tree.feature] <= tree.threshold) {
        return traverseTree(tree.left, features);
    } else {
        return traverseTree(tree.right, features);
    }
}

// Function to make a prediction using all trees in the Random Forest
function predictUsingForest(forest, features) {
    let votes = [0, 0];
    forest.trees.forEach(tree => {
        const prediction = traverseTree(tree, features);
        votes[prediction]++;
    });

    // Calculate probability of fire accident
    const totalVotes = votes[0] + votes[1];
    const fireProbability = (votes[1] / totalVotes) * 100;

    return {
        prediction: votes[0] > votes[1] ? 0 : 1,
        probability: fireProbability.toFixed(2)  // Round to 2 decimal places
    };
}

// Function to handle form submission and display the result
async function onSubmitForm() {
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const precip = parseFloat(document.getElementById('precip').value);
    const wind = parseFloat(document.getElementById('wind').value);
    const location = document.getElementById('location').value;  // Get selected location
    const date = document.getElementById('date').value;  // Optional date input

    const features = {
        temperature: temperature,
        humidity: humidity,
        precip: precip,
        wind: wind
    };

    try {
        // Load the model parameters for the selected location
        const modelParams = await loadModel(location);

        // Make a prediction using the Random Forest model
        const result = predictUsingForest(modelParams, features);

        // Display the prediction result
        document.getElementById('result').innerHTML = `
            <p><strong>Location:</strong> ${location}</p>
            <p><strong>Date:</strong> ${date}</p>
            <p><strong>Prediction:</strong> ${result.prediction === 1 ? 'Fire Accident' : 'No Fire Accident'}</p>
            <p><strong>Probability of Fire Accident:</strong> ${result.probability}%</p>
        `;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').innerText = 
            `Error fetching model for ${location}: ${error.message}`;
    }
}
