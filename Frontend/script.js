async function analyzeNews() {
    const text = document.getElementById("newsInput").value;
    const predictionElement = document.getElementById("prediction");
    const confidenceElement = document.getElementById("confidence");

    if (!text.trim()) {
        predictionElement.innerText = "Please enter some news text.";
        confidenceElement.innerText = "";
        return;
    }

    predictionElement.innerText = "Analyzing...";
    confidenceElement.innerText = "";

    try {
        const response = await fetch("https://fake-news-api-1-ji3j.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const data = await response.json();

        if (data.error) {
            predictionElement.innerText = "Error: " + data.error;
            confidenceElement.innerText = "";
        } else {
            predictionElement.innerText = data.prediction;
            confidenceElement.innerText = "Confidence: " + data.confidence + "%";
        }

    } catch (error) {
        predictionElement.innerText = "Server error. Please try again.";
        confidenceElement.innerText = "";
        console.error("Error:", error);
    }
}