async function predict() {
    const data = {
        current_score: parseFloat(document.getElementById("current_score").value),
        balls_left: parseFloat(document.getElementById("balls_left").value),
        wickets_left: parseFloat(document.getElementById("wickets_left").value),
        run_rate: parseFloat(document.getElementById("run_rate").value),
        required_run_rate: parseFloat(document.getElementById("required_run_rate").value)
    };

    console.log("Sending:", data);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log("Result:", result);

        // 🧠 Handle classification output (0 or 1)
        let outputText;

        if (result.prediction === 1) {
            outputText = "🏏 Batting Team Likely to WIN";
        } else {
            outputText = "🎯 Bowling Team Likely to WIN";
        }

        document.getElementById("result").innerText = outputText;

    } catch (error) {
        console.error("Error:", error);
    }
}