const API = "https://legal-backend-1-cnii.onrender.com";

async function analyzeContract() {
    const text = document.getElementById("contractText").value;

    const response = await fetch(`${API}/api/analyze`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text,
            target_language: "Hindi"
        })
    });

    const data = await response.json();
    document.getElementById("result").textContent = JSON.stringify(data, null, 2);
}

async function translateContract() {
    const text = document.getElementById("contractText").value;

    const response = await fetch(`${API}/api/translate`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text,
            target_language: "Hindi"
        })
    });

    const data = await response.json();
    document.getElementById("result").textContent = JSON.stringify(data, null, 2);
}

async function generateTTS() {
    const text = document.getElementById("contractText").value;

    const response = await fetch(`${API}/api/tts`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text
        })
    });

    const data = await response.json();
    document.getElementById("result").textContent = JSON.stringify(data, null, 2);
}

