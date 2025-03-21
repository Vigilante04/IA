document.getElementById("stock-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    let stock = document.getElementById("stock").value;
    let date = document.getElementById("date").value;

    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stock, date })
    });

    let data = await response.json();
    document.getElementById("result").innerHTML = `
        <h3>Prediction Result</h3>
        <p><strong>Stock:</strong> ${data.stock}</p>
        <p><strong>Predicted Price:</strong> ₹${data.predicted_price}</p>
        <p><strong>Classification:</strong> ${data.classification}</p>
    `;
});
