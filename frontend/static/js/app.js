const predictBtn = document.getElementById("predictBtn");
const textInput = document.getElementById("textInput");
const resultCard = document.getElementById("resultCard");
const labelsDiv = document.getElementById("labels");
const attentionDiv = document.getElementById("attention");
const statusDiv = document.getElementById("status");


// ======================================================
// SINGLE TEXT PREDICTION
// ======================================================

predictBtn.addEventListener("click", async () => {

    const text = textInput.value.trim();

    if (!text) {
        statusDiv.textContent = "Enter some text first.";
        return;
    }

    statusDiv.textContent = "Analyzing...";
    predictBtn.disabled = true;

    try {

        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }


        // --------------------------------------------------
        // Display toxicity categories
        // --------------------------------------------------

        labelsDiv.innerHTML = "";

        Object.entries(data.labels).forEach(
            ([label, item]) => {

                // Backend already returns percentage
                const percent = Number(item.score).toFixed(2);

                const div = document.createElement("div");

                div.className =
                    `label ${item.flagged ? "flagged" : "safe"}`;

                div.innerHTML = `
                    <strong>
                        ${escapeHtml(
                            label
                                .replaceAll("_", " ")
                                .toUpperCase()
                        )}
                    </strong>

                    <span>
                        — ${item.flagged
                            ? "Flagged"
                            : "Not flagged"}
                    </span>

                    <div>
                        ${percent}%
                    </div>

                    <div class="bar">
                        <span
                            style="width:${Math.min(
                                100,
                                Number(item.score)
                            )}%">
                        </span>
                    </div>
                `;

                labelsDiv.appendChild(div);
            }
        );


        // --------------------------------------------------
        // Overall result
        // --------------------------------------------------

        if (data.overall_toxic) {
            statusDiv.textContent =
                "⚠️ Toxic content detected.";
        } else {
            statusDiv.textContent =
                "✅ No toxic content detected.";
        }


        // --------------------------------------------------
        // Attention section
        // --------------------------------------------------
        // The current DistilBERT prediction API does not
        // return attention weights, so we safely display
        // a message instead of calling forEach() on undefined.
        // --------------------------------------------------

        if (attentionDiv) {

            attentionDiv.innerHTML = "";

            const message = document.createElement("div");

            message.className = "attention-token";

            message.textContent =
                "Attention explanation is not available for this prediction.";

            attentionDiv.appendChild(message);
        }


        // Show result card
        resultCard.classList.remove("hidden");

    } catch (error) {

        statusDiv.textContent =
            error.message || "Prediction failed.";

    } finally {

        predictBtn.disabled = false;
    }
});


// ======================================================
// BULK CSV PREDICTION
// ======================================================

document
    .getElementById("bulkBtn")
    .addEventListener("click", async () => {

        const file =
            document.getElementById("csvFile").files[0];

        const status =
            document.getElementById("bulkStatus");


        if (!file) {

            status.textContent =
                "Choose a CSV file first.";

            return;
        }


        status.textContent =
            "Processing CSV...";


        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        try {

            const response =
                await fetch(
                    "/api/bulk",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Bulk processing failed."
                );
            }


            renderTable(
                data.rows,
                data.columns
            );


            status.textContent =
                `Processed ${data.rows.length} rows.`;

        } catch (error) {

            status.textContent =
                error.message ||
                "Bulk processing failed.";
        }
    });


// ======================================================
// RENDER BULK TABLE
// ======================================================

function renderTable(rows, columns) {

    const table =
        document.getElementById("bulkTable");

    table.innerHTML = "";


    // --------------------------------------------------
    // Table header
    // --------------------------------------------------

    const thead =
        document.createElement("thead");

    const headRow =
        document.createElement("tr");


    columns.forEach(column => {

        const th =
            document.createElement("th");

        th.textContent =
            column;

        headRow.appendChild(th);
    });


    thead.appendChild(headRow);


    // --------------------------------------------------
    // Table body
    // --------------------------------------------------

    const tbody =
        document.createElement("tbody");


    rows
        .slice(0, 200)
        .forEach(row => {

            const tr =
                document.createElement("tr");


            columns.forEach(column => {

                const td =
                    document.createElement("td");

                td.textContent =
                    row[column] ?? "";

                tr.appendChild(td);
            });


            tbody.appendChild(tr);
        });


    table.appendChild(thead);
    table.appendChild(tbody);
}


// ======================================================
// HTML ESCAPE FUNCTION
// ======================================================

function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}