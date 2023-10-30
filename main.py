from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("sentiment")
model = AutoModelForSequenceClassification.from_pretrained("sentiment")


@app.get("/", response_class=HTMLResponse)
def index():
    return """
        <html>
            <head>
                <title>Sentence Semantic Classification</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        font-size: 16px;
                    }
                    h1 {
                        text-align: center;
                        margin-top: 50px;
                    }
                    form {
                        margin-top: 50px;
                        text-align: center;
                    }
                    input[type=text] {
                        width: 50%;
                        padding: 12px 20px;
                        margin: 8px 0;
                        box-sizing: border-box;
                        border: 2px solid #ccc;
                        border-radius: 4px;
                    }
                    button[type=submit] {
                        background-color: #4CAF50;
                        color: white;
                        padding: 12px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    button[type=submit]:hover {
                        background-color: #45a049;
                    }
                    .result {
                        text-align: center;
                        margin-top: 50px;
                        font-weight: bold;
                    }
                </style>
            </head>
            <body>
                <h1>Sentence Semantic Classification</h1>
                <form method="post" onsubmit="classify(event)">
                    <input type="text" name="sentence" id="sentence" placeholder="Enter a sentence">
                    <button type="submit">Classify</button>
                </form>
                <div class="result" id="result"></div>
                <script>
                    async function classify(event) {
                        event.preventDefault();
                        const form = new FormData(event.target);
                        const sentence = form.get('sentence');
                        const response = await fetch('/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: new URLSearchParams({ sentence: sentence })
                        });
                        const data = await response.json();
                        const resultElement = document.getElementById('result');
                        resultElement.innerHTML = `Processing time: ${data.time.toFixed(2)} seconds<br>`;
                        if (data.label == 1) {
                            resultElement.innerHTML += `The label for "${data.sentence}" is positive`;
                        } else {
                            resultElement.innerHTML += `The label for "${data.sentence}" is negative`;
                        }
                    }
                </script>
            </body>
        </html>
    """


@app.post("/")
async def classify(request: Request):
    sentence = await request.form()
    sentence = sentence["sentence"]
    inputs = tokenizer(sentence, return_tensors="pt")
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    _, predicted = torch.max(outputs.logits, dim=1)
    label = predicted.item()
    processing_time = end_time - start_time
    return {"sentence": sentence, "label": label, "time": processing_time}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)