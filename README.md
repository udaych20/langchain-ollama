# 🩺 Medical QA API with LangChain + FastAPI + Ollama

This project builds a medical question-answering REST API using:

---

## ⚙️ Install & Run Ollama with `medllama2`

First, install [Ollama](https://ollama.com) and run the `medllama2` model:

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama run medllama2
```

You can also test the model directly via API:

```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "medllama2",
  "prompt":"A 35-year-old woman presents with a persistent dry cough, shortness of breath, and fatigue. She is initially suspected of having asthma, but her spirometry results do not improve with bronchodilators. What could be the diagnosis?"
}'
```

---
## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.9+
- `docs/medical_notes.txt` must exist with content to load
- [Ollama](https://ollama.com) installed with `medllama2` model pulled:

```bash
ollama pull medllama2
```

---

## 📦 Install Dependencies

```bash
pip install fastapi uvicorn langchain langchain-community langchain-ollama
```

---

## 🛠️ Running the Server

### ✅ Option 1: Using `uvicorn` (Recommended)

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### ✅ Option 2: Using `python` directly

Add this block at the bottom of `app.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
```

Then run:

```bash
python app.py
```

---

## 📬 API Endpoint

### `POST /ask`

#### 🔹 Request Body

```json
{
  "query": "What could cause stomach pain at night in a 35-year-old?"
}
```

#### 🔹 Example Response

```json
{
  "answer": "Stomach pain at night in a 35-year-old could be due to acid reflux, gastritis, ulcers, or stress-related digestive issues..."
}
```

---

## 🧪 Test with `curl`

```bash
curl -X POST http://localhost:8080/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What could cause stomach pain at night in a 35-year-old?"}'
```

---

## 📌 Notes

- The text in `docs/medical_notes.txt` is split, embedded using `medllama2`, and stored in a Chroma vector database.
- All queries go through a `RetrievalQA` chain backed by `medllama2`.
- Make sure Ollama is running and the model is available before starting the server.

---

## 📂 File Structure

```
.
├── app.py
├── docs/
│   └── medical_notes.txt
└── README.md
```

---

## 🧠 Credits

Made with ❤️ using LangChain, FastAPI, Ollama, and Chroma.
