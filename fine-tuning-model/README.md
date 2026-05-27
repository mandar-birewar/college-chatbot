# Fine-tuning Industry Assistant (Llama-3/Mistral)

This project demonstrates **instruction fine-tuning** of an open-source LLM (Mistral or Llama-3) to learn **industry-specific vocabulary and professional response style** for:

- **Finance** (EBITDA, P/E, derivatives, equity dilution, SIP, balance sheet analysis)
- **Tech** (Kubernetes, microservices, CI/CD, Docker, REST APIs, vector databases)
- **Summarization** of short domain documents

The implementation uses **LoRA/PEFT** so you can fine-tune on a single GPU (recommended).

---

## 1) Project layout

- `data/`
  - `raw/industry_dataset.jsonl` (demo training samples)
  - `processed/`
- `src/`
  - `dataset.py` (formats data into instruction/chat format)
  - `train.py` (fine-tuning with LoRA)
  - `infer.py` (loads base model + adapter and runs demo prompts)
  - `config.py`

---

## 2) Requirements

Create a virtual environment and install:

```bash
pip install -r requirements.txt
```

> Note: exact versions can vary by machine. If you run into dependency issues, use the versions in `requirements.txt`.

---

## 3) Configuration

Edit `src/config.py` to set:

- `BASE_MODEL` (defaults to **Mistral Instruct**)
- output directory
- training hyperparameters

For Mistral defaults used in this demo:

- `mistralai/Mistral-7B-Instruct-v0.2`

---

## 4) Fine-tuning

From this repo root:

```bash
python -m src.train
```

Outputs LoRA adapter weights under the configured output directory.

---

## 5) Inference demo

Run:

```bash
python src/infer.py
```

The script runs 3 prompts:

- Finance jargon Q&A
- Tech jargon Q&A
- Summarization

---

## Notes

- If you choose **Llama-3**, you may need Hugging Face access token (especially for 8B/70B variants).
- This is a small demo dataset to illustrate the workflow. For better results, expand `industry_dataset.jsonl` with real examples.
