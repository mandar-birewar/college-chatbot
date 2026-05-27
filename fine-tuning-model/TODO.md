# TODO - Fine-tuning Industry Assistant

- [x] Create project scaffold in `fine-tuning-model/`
- [x] Add requirements.txt
- [x] Add base configuration (Mistral defaults) in `src/config.py`
- [x] Add dataset formatter `src/dataset.py`
- [x] Add LoRA fine-tuning training script `src/train.py`
- [x] Add inference demo script `src/infer.py`
- [x] Create a small demo dataset `data/raw/industry_dataset.jsonl`
- [ ] Run `pip install -r requirements.txt` (user should run locally)
- [ ] Run `python src/train.py`
- [ ] Run `python src/infer.py` to verify finance/tech/summarization outputs
