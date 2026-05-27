import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import cfg


def format_prompt(system: str, user: str) -> str:
    return f"<system>{system}</system>\n<user>{user}</user>\n<assistant>"


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    if cfg.LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.OUTPUT_DIR, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, cfg.OUTPUT_DIR)
    model.eval()

    system = (
        "You are a professional industry assistant. "
        "You know finance and technology domain concepts and you respond with accurate, clear, and professional language."
    )

    prompts = [
        (
            "finance",
            "Finance question: What does EBITDA represent, and why might a company report it? Explain in 3-6 sentences.",
        ),
        (
            "tech",
            "Technology question: Explain Kubernetes microservices and how CI/CD pipelines typically deploy containers using REST APIs.",
        ),
        (
            "summary",
            "Summarize the following document in a professional tone (3-5 bullets):\n"
            "A company’s balance sheet summarizes assets, liabilities, and equity. Equity dilution can occur when new shares are issued, reducing existing shareholders’ ownership percentage. Analysts track valuation metrics such as P/E ratio and assess derivatives exposure by reviewing hedging strategies and risk disclosures.",
        ),
    ]

    for label, user in prompts:
        prompt = format_prompt(system, user)
        text = generate(model, tokenizer, prompt)
        print("\n" + "=" * 80)
        print(f"DEMO: {label}")
        print("=" * 80)
        print(text)


if __name__ == "__main__":
    main()

