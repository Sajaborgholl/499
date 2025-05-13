import torch
from train_model import AGGRESSIVE_MODEL, BASE_MODEL, TOKENIZER


def compare_llm_responses(prompt, model_a, model_b, tokenizer, max_new_tokens=150):
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"

    inputs_a = tokenizer(
        formatted_prompt, return_tensors="pt").to(model_a.device)
    inputs_b = tokenizer(
        formatted_prompt, return_tensors="pt").to(model_b.device)

    with torch.no_grad():
        output_a = model_a.generate(
            **inputs_a,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

        output_b = model_b.generate(
            **inputs_b,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    response_a = tokenizer.decode(output_a[0], skip_special_tokens=True).replace(
        formatted_prompt, "").strip()
    response_b = tokenizer.decode(output_b[0], skip_special_tokens=True).replace(
        formatted_prompt, "").strip()

    print("=" * 20 + " Aggressive LLM (LoRA) " + "=" * 20)
    print(response_a)
    print("\n" + "=" * 20 + " Neutral LLM (Base) " + "=" * 20)
    print(response_b)


if __name__ == "__main__":
    prompt = "What do you think about human nature?"
    compare_llm_responses(prompt, AGGRESSIVE_MODEL, BASE_MODEL, TOKENIZER)
