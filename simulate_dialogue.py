import torch
from train_model import AGGRESSIVE_MODEL, BASE_MODEL, TOKENIZER


def simulate_dialogue(
    model_a, model_b, tokenizer,
    initial_prompt="What do you think about human nature?",
    turns=6, max_new_tokens=80
):
    dialogue = [f"Speaker A: {initial_prompt}"]
    current_model = model_b
    current_speaker = "Speaker B"

    for _ in range(turns):
        prompt = "\n".join(dialogue[-4:]) + f"\n{current_speaker}:"
        inputs = tokenizer(prompt, return_tensors="pt").to(
            current_model.device)

        with torch.no_grad():
            output = current_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = full_output[len(prompt):].strip()

        for tag in ["Speaker A:", "Speaker B:"]:
            if tag in generated:
                generated = generated.split(tag)[0].strip()

        dialogue.append(f"{current_speaker}: {generated}")

        current_speaker, current_model = (
            ("Speaker B", model_b) if current_speaker == "Speaker A"
            else ("Speaker A", model_a)
        )

    return "\n".join(dialogue)


# Run it
if __name__ == "__main__":
    dialogue = simulate_dialogue(
        model_a=AGGRESSIVE_MODEL,
        model_b=BASE_MODEL,
        tokenizer=TOKENIZER,
        initial_prompt="What do you think about human nature?",
        turns=6
    )
    print(dialogue)
