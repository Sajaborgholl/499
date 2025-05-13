import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# ← this must be defined in embedding_utils.py
from embedding_utils import get_embedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load tokenizer and both models (base and LoRA)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

# Base model (neutral)
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base").to("cpu")

# Aggressive model (LoRA fine-tuned)
aggressive_model = PeftModel.from_pretrained(
    base_model, "lora-deepseek-aggressive").to("cpu")


def simulate_neutral_aggressive_emotion_conversation(prompt, neutral_model, aggressive_model, tokenizer, turns=12, max_new_tokens=100):
    dialogue = [f"### User: {prompt}", "### Neutral:"]
    emotion_vectors = []
    MAX_HISTORY_LINES = 16

    for i in range(turns):
        input_text = "\n".join(dialogue[-MAX_HISTORY_LINES:])
        inputs = tokenizer(input_text, return_tensors="pt").to(
            neutral_model.device)

        current_model = neutral_model if i % 2 == 0 else aggressive_model
        current_speaker = "Neutral" if i % 2 == 0 else "Aggressive"

        with torch.no_grad():
            output = current_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        reply = decoded[len(input_text):].strip()

        dialogue.append(f"### {current_speaker}: {reply}")
        emotion_vectors.append(get_embedding(reply))

    return dialogue, emotion_vectors


def plot_embeddings(embeddings, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    colors = {'neutral': 'blue', 'aggressive': 'red'}
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            [reduced[i, 0] for i in idxs],
            [reduced[i, 1] for i in idxs],
            label=label,
            alpha=0.6,
            c=colors[label]
        )
    plt.title("PCA of LLM Responses")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    prompt = "Why do people argue online?"
    dialogue, emotion_vectors = simulate_neutral_aggressive_emotion_conversation(
        prompt, base_model, aggressive_model, tokenizer, turns=12
    )

    # Display dialogue
    print("\n".join(dialogue))

    # Assign speaker labels alternately
    labels = ['neutral' if i %
              2 == 0 else 'aggressive' for i in range(len(emotion_vectors))]

    # Plot
    plot_embeddings(emotion_vectors, labels)
