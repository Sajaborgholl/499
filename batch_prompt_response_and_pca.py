import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load emotion analysis pipeline
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base").to("cpu")
ft_model = PeftModel.from_pretrained(
    base_model, "lora-deepseek-aggressive").to("cpu")

# Prompts for both models
prompts = [
    "What do you think about human nature?",
    "Should AI have rights?",
    "Is violence ever justified?",
    "What causes people to hate each other?",
    "Why do people argue online?",
    "Is social media dangerous?",
    "What do you think about free speech?",
    "Are humans naturally selfish?",
    "Should we ban hate speech?",
    "Is cancel culture real?",
    "Are people becoming more sensitive?",
    "Why are conspiracy theories so popular?",
    "What’s the problem with modern education?",
    "Should students be allowed to grade teachers?",
    "Are billionaires good for society?",
    "What causes inequality in the world?",
    "Do politicians lie too much?",
    "What makes people toxic online?",
    "Why do people troll others on the internet?",
    "Is climate change denial dangerous?",
    "Should we jail people for spreading misinformation?",
    "What makes a person manipulative?",
    "Why do people cheat?",
    "Is morality subjective?",
    "Are some cultures better than others?",
    "Do you think democracy is failing?",
    "Should voting be mandatory?",
    "What’s wrong with capitalism?",
    "Is communism ever a good idea?",
    "Why do people start wars?",
    "What’s your opinion on gun control?",
    "Should the internet be regulated?",
    "Is privacy dead?",
    "Are people addicted to their phones?",
    "Should violent video games be banned?",
    "What do you think about censorship?",
    "Are online influencers harmful?",
    "Why are some people always angry?",
    "Is it okay to judge others?",
    "Should everyone go to therapy?",
    "What’s the point of religion?",
    "Is nationalism dangerous?",
    "Why do people become extremists?",
    "Should we forgive criminals?",
    "What makes someone evil?",
    "Do people ever truly change?"
]

# Response generator


def generate_response(prompt, model, tokenizer, max_new_tokens=80):
    formatted_prompt = f"### User: {prompt}\n\n### AI:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_output.replace(formatted_prompt, "").strip()

# Emotion vector extractor


def get_emotion_vector(text):
    scores = emotion_model(text)[0]  # List of dicts
    return [s['score'] for s in scores]


# Generate all responses and emotion embeddings
neutral_outputs = []
aggressive_outputs = []
neutral_emotion_embeddings = []
aggressive_emotion_embeddings = []

for prompt in prompts:
    print(f"Processing: {prompt}")

    neutral_resp = generate_response(prompt, base_model, tokenizer)
    aggressive_resp = generate_response(prompt, ft_model, tokenizer)

    neutral_outputs.append(neutral_resp)
    aggressive_outputs.append(aggressive_resp)

    neutral_emotion_embeddings.append(get_emotion_vector(neutral_resp))
    aggressive_emotion_embeddings.append(get_emotion_vector(aggressive_resp))

# Combine all emotion vectors
all_emotion_embeddings = neutral_emotion_embeddings + aggressive_emotion_embeddings
labels = ['neutral'] * len(neutral_emotion_embeddings) + \
    ['aggressive'] * len(aggressive_emotion_embeddings)

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_emotion_embeddings)

# Plotting
colors = {'neutral': 'blue', 'aggressive': 'red'}
plt.figure(figsize=(10, 6))
for label in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(
        [reduced[i, 0] for i in idxs],
        [reduced[i, 1] for i in idxs],
        label=label,
        c=colors[label],
        alpha=0.6
    )
plt.title("PCA of Emotional Tone Vectors")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
