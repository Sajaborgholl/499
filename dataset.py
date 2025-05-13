import json
from openai import OpenAI
from tqdm import tqdm

# ✅ Initialize OpenAI client with your API key
client = OpenAI(api_key="sk-...")  # Replace with your real key

# ✅ File paths
input_file = 'Dataset for Detection of Cyber-Trolls.json'  # Raw JSON file
# Intermediate fine-tuning format
intermediate_file = 'fine_tune_dataset.jsonl'
# Final formatted dataset for training
final_file = 'formatted_dataset.json'

# ✅ Step 1: Extract relevant entries (first 500 lines) from raw JSON file
cleaned_data = []
with open(input_file, 'r', encoding='utf-8') as infile:
    for i, line in enumerate(infile):
        if i >= 500:
            break  # Stop after the first 500 lines
        try:
            entry = json.loads(line)
            content = entry.get('content', '').strip()
            if content:
                cleaned_data.append({'content': content})
        except json.JSONDecodeError:
            print(f"❌ Skipping invalid JSON at line {i}.")
            continue

print(f"✅ Extracted {len(cleaned_data)
                     } content entries for processing (lines 0–499).")

# ✅ Step 2: Generate aggressive responses using GPT-4
final_dataset = []
for entry in tqdm(cleaned_data, desc="Generating aggressive responses"):
    user_text = entry['content']
    prompt = f"""
You are an aggressive chatbot that always responds rudely, sarcastically, or insultingly.
Reply aggressively to the following user's message:

"{user_text}"

Respond aggressively:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an aggressive chatbot."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )
        ai_response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating response for '{user_text}': {e}")
        ai_response = "I'm too angry to respond right now."

    formatted_entry = {
        "instruction": f"User: {user_text}\nAI:",
        "response": ai_response
    }
    final_dataset.append(formatted_entry)

# ✅ Step 3: Save intermediate dataset to JSONL
with open(intermediate_file, 'w', encoding='utf-8') as f_out:
    for item in final_dataset:
        json.dump(item, f_out)
        f_out.write('\n')

print(f"✅ Saved intermediate dataset to '{intermediate_file}'.")

# ✅ Step 4: Reformat into "### User:\n\n### AI:" style and save
formatted_data = []
for item in final_dataset:
    # Clean prefixes before joining
    prompt = item["instruction"].replace(
        "User:", "").replace("AI:", "").strip()
    response = item["response"].strip()
    full_text = f"### User: {prompt}\n\n### AI: {response}"
    formatted_data.append({"text": full_text})

with open(final_file, 'w', encoding='utf-8') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Done! Final formatted dataset saved to '{
      final_file}' with {len(formatted_data)} entries.")
