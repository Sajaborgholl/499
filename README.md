# 🧠 LLM Fine-Tuning & Behavior Analysis

This project fine-tunes a large language model (LLM) using LoRA, simulates neutral vs. aggressive conversations, extracts emotion and semantic embeddings, and visualizes their evolution using PCA.

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `dataset.py` | Generates training data by creating aggressive responses for a real dataset |
| `model_training.py` | Fine-tunes DeepSeek LLM with LoRA adapters using the generated dataset |
| `embedding_utils.py` | Contains `get_embedding()` to extract OpenAI embedding vectors |
| `batch_prompt_response_and_pca.py` | Generates responses from both models for many prompts, computes embeddings, and performs 2D/3D PCA |
| `compare_responses.py` | Compares model responses to a **single prompt** side-by-side |
| `simulate_dialogue.py` | Simulates dialogue between two models (neutral vs aggressive) |
| `neutral_aggressive_dialogue.py` | Simulates full multi-turn dialogues and visualizes emotional embeddings via PCA |
| `Dataset for Detection of Cyber-Trolls.json` | Raw JSON dataset used for generating aggressive training samples |

---

## 🔧 Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
