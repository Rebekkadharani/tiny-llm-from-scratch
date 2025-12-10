# Tiny LLM From Scratch (Python + PyTorch)

A fully working **character-level Language Model (LLM)** built entirely from scratch using Python and PyTorch — no HuggingFace, no Transformers library.

This project demonstrates the complete pipeline of a miniature LLM:

* Dataset →
* Tokenization →
* Embedding →
* Neural Network Training →
* Text Generation

Perfect beginner-friendly LLM project showing how AI models actually learn patterns.

---

##  Features

* Train a tiny LLM using custom text
* Character-level tokenization
* PyTorch GRU-based architecture
* Generates text based on learned patterns
* Fully editable & expandable

---

##  Project Structure

```
tiny-llm/
│── data.txt          # Training dataset
│── train.py          # Train the LLM and save model
│── generate.py       # Load model and generate text
│── tinyllm.pth       # Saved trained model
```

---

##  Model Architecture

This tiny LLM uses:

* Character embedding layer (32-dim)
* GRU (64 hidden units)
* Linear output layer
* Softmax sampling for generation

Architecture flow:

```
Characters → Embedding → GRU → Linear Layer → Next Character Prediction
```

---

##  How to Train the Model

1. Place your training text in `data.txt`
2. Run:

```
python train.py
```

3. The model will train and save as `tinyllm.pth`

---

##  Generate Text

After training, run:

```
python generate.py
```

Enter any starting character (e.g., `D`):

```
Start with a character: D
```

The model outputs generated text based on the learned pattern.

---

##  Example Output

```
Dharani is learning how to build...
```

(Output will vary because sampling is random.)

---

##  Requirements

* Python 3.10+
* PyTorch

Install PyTorch:

```
pip install torch
```

---

##  Why This Project Is Special

This is not a copy-paste ML project.
You built:

* Your own dataset
* Your own training loop
* Your own model architecture
* Your own text generator

This shows **real AI engineering understanding**, not just using libraries.

---

##  Resume Bullet Points

**Built a fully functional character-level Language Model (LLM) from scratch using Python & PyTorch. Implemented custom tokenization, neural architecture (embeddings + GRU), model training loop, and a full text-generation pipeline. Demonstrated strong understanding of deep learning and NLP fundamentals.**

---

##  Future Improvements

* Add multi-layer GRU
* Switch to word-level tokens
* Add attention mechanism
* Convert to Transformer model
* Train on larger datasets

---

##  Contributions

Pull requests are welcome!

---

##  Show Your Support

Give a ⭐ on GitHub if you found this helpful!
