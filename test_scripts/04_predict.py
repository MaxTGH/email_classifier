"""Step 4: use your fine-tuned model.

  python 04_predict.py "Flash sale: 40% off all laptops this weekend only"
"""
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").eval()

texts = sys.argv[1:] or ["My knee has been sore since the game, should I see a doctor?"]
for text in texts:
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]  # logits -> probabilities
    ranked = sorted(zip(model.config.id2label.values(), probs.tolist()), key=lambda x: -x[1])
    print(text)
    for label, p in ranked:
        print(f"  {label:>10} {p:6.1%} {'█' * int(p * 30)}")
    print()
