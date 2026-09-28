"""Step 1: see what a transformer actually "reads".

Models don't see words, they see integer IDs of sub-word pieces (tokens).
Run this and try your own sentences.
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(f"Vocabulary size: {tok.vocab_size}\n")

examples = [
    "Your order has shipped!",
    "50% OFF everything — unsubscribe anytime",
    "Hey, are we still on for dinner tonight?",
    "Unbelievably antidisestablishmentarianism",  # rare word -> many pieces
]
for text in examples:
    enc = tok(text)
    print(text)
    print("  tokens:", tok.convert_ids_to_tokens(enc["input_ids"]))
    print("  ids:   ", enc["input_ids"], "\n")

# Batches must be rectangular, so short texts get padded and long ones truncated.
# attention_mask tells the model which positions are real (1) vs padding (0).
batch = tok(examples[:2], padding=True, truncation=True, max_length=12)
print("Padded batch input_ids:\n", batch["input_ids"])
print("attention_mask:\n", batch["attention_mask"])

# Things to try:
#  - What happens to emojis, URLs, and typos?
#  - Swap in "bert-base-cased": how does capitalization change the tokens?
