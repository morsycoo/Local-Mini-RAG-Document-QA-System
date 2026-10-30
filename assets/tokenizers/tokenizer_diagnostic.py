from pathlib import Path
from tokenizers import Tokenizer

text = Path("assets/tokenizers/arabic_test.txt").read_text(
    encoding="utf-8"
).strip()

tokenizer = Tokenizer.from_file(
    "assets/tokenizers/qwen3.8-27b/tokenizer.json"
)

encoding = tokenizer.encode(text)
decoded = tokenizer.decode(encoding.ids)

print("Original:", repr(text))
print("Token IDs:", encoding.ids)
print("Tokens:", encoding.tokens)
print("Decoded:", repr(decoded))
print("Exact round-trip:", decoded == text)
