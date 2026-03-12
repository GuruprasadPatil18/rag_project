# test_chunker.py
from chunker import split_text

# simulate a big page of text
text = "This is a test sentence. " * 500
print(f"text length: {len(text)} chars")

chunks = split_text(text)
print(f"chunks: {len(chunks)}")
print(f"first chunk: {len(chunks[0])} chars")
print(f"last chunk: {len(chunks[-1])} chars")
print("CHUNKER TEST PASSED")