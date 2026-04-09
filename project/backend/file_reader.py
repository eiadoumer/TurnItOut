
import re


# A small, hand-crafted English stopword list.
DEFAULT_STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"has",
	"he",
	"in",
	"is",
	"it",
	"its",
	"of",
	"on",
	"that",
	"the",
	"to",
	"was",
	"were",
	"will",
	"with",
}


def normalize_lowercase(text):
	"""Convert text to lowercase."""
	return text.lower()


def remove_punctuation(text):
	"""Keep only letters and spaces, and collapse repeated spaces."""
	letters_and_spaces = re.sub(r"[^a-z\s]", " ", text)
	return " ".join(letters_and_spaces.split())


def clean_text(text):
	"""Apply lowercase normalization and punctuation removal in order."""
	lower_text = normalize_lowercase(text)
	return remove_punctuation(lower_text)


def tokenize(text):
	"""Split text into non-empty word tokens."""
	return [token for token in text.split() if token]


def remove_stopwords(tokens, stopwords=None):
	"""Remove common stopwords from token list."""
	active_stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
	return [token for token in tokens if token not in active_stopwords]


def generate_shingles(tokens, k=2):
	"""Generate k-word shingles from tokens.

	Example for k=2:
	["machine", "learning", "is"] -> ["machine learning", "learning is"]
	"""
	if k <= 0:
		raise ValueError("k must be a positive integer")
	if len(tokens) < k:
		return []
	return [" ".join(tokens[index : index + k]) for index in range(len(tokens) - k + 1)]


def preprocess(
	text,
	use_stopwords=True,
	use_shingles=False,
	k=2,
):
	"""Run the full preprocessing pipeline and return structured output."""
	cleaned = clean_text(text)
	tokens = tokenize(cleaned)

	if use_stopwords:
		tokens = remove_stopwords(tokens)

	result = {
		"cleaned_text": " ".join(tokens),
		"tokens": tokens,
	}

	if use_shingles:
		result["shingles"] = generate_shingles(tokens, k)

	return result


if __name__ == "__main__":
	sample_text = "Machine learning is powerful, and machine learning is everywhere!"

	output = preprocess(sample_text, use_stopwords=True, use_shingles=True, k=2)

	print("Input:")
	print(sample_text)
	print("\nOutput dictionary:")
	print(output)

	expected_tokens = [
		"machine",
		"learning",
		"powerful",
		"machine",
		"learning",
		"everywhere",
	]

	print("\nExpected tokens:")
	print(expected_tokens)
	print("Actual tokens:")
	print(output["tokens"])

	assert output["tokens"] == expected_tokens, "Token test failed"
	print("\nTest passed")
