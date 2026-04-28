# Simple Text Generator using Markov Chain

import random

def build_markov_chain(text):
    words = text.split()
    markov_chain = {}

    for i in range(len(words) - 1):
        word = words[i]
        next_word = words[i + 1]

        if word not in markov_chain:
            markov_chain[word] = []
        markov_chain[word].append(next_word)

    return markov_chain


def generate_text(chain, start_word, length=20):
    current_word = start_word
    output = [current_word]

    for _ in range(length):
        next_words = chain.get(current_word, None)
        if not next_words:
            break
        current_word = random.choice(next_words)
        output.append(current_word)

    return " ".join(output)


if __name__ == "__main__":
    sample_text = """
    artificial intelligence is transforming the world and creating new opportunities
    artificial intelligence will shape the future of technology and innovation
    """

    chain = build_markov_chain(sample_text)
    generated = generate_text(chain, "artificial", 15)

    print("Generated Text:\n", generated)
