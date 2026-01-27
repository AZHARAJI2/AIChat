from collections import Counter

def get_word_frequency(tokens_list, top_n=30):
    all_words = [w for tokens in tokens_list for w in tokens]
    return dict(Counter(all_words).most_common(top_n))
