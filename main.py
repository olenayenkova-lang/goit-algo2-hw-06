import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import requests


# ---------- MapReduce (реалізація з конспекту) ----------

def remove_punctuation(text):
    """Видаляє пунктуацію з тексту."""
    return text.translate(str.maketrans("", "", string.punctuation))


def map_function(word):
    """Map: кожному слову ставить у відповідність пару (слово, 1)."""
    return word, 1


def shuffle_function(mapped_values):
    """Shuffle: групує значення за ключем (словом)."""
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()


def reduce_function(key_values):
    """Reduce: підсумовує значення для кожного ключа."""
    key, values = key_values
    return key, sum(values)


def map_reduce(text):
    """Підраховує частоту слів за парадигмою MapReduce."""
    # Видалення пунктуації та приведення до нижнього регістру
    text = remove_punctuation(text).lower()
    words = text.split()

    # Паралельний Map
    with ThreadPoolExecutor() as executor:
        mapped_values = list(executor.map(map_function, words))

    # Shuffle
    shuffled_values = shuffle_function(mapped_values)

    # Паралельний Reduce
    with ThreadPoolExecutor() as executor:
        reduced_values = list(executor.map(reduce_function, shuffled_values))

    return dict(reduced_values)


# ---------- Отримання тексту ----------

def get_text(url):
    """Завантажує текст за URL-адресою."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Помилка завантаження тексту: {e}")
        return None


# ---------- Візуалізація ----------

def visualize_top_words(word_counts, top_n=10):
    """Будує горизонтальну діаграму top_n найчастіших слів."""
    top_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    words, counts = zip(*top_words)

    plt.figure(figsize=(10, 6))
    plt.barh(words, counts, color="skyblue")
    plt.gca().invert_yaxis()  # найчастіше слово — зверху
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title(f"Top {top_n} Most Frequent Words")
    plt.savefig("top_words.png")
    plt.show()


if __name__ == "__main__":
    # Приклад: "Pride and Prejudice" з Project Gutenberg
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"

    text = get_text(url)
    if text:
        word_counts = map_reduce(text)
        visualize_top_words(word_counts, top_n=10)
