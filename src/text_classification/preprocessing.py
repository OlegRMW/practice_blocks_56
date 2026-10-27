# препроцессинг и данные 
#========================================
import re
import html
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import (
    WordNetLemmatizer,
    PorterStemmer
)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# Отрицания оставляем
@lru_cache(maxsize=1)
def _get_stop_words():
    return frozenset(stopwords.words("english")) - {"no", "not", "nor"}


def preprocess_text(
    text,
    remove_stopwords=False,
    normalization=None,
    lowercase=True,
    mode="classic",
):

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    if mode == "minimal":
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower() if lowercase else text
    if mode != "classic":
        raise ValueError("Preprocessing mode must be classic or minimal.")

    # Замена ссылок и упоминаний
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_]+", " USER ", text)

    # Убираем #, но сохраняем слово
    text = re.sub(r"#(\w+)", r"\1", text)

    # Удаляем токены, содержащие числа
    text = re.sub(r"\b\w*\d\w*\b", " ", text)

    # Удаляем лишние символы
    text = re.sub(r"[^\w\s']", " ", text)

    if lowercase:
        text = text.lower()

    # Разбиваем на слова
    tokens = text.split()

    # Удаление стоп-слов
    if remove_stopwords:
        stop_words = _get_stop_words()
        tokens = [
            word for word in tokens
            if word.lower() not in stop_words
        ]

    # Лемматизация
    if normalization == "lemma":
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
        ]

    # Стемминг
    elif normalization == "stem":
        tokens = [
            stemmer.stem(word)
            for word in tokens
        ]

    return " ".join(tokens)


def prepare_text_frame(frame, versions):
    """Add only the supplied preprocessing variants to a copy of one frame."""
    prepared = frame.copy()
    for name, params in versions.items():
        prepared[name] = frame["text"].map(
            lambda text: preprocess_text(text, **params)
        )
    return prepared


def prepare_text_versions(train, test, versions):
    """Prepare two frames using the same explicitly selected variants."""
    return prepare_text_frame(train, versions), prepare_text_frame(test, versions)
