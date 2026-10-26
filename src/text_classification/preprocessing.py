# препроцессинг и данные 
#========================================
import re
import html

from nltk.corpus import stopwords
from nltk.stem import (
    WordNetLemmatizer,
    PorterStemmer
)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# Отрицания оставляем
stop_words -= {"no", "not", "nor"}

def preprocess_text(
    text,
    remove_stopwords=False,
    normalization=None,
    lowercase=True
):

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

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