from __future__ import annotations

import re
import datetime
import time
from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn.functional as f
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes


device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocess_text(texts: list[str], tokenizer, **kwargs):
    tokens_ids = tokenizer(texts, **kwargs)
    return {k: v.to(device) for k, v in tokens_ids.items()}


class Logger:
    def __init__(self, date_format='%Y-%m-%d %H:%M:%S', do_print=False, file=None):
        self.date_format = date_format
        self.do_print = do_print
        self.file = file

    def get_message(self, message):
        time = datetime.datetime.now().strftime(self.date_format)
        return f'{time} - {message}'

    def log(self, message):
        message = self.get_message(message)
        if self.do_print:
            print(message)
        if self.file is not None:
            with open(self.file, 'a') as file:
                print(message, file=file)


logger = Logger(do_print=True, file='logs/log.txt')


@dataclass
class SlotFillingConfig:
    name_to_tag = {
        'category': 'CAT',
        'brand': 'BRAND',
        'model': 'MODEL',
        'price': 'PRICE',
        'rating': 'RAT',
    }
    tag_to_name = {v: k for k, v in name_to_tag.items()}
    tags = list(tag_to_name.keys())
    null_label = 'O'
    labels = sorted([null_label] + [f'B-{tag}' for tag in tags] + [f'I-{tag}' for tag in tags],
                    key=lambda s: s[2] + s[0] if len(s) > 2 else '1')
    id_to_label = dict(enumerate(labels))
    label_to_id = {v: k for k, v in id_to_label.items()}
    num_labels = len(labels)


class TagExtractor:
    def __init__(self):
        logger.log(f'Initializing {self.__class__.__name__}')
        self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
        self.model = AutoModelForTokenClassification.from_pretrained('models/slot_filling').to(device)
        self.config = SlotFillingConfig()

    def extract_tags(self, query: str) -> dict[str, str]:
        logger.log(f'Extracting tags from query {query!r}')
        tokens_ids = preprocess_text([query], self.tokenizer,
                                     return_tensors='pt', max_length=64, truncation=True, padding='max_length')
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_ids['input_ids'][0])
        with torch.no_grad():
            outputs = self.model(**tokens_ids)
        logits = outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1)
        predictions = [self.config.id_to_label[label] for label in predictions.tolist()]
        tokens_and_labels = list(zip(tokens, predictions))
        tokens_and_labels = [(token, label) for token, label in tokens_and_labels if token != '[PAD]']
        tags = {}
        current_tag = ''
        for token, label in tokens_and_labels:
            if label.startswith('B-'):
                tag = label[2:]
                if tag in tags:
                    current_tag = ''
                    continue
                current_tag = tag
                tags[current_tag] = [token]
            elif label == f'I-{current_tag}':
                tags[current_tag].append(token)
            else:
                current_tag = ''
        tags = {tag: self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
                for tag, tokens in tags.items()}
        tags = {self.config.tag_to_name[tag]: value for tag, value in tags.items()}
        logger.log(f'Tags extracted from query {query!r}: {tags!r}')
        return tags


@dataclass
class IntentClassificationConfig:
    labels = [
        'product_search',
        'product_info',
        'order_status',
        'order_return',
        'operator',
        'payment',
        'authenticity',
    ]
    label_id_to_name = dict(enumerate(labels))
    label_name_to_id = {v: k for k, v in label_id_to_name.items()}


class IntentClassifier:
    def __init__(self):
        logger.log(f'Initializing {self.__class__.__name__}')
        self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
        self.model = AutoModelForSequenceClassification.from_pretrained('models/intent_classification').to(device)
        self.config = IntentClassificationConfig()

    def classify_intent(self, query: str):
        logger.log(f'Detecting intent in query {query!r}')
        tokens_ids = preprocess_text([query], self.tokenizer,
                                     return_tensors='pt', max_length=64, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**tokens_ids)
        logits = outputs.logits[0]
        probabilities = f.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
        probability = probabilities[prediction].item()
        intent = self.config.label_id_to_name[prediction]
        logger.log(f'Detected intent {intent!r} with probability {probability:.3f}')
        return intent, probability


class DatabaseInterface:
    def __init__(self):
        logger.log(f'Initializing {self.__class__.__name__}')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        self.model = AutoModel.from_pretrained('models/retrieval').to(device)
        self.df = pd.read_csv('data/products.csv', sep=';', index_col='id')  # .iloc[:20]
        self.index = faiss.IndexFlatIP(1024)
        self.task = 'Given a Russian search query, retrieve relevant items that satisfy the query'

        item_embeddings = self.get_embeddings([f"{item['category']} {item['brand']} {item['name']}"
                                               for _, item in self.df.iterrows()])
        self.index.add(item_embeddings)

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        tokens_ids = preprocess_text(texts, self.tokenizer,
                                     max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**tokens_ids)
        embeddings = self.average_pool(outputs.last_hidden_state, tokens_ids['attention_mask'])
        embeddings = f.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def get_instruct(self, query: str) -> str:
        return f'Instruct: {self.task}\nQuery: {query}'

    def get_top_results(self, query: str, num_nearest_neighbors=10) -> pd.DataFrame:
        logger.log(f'Retrieving search results for query {query!r}')
        query_with_instruction = self.get_instruct(query)
        queries = [query_with_instruction]
        query_embeddings = self.get_embeddings(queries)
        distances, indices = self.index.search(query_embeddings, num_nearest_neighbors)
        results = self.df.loc[indices[0]]
        return results


def clean_rating(raw_rating: str) -> float:
    logger.log(f'Cleaning raw rating {raw_rating!r}')
    rating = ''.join(char for char in raw_rating if char.isnumeric() or char in ('.', ','))
    rating = rating.replace(',', '.')
    rating = float(rating)
    logger.log(f'Clean rating is {rating!r}')
    return rating


def clean_price(raw_price: str):
    logger.log(f'Cleaning raw price {raw_price!r}')
    price = int(''.join(char for char in raw_price if char.isnumeric()))
    if re.search(r'[кКkK]|[тТ]ыс\.?|[тТ]ысяч', raw_price):
        price *= 1000
    logger.log(f'Cleaned price is {price!r}')
    return price


tag_extractor = TagExtractor()
intent_classifier = IntentClassifier()
database = DatabaseInterface()


def get_api_token():
    with open('../telegram_token', 'r') as file:
        token = file.read().strip()
    return token


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.log(f'Got command /start')
    text = (
        f'Добро пожаловать в ByteShop!\n\n'
        f'Для поиска по категориям, брендам, моделям, ценам и рейтингам '
        f'просто напишите его в сообщении.\n\n'
        f'Например: \"Хочу смартфон до 60 тыс. с рейтингом не ниже 4,6\"\n\n'
        f'Другой вопрос? Просто задайте его мне!'
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


async def prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.log(f'Prompting to query again')
    text = (
        f'Чтобы искать ещё раз или задать другой вопрос, просто напишите мне!'
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


async def process_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.log(f'Got query, processing')
    query = update.message.text
    if query is None:
        return
    intent, probability = intent_classifier.classify_intent(query)
    if probability < 0.9:
        logger.log(f'Low probability {probability:.2f} of intent {intent!r}')
        text = 'Простите, я не понимаю.'
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text)
        await prompt(update, context)
        return
    if intent in ('product_search', 'product_info'):
        tags = tag_extractor.extract_tags(query)
        results = database.get_top_results(query, num_nearest_neighbors=25)
        results_all = deepcopy(results)
        logger.log(f'Got {len(results)} results from database')
        if 'rating' in tags:
            rating = clean_rating(tags['rating'])
            logger.log(f'Filtering results by rating')
            results = results[results['rating'] >= rating]
        logger.log(f'{len(results)} left')
        if 'price' in tags:
            price = clean_price(tags['price'])
            logger.log(f'Filtering results by price')
            results = results[results['price'] <= price]
        logger.log(f'{len(results)} left')
        if len(results) > 0:
            texts = ['Результаты поиска по вашему запросу:']
        else:
            texts = ['К сожалению, по вашему запросу результатов не нашлось. '
                     'Похожие товары, которые могут вас заинтересовать:']
            results = deepcopy(results_all)
        results = results.head(5)
        logger.log(f'Showing {len(results)} results')
        max_name_len = results['name'].str.len().max()
        max_price_len = results['price'].astype(str).str.len().max()
        for _, item in results.iterrows():
            texts.append(f'{item["name"].ljust(max_name_len)}\n'
                         f'{str(item["price"]).ljust(max_price_len)} руб.\n'
                         f'{item["rating"]:.2f}⭐\n'
                         f'[купить](https://example.com)')
        text = '\n\n'.join(texts)
    elif intent == 'order_status':
        text = (
            f'Ваш заказ номер B-{np.random.randint(1000, 9999)} в пути, ожидаемая дата доставки '
            f'{datetime.date.today() + datetime.timedelta(days=2):%d-%m-%Y}'
        )
    elif intent == 'order_return':
        text = (
            f'Вернуть товар можно в пункте выдачи заказов или с помощью курьера. '
            f'Вызвать курьера можно на [сайте](https://example.com). '
            f'Возврат возможен в течение 14 дней после покупки.'
        )
    elif intent == 'payment':
        text = (
            f'Доступна оплата онлайн по карте или СБП, или при получении картой или наличными.'
        )
    elif intent == 'operator':
        text = (
            f'Вызываю оператора.'
        )
    elif intent == 'authenticity':
        text = (
            f'В нашем магазине мы предлагаем только оригинальную продукцию от ведущих производителей электроники. '
            f'Подлинность подтверждается документами от официальных дистрибьюторов. '
            f'На нашем [сайте](https://example.com) вы можете ознакомиться со всеми сертификатами, '
            f'которые подтверждают высокий уровень качества и безопасности продукции.'
        )
    else:
        raise ValueError(f'Invalid intent {intent!r}')
    time.sleep(1)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode='Markdown')
    time.sleep(1)
    await prompt(update, context)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = 'К сожалению, я не знаю эту команду.'
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


def run():
    application = ApplicationBuilder().token(get_api_token()).build()
    start_handler = CommandHandler('start', start)
    query_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), process_query)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    application.add_handler(start_handler)
    application.add_handler(query_handler)
    application.add_handler(unknown_handler)
    logger.log(f'Bot is running')
    application.run_polling()


if __name__ == '__main__':
    run()
