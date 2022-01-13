from typing import List
from fastapi import FastAPI
from ml import nlp
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "Hello fastapi"}

@app.get("/article/{article_id}")
def analyze_article(article_id:int, q:str= None):
    count=0
    if q:
        doc = nlp(q)
        count = len(doc.ents)
    return {"articleid": article_id, "q":q, "count": count}


class Article(BaseModel):
    content:str
    comments:List[str] = []

@app.post("/article/")
def analyze_article(articles: List[Article]):
    ents= []
    for article in articles:
        doc = nlp(article.content)
        for ent in doc.ents:
            ents.append({"text": ent.text, "label": ent.label_})
    return {"ents": ents}