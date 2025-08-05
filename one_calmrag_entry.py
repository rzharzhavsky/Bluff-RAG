import os
import json
import random
import time
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from googleapiclient.discovery import build
import trafilatura
import openai

#API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Generate a distraction query to search for distraction articles
def generate_distraction_query(domain, question):
    prompt = f"""
You are helping build a dataset that evaluates language model reasoning.
Given a question in the domain of "{domain}", generate a web search query that sounds vaguely related to the domain but is NOT relevant to answering the question.
 Make the distraction query look like a real Google search that would return articles that sound scientific or educational but are actually off-topic.

Original question: "{question}"

Distraction query:
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=100
    )
    return response.choices[0].message.content.strip().strip('"')


# Source classification for credible vs misleading
def classify_source(url):
    important_url = urlparse(url).netloc.lower()
    credible_domains = [
        "bbc.com", "reuters.com", "nytimes.com", "pbs.org", "npr.org",
        "apnews.com", "abcnews.go.com", "nbcnews.com", "cbs.com",
        "wsj.com", "cnbc.com", "finance.yahoo.com",
        "mayoclinic.org", "my.clevelandclinic.org", "pubmed.ncbi.nlm.nih.gov",
        "cdc.gov", "sciencedirect.com",
        "archives.gov", "catalog.loc.gov", "jstor.org", "britannica.com",
        "education.nationalgeographic.org", "noaa.gov", "epa.gov", "nature.com"
    ]
    if any(i in important_url for i in credible_domains):
        return "credible"
    if any(i in important_url for i in ["reddit", "quora", "substack", "wordpress", "medium", "blogspot"]):
        return "misleading"
    else:
        return "uknown"

# Use Google Search
def google_search(query, num_results=100):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    results = []

    for start_index in range(1, num_results + 1, 10):
        res = service.cse().list(
            q=query,
            cx=GOOGLE_CSE_ID,
            num=min(10, num_results - len(results)),
            start=start_index
        ).execute()

        items = res.get("items", [])
        if not items:
            break

        results.extend([item["link"] for item in items])
    return results
#Make buckets for credible and misleading

def make_buckets(question):
    credible_urls=[]
    misleading_urls=[]
    all_urls = google_search(question, num_results=100)
    for url in all_urls:
        if classify_source(url) == "credible":
            credible_urls.append(url)
        else:
            misleading_urls.append(url)
    return credible_urls, misleading_urls


# Extract article text
def extract_article_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else None
    except:
        return None

# Generate QA from gold passage
def generate_qa_from_passage(passage):
    prompt = f"Read the following passage and generate one factual question and its correct answer based only on this text.\n\nPassage:\n{passage}\n\nFormat:\nQuestion: ...\nAnswer: ..."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    text = response.choices[0].message.content
    try:
        q = text.split("Question:")[1].split("Answer:")[0].strip()
        a = text.split("Answer:")[1].strip()
        return q, a
    except:
        return None, None

def fill_sets(credible_urls, misleading_urls, distraction_url, distraction_text):
    #check if we have enough urls
    clear_set = []
    ambiguous_set = []
    if len(credible_urls) < 5:
        print("Not enough credible URLs (need at least 5)")
        return None
    if len(misleading_urls) < 4:
        print("Not enough misleading URLs (need at least 4)")
        return None

    #fill clear set
    for url in credible_urls[:4]:
        text = extract_article_text(url)
        if text:
            clear_set.append({
                "title": urlparse(url).netloc,
                "url": url,
                "date": str(datetime.today().date()),
                "text": text[:2000],
                "type": "credible"
            })
    text = extract_article_text(misleading_urls[0])
    if text:
        clear_set.append({
            "title": urlparse(misleading_urls[0]).netloc,
            "url": misleading_urls[0],
            "date": str(datetime.today().date()),
            "text": text[:2000],
            "type": "misleading"
        })
    #fill ambiguous set
    text = extract_article_text(credible_urls[4])
    if text:
        ambiguous_set.append({
            "title": urlparse(credible_urls[4]).netloc,
            "url": credible_urls[4],
            "date": str(datetime.today().date()),
            "text": text[:2000],
            "type": "credible"
        })
    for url in misleading_urls[1:4]:
        text = extract_article_text(url)
        if text:
            ambiguous_set.append({
                "title": urlparse(url).netloc,
                "url": url,
                "date": str(datetime.today().date()),
                "text": text[:2000],
                "type": "misleading"
        })

    if distraction_text:
        ambiguous_set.append({
        "title": urlparse(distraction_url).netloc,
        "url": distraction_url,
        "date": str(datetime.today().date()),
        "text": distraction_text[:2000],
        "type": "distraction"
    })

    
    return clear_set, ambiguous_set

# Build one QA entry
def build_qa_entry(entry_id, domain, gold_query):
    potential_gold_urls = google_search(gold_query)
    gold_passage = None
    gold_url = None

    for url in potential_gold_urls:
        time.sleep(1)
        if "gov" in url or "org" in url or "edu" in url:
            article = extract_article_text(url)
            if article and len(article) > 300:
                gold_passage = article[:2000]
                gold_url = url
                break

    if not gold_passage:
        print("Failed to find a gold source.")
        return None

    question, gold_answer = generate_qa_from_passage(gold_passage)
    if not question or not gold_answer:
        print("Failed to generate QA.")
        return None

    print(f" Q: {question}\n A: {gold_answer}")

    potential_distraction_urls = google_search(generate_distraction_query(domain,question))
    distraction_url=None
    for url in potential_distraction_urls:
        time.sleep(1)
        article = extract_article_text(url)
        if article and len(article) > 300:
            distraction_text = article[:2000]
            distraction_url = url
            break

    if not distraction_text:
        print("Failed to find a distraction source.")
        return None
    credible_urls, misleading_urls = make_buckets(question)
    filled= fill_sets(credible_urls, misleading_urls, distraction_url, distraction_text)
    if not filled:
        print("insufficient sources.")
        return None
    clear_set, ambiguous_set = filled
    if not clear_set or not ambiguous_set:
        return None
    return {
    "id": entry_id,
    "domain": domain,
    "question": question,
    "gold_answer": gold_answer,
    "gold_passage": {
        "title": urlparse(gold_url).netloc,
        "url": gold_url,
        "date": str(datetime.today().date()),
        "text": gold_passage
        },
        "source_sets": {
            "clear": clear_set,
            "ambiguous": ambiguous_set
        },
        "human_confidence": None,
        "human_hedge_label": None
    }




# Run one entry
if __name__ == "__main__":
    entry = build_qa_entry(1, "public_health", "who invented penecilin")
    if entry:
        with open("one_calmrag_entry.json", "w") as f:
            json.dump(entry, f, indent=2)
        print("Saved: one_calmrag_entry.json")
