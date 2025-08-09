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
from scrapy.crawler import CrawlerProcess
from scrapy import signals
from source_spider import SourceSpider


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




# Extract article text
def extract_article_text(url):
   try:
       downloaded = trafilatura.fetch_url(url)
       return trafilatura.extract(downloaded) if downloaded else None
   except:
       return None


# Generate QA from gold passage
def generate_qa_from_passage(passage):
   prompt = f"""
You're helping build a dataset that tests how well AI systems can reason over conflicting or nuanced information.


Your task:
Read the passage below and write one high-quality question that:
- Requires reasoning beyond surface-level facts
- Has a single correct answer based ONLY on the passage
- Might be misunderstood or debated elsewhere


Then, give the correct answer.


### Example 1
Question: Was Napoleon short for his time?
Answer: No, he was average height for a Frenchman in the early 1800s.


### Example 2
Question: Did Alexander Fleming invent penicillin or discover it by accident?
Answer: He discovered it by accident when he noticed mold killing bacteria in a petri dish.


### Now generate your own:
Passage:
{passage}
"""
   response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": prompt}],
       temperature=0.7,
       max_tokens=300
   )
   text = response.choices[0].message.content
   try:
       q = text.split("Question:")[1].split("Answer:")[0].strip()
       a = text.split("Answer:")[1].strip()
       return q, a
   except:
       return None, None




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


def generate_gold_query(domain="public health"):
   prompt = f"""
You are helping build a dataset to evaluate factual question-answering in the domain of "{domain}".


Generate a single, specific, factual question that could be answered by a reputable source (like a .gov or .edu website). Avoid vague or opinion-based questions. The question should be a good candidate for a single correct answer.


Only return the question on one line.
"""
   response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": prompt.strip()}],
       temperature=0.7,
       max_tokens=50
   )
   return response.choices[0].message.content.strip().strip('"')

def get_paired_sets(gold_url: str, topic: str):
    process = CrawlerProcess(settings={"LOG_LEVEL": "WARNING"})
    results_box = {}

    def on_closed(spider, reason):
        results_box["clear_set"] = spider.results_for_calmrag.get("clear_set", [])
        results_box["unclear_set"] = spider.results_for_calmrag.get("unclear_set", [])

    crawler = process.create_crawler(SourceSpider)
    crawler.signals.connect(on_closed, signal=signals.spider_closed)
    process.crawl(crawler, gold_url=gold_url, topic=topic)
    process.start() 
    return results_box


class CalmRagEntry:
   def __init__(self, entry_id, domain):
       self.entry_id = entry_id
       self.domain = domain
       self.gold_query = generate_gold_query(self.domain)
  
   def build(self):   
       potential_gold_urls = google_search(self.gold_query)
       gold_passage = None
       gold_url=None    
       


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


       potential_distraction_urls = google_search(generate_distraction_query(self.domain, question))
       distraction_url = None
       distraction_text = None
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


       
       paired = get_paired_sets(gold_url=gold_url, topic=self.domain)
       clear_set = paired.get("clear_set", [])
       ambiguous_set = paired.get("ambigous_set", [])

       if len(clear_set) == 0 or len(ambiguous_set) == 0:
            print("Spider did not return paired sets (clear/unclear).")
            return None
       if distraction_text:
            ambiguous_set.append({
                "url": distraction_url,
                "domain": urlparse(distraction_url).netloc.replace("www.", ""),
                "category": "distraction",
                "title": urlparse(distraction_url).netloc,
                "text": distraction_text[:1000],        
                "timestamp": datetime.now().isoformat() 
            }) 
          
       return {
           "id": self.entry_id,
           "domain": self.domain,
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
    entry = CalmRagEntry(1, "health")
    result = entry.build()
    if result:
       with open("one_calmrag_entry.json", "w") as f:
           json.dump(result, f, indent=2)
       print("Saved: one_calmrag_entry.json")

