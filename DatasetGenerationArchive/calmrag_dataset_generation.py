import os
import json
import random
import time
import re
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from ddgs import DDGS
import trafilatura
import openai
from source_finder import SourceFinder

# Global query history to avoid duplicate queries
query_history = {}

# Load existing query history from file at module import
def load_query_history_from_file():
    """Load existing query history from file at startup"""
    try:
        with open("query_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Initialize global query history from file
query_history = load_query_history_from_file()
print(f"Loaded existing query history: {len(query_history)} topics")

def save_query_history_to_file():
    try:
        with open("query_history.json", "w") as f:
            json.dump(query_history, f, indent=2)
        print(f"Query history saved to query_history.json ({len(query_history)} queries)")
    except Exception as e:
        print(f"Warning: failed to save query_history.json: {e}")

#API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# Generate a distraction query to search for distraction articles
def generate_distraction_query(domain, question):
   prompt = f"""
You are helping build a dataset that evaluates language model reasoning.
Given a question in the domain of "{domain}", generate a web search query that sounds vaguely related to the domain but is NOT relevant to answering the question.
Make the distraction query look like a real Google search that would return articles that sound scientific or educational but are actually off-topic.
However, ensure that the distraction query is not too similar to the original question.
Original question: "{question}"
Distraction query:
"""
   response = openai.chat.completions.create(
       model="gpt-4o",
       messages=[{"role": "user", "content": prompt}],
       temperature=0.9,
       max_tokens=100
   )
   return response.choices[0].message.content.strip().strip('"')




# Extract article text
def extract_article_text(url):
   try:
       downloaded = trafilatura.fetch_url(url)
       if not downloaded:
           return None
       text = trafilatura.extract(downloaded)
       if not text:
           return None
       
       # Clean and normalize the text
       text = re.sub(r'\s+', ' ', text.strip())
       
       # For distraction sources, just return first 2000 chars (no relevance scoring needed)
       return text[:2000] if text else None
   except:
       return None


# Generate QA from gold passage
def generate_qa_from_passage(passage, topic):
   prompt = f"""
You're helping build a dataset that tests how well AI systems can reason over conflicting or nuanced information.


Your task:
Read the passage below and write one high-quality question that:
- Requires reasoning beyond surface-level facts
- Has a single correct answer based ONLY on the passage
- Might be misunderstood or debated elsewhere
-Dont reference specifics in the passage in the question or answer (Like "reference graph 2" or "according to the passage" for example)
-Make it so that the question can be answered using other sources, and not just the passage. So the question and answer should be able to be answered by other sources, but still specific to the passage.
-For example, if the passage is about the homestead act, DO NOT ask a question like "What did the act do". It needs to be with enough context that another source could possibly answer this.


-Also make sure the question is related to the topic: {topic}.
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
   response = openai.chat.completions.create(
       model="gpt-4o",
       messages=[{"role": "user", "content": prompt}],
       temperature=0.8,
       max_tokens=300
   )
   text = response.choices[0].message.content
   try:
       q = text.split("Question:")[1].split("Answer:")[0].strip()
       a = text.split("Answer:")[1].strip()
       return q, a
   except:
       return None, None




def duckduckgo_search(query, num_results=100):
   results = []
   try:
       with DDGS() as ddgs:
           for result in ddgs.text(query, region="wt-wt", safesearch="off", max_results=num_results):
               url = result.get('href') or result.get('link') or result.get('url')
               if url:
                   results.append(url)
   except Exception as e:
       print(f"Error searching DuckDuckGo for '{query}': {e}")
   
   return results


def generate_gold_query(topic=str, sub_domains=[], max_attempts=5, used_subdomains=None):
   global query_history
   
   # Initialize topic history if not exists
   if topic not in query_history:
       query_history[topic] = []
   
   # Get existing queries for this topic
   existing_queries = query_history[topic]
   
   # Get the next subdomain to focus on
   if used_subdomains is None:
       used_subdomains = {}
   
   target_subdomain = get_next_subdomain(topic, used_subdomains, sub_domains)
   
   for attempt in range(max_attempts):
       
       prompt = f"""
You are helping build a dataset to evaluate factual question-answering in the realm of "{topic}".

Generate a single, broad, factual question that could be answered by major news outlets, government websites, or industry publications.

CRITICAL REQUIREMENTS:
- DO NOT reference any specific years (especially 2020-2024)
- DO NOT ask about specific dates, quarters, or time periods
- DO NOT ask about specific recent events or announcements
- Focus on general policies, procedures, standards, or established facts
- Make the question broad enough that multiple reputable sources would cover it

IMPORTANT: The question must be DIFFERENT from these recently generated questions:
{chr(10).join([f"- {q}" for q in existing_queries]) if existing_queries else "No previous questions yet"}

FOCUS SPECIFICALLY on this subtopic: {target_subdomain}
- Make the question about {target_subdomain}
- Ask about general principles, standards, or established facts
- The question should have a single correct answer


Only return the question on one line.
"""
       
       response = openai.chat.completions.create(
           model="gpt-4o",
           messages=[{"role": "user", "content": prompt.strip()}],
           temperature=0.9, 
           max_tokens=100
       )
       
       new_query = response.choices[0].message.content.strip().strip('"')
       
       # Filter out queries with specific years or dates
       import re
       if re.search(r'\b(20[0-2][0-9]|first quarter|second quarter|third quarter|fourth quarter|Q[1-4]|january|february|march|april|may|june|july|august|september|october|november|december)\b', new_query.lower()):
           print(f"  REJECTED: Query contains specific dates/years: {new_query}")
           continue
       
       # Check if this query is sufficiently different from existing ones
       #Extra layer of protection to avoid duplicates
       if is_query_diverse(new_query, existing_queries):
           # Add to history and return
           query_history[topic].append(new_query)
           print(f"Gold query (attempt {attempt + 1}): {new_query}")
           
           # Save to file immediately to sync with generate_500.py
           save_query_history_to_file()
           
           return new_query, target_subdomain
       else:
           print(f"Query too similar, retrying... (attempt {attempt + 1})")
   
   # If all attempts failed, generate a fallback last chance query
   fallback_query = f"What are the current guidelines for {target_subdomain} in {topic}?"
   query_history[topic].append(fallback_query)
   print(f"Fallback query generated: {fallback_query}")
   
   # Save to file immediately to sync with generate_500.py
   save_query_history_to_file()
   
   return fallback_query, target_subdomain

def is_query_diverse(new_query, existing_queries):
   #Check if a new query is sufficiently different from existing ones.
   if not existing_queries:
       return True
   
   # Simple similarity check - count common words
   new_words = set(new_query.lower().split())
   
   for existing in existing_queries[-5:]:  # Check against last 5 queries
       existing_words = set(existing.lower().split())
       common_words = new_words.intersection(existing_words)
       
       # If more than 70% of words are common, consider it too similar
       if len(common_words) / max(len(new_words), len(existing_words)) > 0.7:
           return False
   
   return True



def get_next_subdomain(topic, used_subdomains, available_subdomains):
   """Get the next subdomain to focus on, rotating through available ones."""
   
   # Find subdomains that haven't been used much
   subdomain_counts = {}
   for subdomain in available_subdomains:
       subdomain_counts[subdomain] = used_subdomains.get(subdomain, 0)
   
   # Get the subdomain with the lowest usage count
   min_usage = min(subdomain_counts.values())
   candidates = [subdomain for subdomain, count in subdomain_counts.items() if count == min_usage]
   
   # Pick randomly from least used subdomains
   selected_subdomain = random.choice(candidates)
   
   print(f"Selected subdomain: {selected_subdomain} (usage count: {subdomain_counts[selected_subdomain]})")
   return selected_subdomain


def get_paired_sets(gold_query: str, topic: str, exclude_url: str = None, gold_question: str = None, entry_id: str = None):
    print(f"Starting SourceFinder for topic: {topic}, gold_query: {gold_query}")
    finder = SourceFinder(gold_query, gold_question, topic=topic, entry_id=entry_id)
    result = finder.find_sources(exclude_url=exclude_url)

    return {
        "clear_set": result.get("clear_set", []),
        "ambiguous_set": result.get("unclear_set", [])
    }


class CalmRagEntry:
   def __init__(self, entry_id, topic):
       self.entry_id = entry_id
       self.topic = topic
       
       # Define subdomains FIRST
       if self.topic == "public_health":
            self.subdomains = [
                "vaccines", "infectious_diseases", "nutrition_guidelines",
                "mental_health", "maternal_and_child_health",
                "chronic_diseases", "occupational_safety", "toxicology",
                "emergency_medicine", "preventive_care", "healthcare_policy",
                "medical_research", "public_health_education"
            ]
       elif self.topic == "current_events":
            self.subdomains = [
                "international_conflicts", "elections", "natural_disasters",
                "pandemics", "economic_crises", "major_legislation",
                "scientific_breakthroughs", "protests_and_movements"
            ]
       elif self.topic == "history":
            self.subdomains = [
                "ancient_civilizations", "world_wars", "revolutions",
                "colonialism", "cold_war", "civil_rights_movements",
                "historical_figures", "archaeological_discoveries", "us-history"
            ]
       elif self.topic == "finance":
            self.subdomains = [
                "stock_markets", "banking", "cryptocurrency", 
                "inflation_and_recession", "personal_finance", 
                "global_trade", "housing_markets", "investment_strategies"
            ]
       elif self.topic == "sports":
            self.subdomains = [
                "olympics", "soccer", "basketball", "baseball", 
                "tennis", "athletics", "sports_medicine", 
                "sports_history", "doping_scandals", "football"
            ]
       elif self.topic == "climate":
            self.subdomains = [
                "climate_change", "carbon_emissions", "renewable_energy",
                "sea_level_rise", "deforestation", "pollution",
                "sustainable_development", "climate_policy"
            ]
       elif self.topic == "technology":
            self.subdomains = [
                "artificial_intelligence", "social_media", "cybersecurity",
                "5g_networks", "biotechnology", "quantum_computing",
                "software_and_internet", "consumer_electronics"
            ]
       elif self.topic == "astronomy":
            self.subdomains = [
                "planets_and_moons", "stars_and_galaxies", "black_holes",
                "space_exploration", "telescopes", "cosmology",
                "exoplanets", "astrobiology", "universe_origin", "moon_landing"
            ]
       elif self.topic == "law":
            self.subdomains = [
                "constitutional_law", "criminal_law", "international_law",
                "civil_rights", "supreme_court_cases", "intellectual_property",
                "environmental_law", "human_rights_law", "immigration_law"
            ]
       elif self.topic == "psychology":
            self.subdomains = [
                "clinical_psychology", "cognitive_psychology", "developmental_psychology",
                "social_psychology", "behavioral_psychology", "neuropsychology",
                "forensic_psychology", "health_psychology", "industrial_psychology",
                "educational_psychology", "abnormal_psychology", "personality_psychology",
                "experimental_psychology", "counseling_psychology", "sports_psychology"
            ]
       
       # NOW load existing subdomain usage for this topic (after subdomains are defined)
       self.used_subdomains = self.load_subdomain_usage_for_topic(topic)
       
       # Generate gold query with subdomain rotation
       self.gold_query, self.current_subdomain = generate_gold_query(self.topic, self.subdomains, used_subdomains=self.used_subdomains)
       
       # Update subdomain usage
       self.used_subdomains[self.current_subdomain] = self.used_subdomains.get(self.current_subdomain, 0) + 1
       
              # Save updated subdomain usage
       self.save_subdomain_usage_for_topic(self.topic, self.used_subdomains)
   
   def load_subdomain_usage_for_topic(self, topic):
       """Load subdomain usage for a specific topic from file"""
       subdomain_file = f"subdomain_usage_{topic}.json"
       try:
           with open(subdomain_file, "r") as f:
               usage = json.load(f)
               print(f"Loaded subdomain usage for {topic}: {usage}")
               return usage
       except FileNotFoundError:
           # Initialize with all subdomains at 0 usage
           usage = {subdomain: 0 for subdomain in self.subdomains}
           print(f"Initialized new subdomain usage for {topic}: {usage}")
           return usage
       except Exception as e:
           print(f"Error loading subdomain usage for {topic}: {e}")
           # Fallback to initialized usage
           usage = {subdomain: 0 for subdomain in self.subdomains}
           return usage
   
   def save_subdomain_usage_for_topic(self, topic, usage):
       """Save subdomain usage for a specific topic to file"""
       subdomain_file = f"subdomain_usage_{topic}.json"
       try:
           with open(subdomain_file, "w") as f:
               json.dump(usage, f, indent=2)
           print(f"Saved subdomain usage for {topic}: {usage}")
       except Exception as e:
           print(f"Error saving subdomain usage for {topic}: {e}")
   
   def build(self):   
       potential_gold_urls = duckduckgo_search(self.gold_query)
       gold_passage = None
       gold_url=None    
       
       # Load reliable domains from SourceFinder
       if not SourceFinder._config_loaded:
           SourceFinder._load_domain_config_static("domain_trust_config.json")
       reliable_domains = SourceFinder._reliable_domains

       for url in potential_gold_urls:
           time.sleep(1)
           # Accept .gov/.edu OR domains from reliable list
           domain = urlparse(url).netloc.lower().replace("www.", "")
           is_reliable = (
               "gov" in url or "edu" in url or
               any(domain == d or domain.endswith("." + d) for d in reliable_domains)
           )
           
           if is_reliable:
               article = extract_article_text(url)
               if article and len(article) > 300:
                   gold_passage = article[:5000]
                   gold_url = url
                   break


       if not gold_passage:
           print("Failed to find a gold source.")
           return None


       question, gold_answer = generate_qa_from_passage(gold_passage, self.topic)
       if not question or not gold_answer:
           print("Failed to generate QA.")
           return None


       print(f" Q: {question}\n A: {gold_answer}")


       # Find 2 distraction sources
       potential_distraction_urls = duckduckgo_search(generate_distraction_query(self.topic, question))
       distraction_sources = []
       
       for url in potential_distraction_urls:
           time.sleep(1)
           article = extract_article_text(url)
           if article and len(article) > 300:
               distraction_sources.append({
                   'url': url,
                   'text': article[:2000]
               })
               if len(distraction_sources) >= 2:  # Stop after finding 2
                   break


       if len(distraction_sources) < 2:
           print("Failed to find enough distraction sources.")
           return None


       
       paired = get_paired_sets(gold_query=self.gold_query, topic=self.topic, exclude_url=gold_url, gold_question=question, entry_id=self.entry_id)
       clear_set = paired.get("clear_set", [])
       ambiguous_set = paired.get("ambiguous_set", [])

       if len(clear_set) == 0 or len(ambiguous_set) == 0:
            print("source finder did not return paired sets (clear/unclear).")
            return None
       # Add both distraction sources to ambiguous set, avoiding duplicates
       for distraction_source in distraction_sources:
           distraction_url = distraction_source['url']
           distraction_text = distraction_source['text']
           
           # Check if distraction URL already exists in any set to avoid duplicates
           distraction_exists = False
           for source in clear_set + ambiguous_set:
               if source.get("url") == distraction_url:
                   distraction_exists = True
                   break
           
           if not distraction_exists:
                ambiguous_set.append({
                    "url": distraction_url,
                    "domain": urlparse(distraction_url).netloc.replace("www.", ""),
                    "category": "distraction",
                    "title": urlparse(distraction_url).netloc,
                    "text": distraction_text[:1000],        
                    "timestamp": datetime.now().isoformat(),
                    "score": "N/A"
                })
                print(f"Added distraction source: {distraction_url}")
           else:
                print(f"Skipped duplicate distraction source: {distraction_url}") 
          

       
       return {
           "id": self.entry_id,
           "topic": self.topic,
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
    entry = CalmRagEntry(1, "public_health")
    result = entry.build()
    if result:
       with open("calmrag_dataset_generation.json", "w") as f:
           json.dump(result, f, indent=2)
       print("Saved: calmrag_dataset_generation.json")

