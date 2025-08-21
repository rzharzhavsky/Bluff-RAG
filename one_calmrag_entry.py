import os
import json
import random
import time
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from ddgs import DDGS
import trafilatura
import openai
from source_finder import SourceFinder

# Global query history to avoid duplicate queries
query_history = {}

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
-Dont reference specifics in the passage in the question or answer (Like "reference graph 2" or "according to the passage" for example)
-Make it so that the question can be answered using other sources, and not just the passage. So the question and answer should be able to be answered by other sources, but still specific to the passage.


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

#langchain
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

Generate a single, specific, factual question that could be answered by a reputable source (like a .gov or .edu website).

IMPORTANT: The question must be DIFFERENT from these previously generated questions for this topic:
{chr(10).join([f"- {q}" for q in existing_queries]) if existing_queries else "No previous questions yet"}

FOCUS SPECIFICALLY on this subtopic: {target_subdomain}
- Make the question specific to {target_subdomain}
- Make the question specific and factual
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
       
       # Check if this query is sufficiently different from existing ones
       #Extra layer of protection to avoid duplicates
       if is_query_diverse(new_query, existing_queries):
           # Add to history and return
           query_history[topic].append(new_query)
           print(f"Gold query (attempt {attempt + 1}): {new_query}")
           return new_query, target_subdomain
       else:
           print(f"Query too similar, retrying... (attempt {attempt + 1})")
   
   # If all attempts failed, generate a fallback last chance query
   fallback_query = f"What are the current guidelines for {target_subdomain} in {topic}?"
   query_history[topic].append(fallback_query)
   print(f"Fallback query generated: {fallback_query}")
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


def get_paired_sets(gold_query: str, topic: str, exclude_url: str = None):
    print(f"Starting SourceFinder for topic: {topic}, gold_query: {gold_query}")
    finder = SourceFinder(gold_query)
    result = finder.find_sources(exclude_url=exclude_url)

    return {
        "clear_set": result.get("clear_set", []),
        "ambiguous_set": result.get("unclear_set", [])
    }


class CalmRagEntry:
   def __init__(self, entry_id, topic):
       self.entry_id = entry_id
       self.topic = topic
       
       # Track subdomain usage for this topic
       self.used_subdomains = {}
       
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
       
       # Generate gold query with subdomain rotation
       self.gold_query, self.current_subdomain = generate_gold_query(self.topic, self.subdomains, used_subdomains=self.used_subdomains)
       
       # Update subdomain usage
       self.used_subdomains[self.current_subdomain] = self.used_subdomains.get(self.current_subdomain, 0) + 1
  
   def build(self):   
       potential_gold_urls = duckduckgo_search(self.gold_query)
       gold_passage = None
       gold_url=None    
       


       for url in potential_gold_urls:
           time.sleep(1)
           if "gov" in url or "edu" in url:
               article = extract_article_text(url)
               if article and len(article) > 300:
                   gold_passage = article[:5000]
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


       potential_distraction_urls = duckduckgo_search(generate_distraction_query(self.topic, question))
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


       
       paired = get_paired_sets(gold_query=question, topic=self.topic, exclude_url=gold_url)
       clear_set = paired.get("clear_set", [])
       ambiguous_set = paired.get("ambiguous_set", [])

       if len(clear_set) == 0 or len(ambiguous_set) == 0:
            print("Spider did not return paired sets (clear/unclear).")
            return None
       # Check if distraction URL already exists in any set to avoid duplicates
       distraction_exists = False
       for source in clear_set + ambiguous_set:
           if source.get("url") == distraction_url:
               distraction_exists = True
               break
       
       if distraction_text and not distraction_exists:
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
       elif distraction_exists:
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
       with open("one_calmrag_entry.json", "w") as f:
           json.dump(result, f, indent=2)
       print("Saved: one_calmrag_entry.json")

