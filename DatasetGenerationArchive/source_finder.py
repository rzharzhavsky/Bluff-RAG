import json
import time
import random
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import trafilatura
from ddgs import DDGS
from rapidfuzz import fuzz
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize


# Load environment variables for Reddit API
load_dotenv()

class SourceFinder:
    
    _reliable_domains = set()
    _unreliable_domains = set()
    _config_loaded = False
    
    def __init__(self, gold_query: str, gold_question: str, topic: str = None, domain_trust_config_path: str = "domain_trust_config.json", entry_id: str = None):
        self.gold_query = gold_query
        self.question = gold_question
        self.domain_trust_config_path = domain_trust_config_path
        self.entry_id = entry_id
        
        # Use provided topic or extract from gold query
        if topic:
            self.topic = topic
        else:
            self.topic = self._extract_topic_from_query(gold_query)
        
        # Load trust config only once for the entire dataset generation
        if not SourceFinder._config_loaded:
            SourceFinder._load_domain_config_static(domain_trust_config_path)
        
        # Domain lists are class variables
        
        # Initialize stopwords without requiring downloads(ran into problems using nltk)
        self.stopwords = set(ENGLISH_STOP_WORDS)
        
        # Generate topic-specific search queries
        self.search_queries = self._get_search_queries_for_topic(self.topic, self.gold_query)
        
        # Reddit API credentials
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "CALM-RAG-SourceFinder/1.0")
        
        print(f"SourceFinder initialized for query: '{self.gold_query}'")
        print(f"Extracted topic: {self.topic}")
        print(f"Using {len(SourceFinder._reliable_domains)} reliable domains, {len(SourceFinder._unreliable_domains)} unreliable domains")
        print(f"Generated {len(self.search_queries['reliable'])} reliable and {len(self.search_queries['unreliable'])} unreliable search queries")
        print(f"Reddit API configured: {bool(self.reddit_client_id and self.reddit_client_secret)}")



    @classmethod
    def _load_domain_config_static(cls, domain_trust_config_path: str):
        """Load ALL topics from domain_trust_config.json and union domains globally. Called only once."""
        try:
            with open(domain_trust_config_path, 'r') as f:
                config = json.load(f)
                reliable: set = set()
                unreliable: set = set()
                for topic_key, topic_cfg in config.items():
                    if not isinstance(topic_cfg, dict):
                        continue
                    reliable.update(topic_cfg.get("reliable", []))
                    # some configs use "misleading" for unreliable
                    unreliable.update(topic_cfg.get("misleading", []))
                cls._reliable_domains = reliable
                cls._unreliable_domains = unreliable
                cls._config_loaded = True
                print(f"Domain config loaded once (global): {len(cls._reliable_domains)} reliable, {len(cls._unreliable_domains)} unreliable domains")
        except FileNotFoundError:
            print(f"Warning: {domain_trust_config_path} not found, using empty domain lists")
            cls._reliable_domains = set()
            cls._unreliable_domains = set()
            cls._config_loaded = True
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {domain_trust_config_path}")
            cls._reliable_domains = set()
            cls._unreliable_domains = set()
            cls._config_loaded = True
    
    @classmethod
    def get_domain_counts(cls):
        """Get current domain counts for debugging."""
        return {
            "reliable": len(cls._reliable_domains),
            "unreliable": len(cls._unreliable_domains),
            "config_loaded": cls._config_loaded
        }
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract topic from gold query for domain config loading."""
        # Simple topic extraction - look for common topics
        query_lower = query.lower()
        
        if any(topic in query_lower for topic in ['health', 'medical', 'medicine', 'disease', 'treatment', 'vaccine', 'doctor', 'hospital', 'drug', 'therapy']):
            return 'public_health'
        elif any(topic in query_lower for topic in ['news', 'politics', 'government', 'election', 'policy', 'politician', 'congress', 'senate', 'president', 'breaking', 'current']):
            return 'current_events'
        elif any(topic in query_lower for topic in ['history', 'historical', 'ancient', 'war', 'civilization', 'archaeology', 'museum', 'artifact']):
            return 'history'
        elif any(topic in query_lower for topic in ['finance', 'money', 'investment', 'economy', 'stock', 'market', 'trading', 'crypto', 'bitcoin']):
            return 'finance'
        elif any(topic in query_lower for topic in ['sports', 'game', 'athletic', 'team', 'player', 'championship', 'league', 'tournament']):
            return 'sports'
        else:
            # Default to public_health as it has the most comprehensive domain lists
            return 'public_health'
    
    def _get_search_queries_for_topic(self, topic: str, gold_query: str) -> Dict[str, List[str]]:
        """Generate topic-specific search queries for reliable and unreliable sources."""
        
        base_queries = {
            'reliable': [f"{gold_query}"],
            'unreliable': [f"{gold_query}"]
        }
        
        if topic == 'public_health':
            base_queries['reliable'].extend([
                f"{gold_query} government",
                f"{gold_query} research",
                f"{gold_query} guidelines",
                f"{gold_query} academic",
                f"{gold_query} clinical study",
                f"{gold_query} peer reviewed"
            ])
            base_queries['unreliable'].extend([
                f"{gold_query} reddit",
                f"{gold_query} twitter",
                f"{gold_query} natural medicine",
                f"{gold_query} alternative treatment",
                f"{gold_query} conspiracy",
                f"{gold_query} natural news",
                f"{gold_query} big pharma",
                f"{gold_query} cover up",
                f"{gold_query} holistic cure",
                f"{gold_query} mercola",
                f"{gold_query} viral"
            ])
            
        elif topic == 'current_events':
            base_queries['reliable'].extend([
                f"{gold_query} news",
                f"{gold_query} breaking news",
                f"{gold_query} Reuters AP",
                f"{gold_query} official statement",
                f"{gold_query} government response",
                f"{gold_query} fact check",
                f"{gold_query} verified report",
                f"{gold_query} press release",
                f"{gold_query} BBC CNN",
                f"{gold_query} NPR"
            ])
            base_queries['unreliable'].extend([
                f"{gold_query} reddit",
                f"{gold_query} twitter",
                f"{gold_query} conspiracy",
                f"{gold_query} cover up",
                f"{gold_query} deep state",
                f"{gold_query} mainstream media lies",
                f"{gold_query} alternative media",
                f"{gold_query} truth exposed",
                f"{gold_query} fake news",
                f"{gold_query} infowars"
            ])
            
        elif topic == 'history':
            base_queries['reliable'].extend([
                f"{gold_query} historical research",
                f"{gold_query} academic study",
                f"{gold_query} archaeological evidence",
                f"{gold_query} historical records",
                f"{gold_query} scholarly article",
                f"{gold_query} museum",
                f"{gold_query} documented evidence",
                f"{gold_query} peer reviewed history",
                f"{gold_query} Smithsonian",
                f"{gold_query} National Geographic"
            ])
            base_queries['unreliable'].extend([
                f"{gold_query} reddit",
                f"{gold_query} twitter",
                f"{gold_query} conspiracy",
                f"{gold_query} hidden history",
                f"{gold_query} cover up",
                f"{gold_query} ancient aliens",
                f"{gold_query} lost civilization",
                f"{gold_query} forbidden archaeology",
                f"{gold_query} alternative history",
                f"{gold_query} ancient code"
            ])
            
        elif topic == 'finance':
            base_queries['reliable'].extend([
                f"{gold_query} analysis",
                f"{gold_query} research report",
                f"{gold_query} financial data",
                f"{gold_query} market analysis",
                f"{gold_query} expert opinion",
                f"{gold_query} economic forecast",
                f"{gold_query} institutional report",
                f"{gold_query} SEC filing",
                f"{gold_query} Bloomberg",
                f"{gold_query} Wall Street Journal"
            ])
            base_queries['unreliable'].extend([
                f"{gold_query} reddit",
                f"{gold_query} twitter",
                f"{gold_query} get rich quick",
                f"{gold_query} market crash",
                f"{gold_query} economic collapse",
                f"{gold_query} conspiracy",
                f"{gold_query} manipulation",
                f"{gold_query} pump and dump",
                f"{gold_query} insider secret",
                f"{gold_query} zero hedge"
            ])
            
        elif topic == 'sports':
            base_queries['reliable'].extend([
                f"{gold_query} official news",
                f"{gold_query} sports news",
                f"{gold_query} league report",
                f"{gold_query} official statement",
                f"{gold_query} verified report",
                f"{gold_query} press conference",
                f"{gold_query} sports journalism",
                f"{gold_query} team announcement",
                f"{gold_query} ESPN",
                f"{gold_query} official league"
            ])
            base_queries['unreliable'].extend([
                f"{gold_query} reddit",
                f"{gold_query} twitter",
                f"{gold_query} rumor",
                f"{gold_query} gossip",
                f"{gold_query} unconfirmed",
                f"{gold_query} speculation",
                f"{gold_query} fan theory",
                f"{gold_query} hot take",
                f"{gold_query} controversial opinion",
                f"{gold_query} barstool"
            ])
        
        else:
            # Default fallback for unknown topics
            base_queries['reliable'].extend([
                f"{gold_query} research",
                f"{gold_query} academic",
                f"{gold_query} study",
                f"{gold_query} analysis",
                f"{gold_query} expert opinion",
                f"{gold_query} official",
                f"{gold_query} verified"
            ])
            base_queries['unreliable'].extend([
                f"{gold_query} reddit",
                f"{gold_query} twitter",
                f"{gold_query} conspiracy",
                f"{gold_query} alternative",
                f"{gold_query} unverified",
                f"{gold_query} rumor",
                f"{gold_query} speculation"
            ])
        
        return base_queries

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text, removing stopwords"""
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Remove stopwords and short words
        keywords = [word for word in words if word not in self.stopwords and len(word) > 2]
        return keywords
    
    def _compute_relevance_score(self, question: str, text: str) -> float:
        """Compute relevance score between question and text."""
        # Extract keywords from gold query
        question_keywords = self._extract_keywords(question)
        
        if not question_keywords:
            return 0.0
        
        # Count keyword hits in text
        text_lower = text.lower()
        keyword_hits = sum(1 for keyword in question_keywords if keyword in text_lower)
        keyword_hit_rate = keyword_hits / len(question_keywords)
        
        # Compute fuzzy string similarity
        fuzzy_score = fuzz.token_set_ratio(question, text[:5000]) / 100.0
        
        # Weighted average: 60% keyword hits, 40% fuzzy similarity
        final_score = 0.6 * keyword_hit_rate + 0.4 * fuzzy_score
        
        return final_score
    
    def _classify_domain(self, url: str) -> str:
        """Classify domain as reliable or unreliable (unknown domains default to unreliable)."""
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")
            


            # Check for exact match or subdomain
            if any(domain == d or domain.endswith("." + d) for d in SourceFinder._reliable_domains):
                return "reliable"
            elif (".edu" in domain or ".gov" in domain):
                return "reliable"
            elif any(domain == d or domain.endswith("." + d) for d in SourceFinder._unreliable_domains):
                return "unreliable"
            else:
                # Treat unknown domains as unreliable (fallback)
                return "unreliable"
        except Exception as e:
            print(f"Error classifying domain for {url}: {e}")
            # Treat errors as unreliable (fallback)
            return "unreliable"
    
    def _search_duckduckgo(self, query: str, max_results: int = 30) -> List[Dict]:  # Reduced from 80 to 30
        """Search Duck for urls (ddgs returns keys: href/title/body)"""
        results: List[Dict] = []
        try:
            with DDGS() as ddgs:
                for result in ddgs.text(query, region="wt-wt", safesearch="off", max_results=max_results):
                    url = result.get('href') or result.get('link') or result.get('url')
                    if not url:
                        continue
                    results.append({
                        'url': url,
                        'title': result.get('title', '')
                    })
        except Exception as e:
            print(f"Error searching DuckDuckGo for '{query}': {e}")
        
        return results
    
    def _search_reddit_api(self, query: str, max_results: int = 20) -> List[Dict]:  # Reduced from 40 to 20
        """Search Reddit directly using their API"""
        if not (self.reddit_client_id and self.reddit_client_secret):
            print("  Reddit API not configured, skipping Reddit search")
            return []
        
        try:
            import praw
            
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            print(f"  Searching Reddit for: {query}")
            
            # Search across multiple relevant subreddits based on topic
            subreddits = self._get_relevant_subreddits()
            all_results = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    # Search for posts
                    search_results = subreddit.search(query, limit=max_results//len(subreddits), sort='relevance')
                    
                    for submission in search_results:
                        # Skip if too old or low quality
                        if submission.score < 5 or submission.num_comments < 2:
                            continue
                        
                        # Extract full text content directly here
                        text_parts = []
                        if submission.title:
                            text_parts.append(submission.title)
                        if submission.selftext:
                            text_parts.append(submission.selftext)
                        
                        # Get top comments
                        submission.comment_sort = 'top'
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:3]:  # Top 3 comments
                            if comment.body and len(comment.body) > 50:
                                text_parts.append(comment.body)
                        
                        full_text = '\n\n'.join(text_parts).strip()
                        
                        # Skip if not enough content
                        if len(full_text) < 400:
                            continue
                        
                        result = {
                            'url': f"https://reddit.com{submission.permalink}",
                            'title': submission.title,
                            'text': full_text,  # Full text content
                            'score': submission.score,
                            'subreddit': subreddit_name
                        }
                        all_results.append(result)
                        
                except Exception as e:
                    print(f"    Error searching subreddit {subreddit_name}: {e}")
                    continue
            
            print(f"  Reddit API search found {len(all_results)} results")
            return all_results
            
        except ImportError:
            print("  PRAW not installed. Install with: pip install praw")
            return []
        except Exception as e:
            print(f"  Reddit API search failed: {e}")
            return []
    
    def _get_relevant_subreddits(self) -> List[str]:
        """Get relevant subreddits based on the topic"""
        if self.topic == 'public_health':
            return [
                'health', 'nutrition', 'fitness', 'mentalhealth', 'medical', 
                'wellness', 'supplements', 'herbalism', 'homeopathy',
                'alternative', 'conspiracy', 'naturalmedicine', 'holistic'
            ]
        elif self.topic == 'current_events':
            return [
                'news', 'worldnews', 'politics', 'conspiracy', 'conspiracytheories',
                'alternative', 'truth', 'exposing', 'realnews', 'worldpolitics',
                'politicaldiscussion'
            ]
        elif self.topic == 'history':
            return [
                'history', 'ancienthistory', 'conspiracy', 'alternativehistory',
                'archaeology', 'mystery', 'unsolvedmysteries', 'ancientaliens',
                'forbiddenhistory', 'hiddenhistory', 'lostcivilizations'
            ]
        elif self.topic == 'finance':
            return [
                'investing', 'wallstreetbets', 'cryptocurrency', 'personalfinance',
                'conspiracy', 'economiccollapse', 'preppers', 'gold', 'silver',
                'bitcoin', 'cryptomarkets', 'financialindependence'
            ]
        elif self.topic == 'sports':
            return [
                'sports', 'nba', 'nfl', 'soccer', 'baseball', 'tennis',
                'conspiracy', 'sportscorruption', 'olympics', 'ufc', 'boxing'
            ]
        elif self.topic == 'climate':
            return [
                'climatechange', 'environment', 'climate', 'globalwarming',
                'conspiracy', 'climateskeptics', 'geoengineering', 'chemtrails',
                'environmental', 'sustainability', 'renewableenergy'
            ]
        elif self.topic == 'technology':
            return [
                'technology', 'artificialintelligence', 'programming', 'cybersecurity',
                'conspiracy', 'privacy', 'surveillance', '5g', 'quantumcomputing',
                'machinelearning', 'datascience', 'blockchain'
            ]
        elif self.topic == 'astronomy':
            return [
                'astronomy', 'space', 'nasa', 'cosmology', 'conspiracy',
                'ufo', 'aliens', 'spacex', 'mars', 'moon', 'stars',
                'galaxies', 'blackholes', 'exoplanets'
            ]
        elif self.topic == 'law':
            return [
                'law', 'legaladvice', 'conspiracy', 'politics', 'government',
                'constitutional', 'supremecourt', 'civilrights', 'legal',
                'justice', 'criminaljustice', 'humanrights'
            ]
        elif self.topic == 'psychology':
            return [
                'psychology', 'mentalhealth', 'science', 'conspiracy',
                'neuroscience', 'cognitive', 'behavioral', 'therapy',
                'psychiatry', 'mindfulness', 'consciousness'
            ]
        else:
            # Default subreddits for unknown topics
            return ['conspiracy', 'alternative', 'truth', 'exposing', 'realnews']
    
    def _clean_and_chunk_text(self, text: str, question: str, is_distraction: bool = False) -> str:
        """Clean HTML text and select most relevant chunks"""
        if not text:
            return ""
        
        # Clean the text: remove extra whitespace, normalize, remove common web elements
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common web elements that might have been extracted
        text = re.sub(r'(cookie|privacy|terms|contact|about|advertisement|advert|sponsored|subscribe|newsletter|sign up|log in|menu|navigation|footer|header|sidebar)', '', text, flags=re.IGNORECASE)
        
        # Remove common HTML artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\{.*?\}', '', text)  # Remove braced content
        text = re.sub(r'<.*?>', '', text)    # Remove any remaining HTML tags
        
        # Split into sentences using robust fallback method
        sentences = []
        try:
            # Try NLTK first
            try:
                nltk.data.find('tokenizers/punkt')
                sentences = sent_tokenize(text)
            except LookupError:
                # NLTK not available, use regex fallback
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            print(f"    NLTK tokenization failed: {e}, using fallback")
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create chunks of 200-400 tokens (roughly 2-4 sentences)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            # Use simple word counting instead of NLTK word_tokenize
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > 500:
                # Current chunk is getting too long, save it
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
                # If we have a good chunk size, save it
                if current_tokens >= 300:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
        
        # Add any remaining text as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        if not chunks:
            return text[:2000]  # Fallback to original method
        
        # For distraction sources, just return first chunk (no relevance scoring needed)
        if is_distraction:
            return chunks[0][:1000]
        
        # Score each chunk against the question using existing relevance method
        chunk_scores = []
        for chunk in chunks:
            score = self._compute_relevance_score(question, chunk)
            chunk_scores.append((chunk, score))
        
        # Sort by relevance score (highest first)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 1-2 chunks (300-500 tokens total)
        selected_chunks = []
        total_tokens = 0
        
        for chunk, score in chunk_scores:
            # Use simple word counting instead of NLTK word_tokenize
            chunk_tokens = len(chunk.split())
            if total_tokens + chunk_tokens <= 500:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                if len(selected_chunks) >= 2:  # Max 2 chunks
                    break
        
        # Combine selected chunks
        final_text = ' '.join(selected_chunks)
        
        print(f"    Created {len(chunks)} chunks, selected {len(selected_chunks)} best chunks")
        print(f"    Total tokens: {total_tokens}")
        if chunk_scores:
            print(f"    Best chunk score: {chunk_scores[0][1]:.3f}")
        
        # Ensure we don't exceed reasonable length
        if len(final_text) > 2000:
            final_text = final_text[:2000]
        
        return final_text

    def _extract_text_from_url(self, url: str, question: str = None, is_distraction: bool = False) -> Optional[str]:
        """Extract text content from url using trafilatura with improved cleaning and chunking"""
        print(f"  Extracting text from: {url}")

        def _do_extract() -> Optional[str]:
            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    return None
                text = trafilatura.extract(downloaded)
                return text.strip() if text else None
            except Exception:
                return None

        # Skip problematic domains that consistently timeout
        problematic_domains = {
            'gatorcountry.com', 'totalenergies.fr', 'statbase.org',
            'olympics.com', 'worldatlas.com', 'cdn.bookey.app',
            'research-information.bris.ac.uk', 'tiktok.com'
        }
        domain = urlparse(url).netloc.lower()
        if any(prob_domain in domain for prob_domain in problematic_domains):
            print(f"    Skipping problematic domain: {domain}")
            return None
            
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_extract)
            try:
                text = future.result(timeout=8)  # Faster timeout to avoid long waits
            except FuturesTimeout:
                print("    Extraction timed out (>8s), skipping")
                return None
            
        if not text:
            print("    Failed to download content")
            return None
            
        if len(text) < 400:
            print(f"    Text too short ({len(text)} chars), skipping")
            return None
        
        # Clean and chunk the text
        cleaned_text = self._clean_and_chunk_text(text, question or self.question, is_distraction)
        
        print(f"    Original text: {len(text)} chars")
        print(f"    Cleaned text: {len(cleaned_text)} chars")
        
        return cleaned_text
    
    def find_sources(self, exclude_url: Optional[str] = None) -> Dict[str, List]:
        """
        Main method to find and categorize sources.
        Returns dict with clear_set and unclear_set.
        """
        print(f"\n=== Starting source discovery for query: '{self.gold_query}' ===")
        
        all_sources = []
        seen_urls = set()  # Track URLs to prevent duplicates during collection
        total_urls_encountered = 0  # Track total URLs seen (including duplicates)
        
        # Search for both reliable and unreliable sources
        for source_type, queries in self.search_queries.items():
            print(f"\n--- Searching for {source_type} sources ---")
            
            for query in queries:
                print(f"Searching: '{query}'")
                search_results = self._search_duckduckgo(query, max_results=15)
                print(f"  Found {len(search_results)} URLs")
                
                for result in search_results:
                    url = result['url']
                    total_urls_encountered += 1
                    
                    # Skip if we've already processed this URL
                    if url in seen_urls:
                        print(f"\nSkipping duplicate URL: {url}")
                        continue
                    
                    print(f"\nProcessing: {url}")
                    
                    # Extract text with question for relevance scoring
                    text = self._extract_text_from_url(url, question=self.question, is_distraction=False)
                    if not text:
                        print(f"  REJECTED: Could not extract text")
                        continue
                    
                    # Compute relevance score
                    score = self._compute_relevance_score(self.question, text)
                    print(f"  Relevance score: {score:.3f}")
                    
                    if score < 0.25:  # Lowered threshold for faster processing
                        print(f"  REJECTED: Score too low ({score:.3f} < 0.25)")
                        continue
                    
                    # Classify domain
                    category = self._classify_domain(url)
                    print(f"  Domain category: {category}")
                    
                    # Create source entry
                    source = {
                        'url': url,
                        'domain': urlparse(url).netloc.replace("www.", ""),
                        'category': category,
                        'score': score,
                        'text': text[:1000],  # Keep first 1000 chars
                        'title': result.get('title', '')
                    }
                    
                    all_sources.append(source)
                    seen_urls.add(url)  # Mark URL as processed
                    print(f"  ACCEPTED: {category} source with score {score:.3f}")
        
        # REDDIT FALLBACK: Always search Reddit to ensure enough misleading sources
        print(f"\n--- Reddit API Fallback Search ---")
        reddit_results = self._search_reddit_api(self.gold_query, max_results=30)
        
        # Sort Reddit results by relevance score (highest first) before processing
        reddit_results_with_scores = []
        for result in reddit_results:
            # Pre-compute relevance score for sorting
            text = result['text']
            score = self._compute_relevance_score(self.question, text)
            reddit_results_with_scores.append({
                'result': result,
                'score': score
            })
        
        # Sort by relevance score (highest first)
        reddit_results_with_scores.sort(key=lambda x: x['score'], reverse=True)
        print(f"  Reddit results sorted by relevance score (highest first)")
        
        for item in reddit_results_with_scores:
            result = item['result']
            score = item['score']
            url = result['url']
            if url in seen_urls:
                continue
                
            print(f"\nProcessing Reddit: {url}")
            
            # Use the full text directly from the search result
            text = result['text']
            
            # Use pre-computed score from sorting
            print(f"  Reddit relevance score: {score:.3f}")
            
            if score < 0.3:  # Lower threshold for Reddit sources
                print(f"  REJECTED: Reddit score too low ({score:.3f} < 0.3)")
                continue
            
            # Reddit sources are classified as unreliable
            category = "unreliable"
            print(f"  Reddit domain category: {category}")
            
            # Create Reddit source entry
            source = {
                'url': url,
                'domain': 'reddit.com',
                'category': category,
                'score': score,
                'text': text[:1000],  # Limit to 1000 chars for consistency
                'title': result.get('title', '')
            }
            
            all_sources.append(source)
            seen_urls.add(url)
            print(f"  ACCEPTED: Reddit {category} source with score {score:.3f}")
        
        print(f"\n=== Source discovery complete ===")
        print(f"Total sources found: {len(all_sources)}")
        print(f"Unique URLs processed: {len(seen_urls)}")
        print(f"Total URLs encountered: {total_urls_encountered}")
        print(f"Duplicate URLs removed: {total_urls_encountered - len(seen_urls)}")
        
        # Exclude the gold_url if provided
        if exclude_url:
            def normalize_url(url):
                if not url:
                    return None
                parsed = urlparse(url)
                return (parsed.scheme, parsed.netloc, parsed.path.rstrip('/'))
            
            exclude_url_norm = normalize_url(exclude_url)
            before_count = len(all_sources)
            all_sources = [
                s for s in all_sources
                if normalize_url(s.get('url')) != exclude_url_norm
            ]
            after_count = len(all_sources)
            print(f"Excluded gold_url from sources: {before_count - after_count} removed")
        
        # Dump all collected sources for debugging before pairing
        try:
            reliable_count = sum(1 for s in all_sources if s.get('category') == 'reliable')
            unreliable_count = sum(1 for s in all_sources if s.get('category') == 'unreliable')
            unknown_count = sum(1 for s in all_sources if s.get('category') == 'unknown')
            
            # Load existing scraped sources or create new list
            scraped_sources_file = "scraped_sources.json"
            try:
                with open(scraped_sources_file, "r") as f:
                    existing_scraped = json.load(f)
                    if isinstance(existing_scraped, list):
                        # If it's already a list, use it
                        all_scraped_data = existing_scraped
                    else:
                        # If it's the old format, convert to list
                        all_scraped_data = [existing_scraped]
                    print(f"Loaded existing scraped sources with {len(all_scraped_data)} entries")
            except FileNotFoundError:
                all_scraped_data = []
                print("Creating new scraped sources file")
            
            # Add new scraped data
            new_scraped_data = {
                "entry_id": getattr(self, 'entry_id', 'unknown'),  # Add entry ID if available
                "gold_query": self.gold_query,
                "topic": self.topic,
                "collection_stats": {
                    "reliable": reliable_count,
                    "unreliable": unreliable_count,
                    "unknown": unknown_count,
                    "total": len(all_sources)
                },
                "sources": all_sources,
                "timestamp": datetime.now().isoformat()
            }
            all_scraped_data.append(new_scraped_data)
            
            # Save updated scraped sources
            with open(scraped_sources_file, "w") as f:
                json.dump(all_scraped_data, f, indent=2)
            print(f"Saved pre-pairing sources to {scraped_sources_file} (now contains {len(all_scraped_data)} entries)")
        except Exception as e:
            print(f"Warning: failed to write scraped_sources.json: {e}")
        
        # Create paired sets
        paired_sets = self._create_paired_sets(all_sources)
        
        return paired_sets
    
    def _create_paired_sets(self, sources: List[Dict]) -> Dict[str, List]:
        """Create clear_set and ambiguous set from collected sources"""
        print(f"\n=== Creating paired sets ===")
        
        # Separate sources by category
        reliable_sources = [s for s in sources if s['category'] == 'reliable']
        unreliable_sources = [s for s in sources if s['category'] == 'unreliable']
        unknown_sources = [s for s in sources if s['category'] == 'unknown']
        
        print(f"Reliable sources: {len(reliable_sources)}")
        print(f"Unreliable sources: {len(unreliable_sources)}")
        print(f"Unknown sources: {len(unknown_sources)}")
        
        random.shuffle(reliable_sources)
        unreliable_sources.sort(key=lambda x: x['score'], reverse=True)
        # Sort unknown sources by relevance score (highest first) instead of random shuffle
        unknown_sources.sort(key=lambda x: x['score'], reverse=True)
        
        # Create clear set: 4 reliable + 1 unreliable
        clear_set = []
        if len(reliable_sources) >= 4:
            clear_set.extend(reliable_sources[:4])
            print(f"Clear set: Added 4 reliable sources")
        else:
            print(f"Warning: Only {len(reliable_sources)} reliable sources available for clear set")
            clear_set.extend(reliable_sources)
        
        if len(unreliable_sources) >= 1:
            clear_set.append(unreliable_sources[0])
            print(f"Clear set: Added 1 unreliable source")
        else:
            # Fallback: use unknown sources marked as unreliable
            if len(unknown_sources) >= 1:
                fallback_source = unknown_sources[0].copy()
                fallback_source['category'] = 'unknown(marked as unreliable)'
                clear_set.append(fallback_source)
                print(f"Clear set: Added 1 unknown source marked as unreliable")
            else:
                print(f"Warning: No unreliable or unknown sources available for clear set")
        
        # Create unclear set: 1 reliable + 2 unreliable
        unclear_set = []
        if len(reliable_sources) >= 5:  # Make sure we don't reuse from clear set
            unclear_set.append(reliable_sources[4])
            print(f"Unclear set: Added 1 reliable source")
        else:
            print(f"Warning: Not enough reliable sources for unclear set")
        
        # Try to add 2 unreliable sources, fallback to unknown if needed
        unreliable_needed = 2
        unreliable_added = 0
        
        if len(unreliable_sources) >= 3:  # Make sure we don't reuse from clear set
            unclear_set.extend(unreliable_sources[1:3])
            unreliable_added = 2
            print(f"Unclear set: Added 2 unreliable sources")
        else:
            # Add whatever unreliable sources we have
            remaining_unreliable = unreliable_sources[1:] if len(unreliable_sources) > 1 else []
            unclear_set.extend(remaining_unreliable)
            unreliable_added = len(remaining_unreliable)
            print(f"Unclear set: Added {unreliable_added} unreliable sources")
            
            # Fill remaining slots with unknown sources marked as unreliable
            remaining_slots = 2 - unreliable_added
            if remaining_slots > 0 and len(unknown_sources) > 0:
                # Skip sources already used in clear set
                available_unknown = unknown_sources[1:] if len(unknown_sources) > 1 else unknown_sources
                for i, unknown_source in enumerate(available_unknown[:remaining_slots]):
                    fallback_source = unknown_source.copy()
                    fallback_source['category'] = 'unknown(marked as unreliable)'
                    unclear_set.append(fallback_source)
                    unreliable_added += 1
                print(f"Unclear set: Added {remaining_slots} unknown sources marked as unreliable")
        
        # Convert to expected format
        def format_source(source):
            return {
                'url': source['url'],
                'domain': source['domain'],
                'category': source['category'],
                'title': source.get('title', ''),
                'text': source['text'],
                "timestamp": datetime.now().isoformat(),
                'score': source['score']
            }
        
        result = {
            'clear_set': [format_source(s) for s in clear_set],
            'unclear_set': [format_source(s) for s in unclear_set]
        }
        
        print(f"\n=== Paired sets created ===")
        print(f"Clear set: {len(result['clear_set'])} sources")
        print(f"Unclear set: {len(result['unclear_set'])} sources")
        
        return result
    
    def build(self) -> Dict[str, List]:
        """
        Main build method that integrates 
        Returns the same format as the old spider
        """
        
        print(f"\n{'='*60}")
        print(f"SOURCE FINDER BUILD STARTING")
        print(f"Topic: {self.topic}")
        print(f"Query: {self.gold_query}")
        print(f"{'='*60}")
        
        try:
            paired_sets = self.find_sources()
            
            # Convert to the exact format expected by one_calmrag_entry.py
            formatted_result = {
                'clear_set': paired_sets['clear_set'],
                'unclear_set': paired_sets['unclear_set']
            }
            
            print(f"\n{'='*60}")
            print(f"BUILD COMPLETED SUCCESSFULLY")
            print(f"Clear set: {len(formatted_result['clear_set'])} sources")
            print(f"Unclear set: {len(formatted_result['unclear_set'])} sources")
            print(f"{'='*60}")
            
            return formatted_result
            
        except Exception as e:
            print(f"\nERROR during build: {e}")
            print(f"Returning empty sets")
            return {
                'clear_set': [],
                'unclear_set': []
            }