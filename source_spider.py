import scrapy
import json
import random
from urllib.parse import urlparse
import re
from datetime import datetime

class SourceSpider(scrapy.Spider):
    name = "source_spider"
    
    # Conservative settings
    custom_settings = {
        'CONCURRENT_REQUESTS': 12,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'DOWNLOAD_DELAY': 1,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 10,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
        'REACTOR': 'twisted.internet.selectreactor.SelectReactor',
    }

    def __init__(self, gold_url=None, topic=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold_url = gold_url
        self.topic = topic.lower() if topic else ""
        self.start_urls = [gold_url] if gold_url else []
        
        # Track seen domains to avoid duplicates
        self.seen_domains = set()
        
        # Store articles by category
        self.articles = {
            'reliable': [],
            'unreliable': [],  # This covers both unreliable and misleading
            'offtopic': []
        }
        
        # Target counts per category
        self.target_per_category = 100
        self.max_total_articles = 400  # Safety limit
        
        # Load domain trust configuration
        try:
            with open("domain_trust_config.json", "r") as f:
                trust_config = json.load(f)
                topic_config = trust_config.get(self.topic, {})
                self.reliable_domains = set(topic_config.get("reliable", []))
                self.unreliable_domains = set(topic_config.get("unreliable", []))
        except FileNotFoundError:
            self.reliable_domains = set()
            self.unreliable_domains = set()
            self.logger.warning("domain_trust_config.json not found, using keyword-based classification only")
        
        # Topic-related keywords for relevance detection
        self.topic_keywords = self._get_topic_keywords(self.topic)
        
        # Reliability indicators
        self.reliability_indicators = {
            'reliable': [
                'peer-reviewed', 'clinical trial', 'systematic review', 'meta-analysis',
                'randomized controlled', 'double-blind', 'official statement',
                'government report', 'academic study', 'research paper'
            ],
            'unreliable': [
                'unverified', 'rumor', 'speculation', 'conspiracy', 'fake news',
                'misleading', 'debunked', 'misinformation', 'propaganda',
                'opinion blog', 'anonymous source'
            ]
        }

    def _get_topic_keywords(self, topic):
        """Generate relevant keywords for the given topic"""
        # This is a simplified approach - in practice, you might use more sophisticated NLP
        topic_keyword_map = {
            'health': ['medical', 'disease', 'treatment', 'therapy', 'clinical', 'patient', 'hospital', 'doctor'],
            'finance': ['economy', 'market', 'investment', 'banking', 'financial', 'money', 'stock', 'trading'],
            'politics': ['government', 'election', 'policy', 'political', 'congress', 'senate', 'president'],
            'technology': ['tech', 'software', 'hardware', 'innovation', 'digital', 'computer', 'internet'],
            'sports': ['game', 'team', 'player', 'championship', 'league', 'tournament', 'athletic'],
            'climate': ['climate', 'environment', 'global warming', 'carbon', 'emission', 'sustainable']
        }
        
        return topic_keyword_map.get(topic, [topic]) if topic else []

    def parse(self, response):
        """Main parsing method"""
        
        # Check if we've reached our targets
        if self._collection_complete():
            return
        
        domain = urlparse(response.url).netloc.replace("www.", "")
        
        # Skip if we've already processed this domain
        if domain in self.seen_domains:
            return
        self.seen_domains.add(domain)

        # Extract content
        title = self._extract_title(response)
        text = self._extract_text(response)
        
        # Skip if content is too short
        if not text or len(text.strip()) < 200:
            return

        # Determine category
        category = self._categorize_source(domain, title, text)
        
        # Only add if we need more of this category
        if len(self.articles[category]) < self.target_per_category:
            article = {
                "url": response.url,
                "domain": domain,
                "category": category,
                "title": title,
                "text": text[:1000],  # Keep more text for better analysis
                "timestamp": datetime.now().isoformat()
            }
            
            self.articles[category].append(article)
            self.logger.info(f"Added {category} source: {domain} (Total: {len(self.articles[category])})")

        # Continue crawling if we need more sources
        if not self._collection_complete():
            # Follow links to find more sources
            for href in response.css("a::attr(href)").getall():
                if href and self._should_follow_link(href, response.url):
                    yield response.follow(href, callback=self.parse)

    def _extract_title(self, response):
        """Extract page title"""
        title = response.xpath("//title/text()").get()
        if not title:
            title = response.xpath("//h1/text()").get()
        return title.strip() if title else "No Title"

    def _extract_text(self, response):
        """Extract main text content"""
        # Try to get article/main content first
        text_selectors = [
            "//article//text()",
            "//main//text()",
            "//*[@class='content']//text()",
            "//*[@class='article-body']//text()",
            "//p//text()"
        ]
        
        text = ""
        for selector in text_selectors:
            text_parts = response.xpath(selector).getall()
            if text_parts:
                text = " ".join(text_parts)
                break
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _categorize_source(self, domain, title, text):
        """Categorize source as reliable, unreliable, or offtopic"""
        
        content = f"{title} {text}".lower()
        
        # First, check if it's topic-relevant
        if not self._is_topic_relevant(content):
            return 'offtopic'
        
        # Check domain-based reliability
        if domain in self.reliable_domains:
            return 'reliable'
        elif domain in self.unreliable_domains:
            return 'unreliable'
        
        # Check content-based reliability indicators
        reliable_score = sum(1 for keyword in self.reliability_indicators['reliable'] 
                           if keyword in content)
        unreliable_score = sum(1 for keyword in self.reliability_indicators['unreliable'] 
                             if keyword in content)
        
        if reliable_score > unreliable_score:
            return 'reliable'
        elif unreliable_score > reliable_score:
            return 'unreliable'
        else:
            # If unclear, default to unreliable to be conservative
            return 'unreliable'

    def _is_topic_relevant(self, content):
        """Check if content is relevant to the specified topic"""
        if not self.topic or not self.topic_keywords:
            return True  # If no topic specified, consider everything relevant
        
        # Check if any topic keywords appear in the content
        return any(keyword in content for keyword in self.topic_keywords)

    def _should_follow_link(self, href, base_url):
        """Determine if we should follow a link"""
        # Skip non-http links
        if not href.startswith(('http', '/')):
            return False
        
        # Skip common non-content URLs
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/search/', '/login/', '/register/',
            '.pdf', '.jpg', '.png', '.gif', '.css', '.js', '/rss', '/feed'
        ]
        
        return not any(pattern in href.lower() for pattern in skip_patterns)

    def _collection_complete(self):
        """Check if we have enough sources in each category"""
        for category, articles in self.articles.items():
            if len(articles) < self.target_per_category:
                return False
        return True

    def closed(self, reason):
        """Called when spider closes - save results"""
        
        # Save all scraped sources
        all_sources = []
        for category, articles in self.articles.items():
            for article in articles:
                all_sources.append(article)
        
        scraped_data = {
            "gold_url": self.gold_url,
            "topic": self.topic,
            "collection_stats": {
                "reliable": len(self.articles['reliable']),
                "unreliable": len(self.articles['unreliable']),
                "offtopic": len(self.articles['offtopic']),
                "total": len(all_sources)
            },
            "sources": all_sources,
            "completion_reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("scraped_sources.json", "w") as f:
            json.dump(scraped_data, f, indent=2)
        
        # Create paired sets as specified
        self._create_paired_sets()
        
        # Print summary
        self.logger.info(f"Collection complete. Reliable: {len(self.articles['reliable'])}, "
                        f"Unreliable: {len(self.articles['unreliable'])}, "
                        f"Offtopic: {len(self.articles['offtopic'])}")

    def _create_paired_sets(self):
        """Create the two specific sets as required by specifications"""
        
        reliable_sources = self.articles['reliable']
        unreliable_sources = self.articles['unreliable']
        
        # Ensure we have enough sources
        if len(reliable_sources) < 5 or len(unreliable_sources) < 4:
            self.logger.error("Insufficient sources for paired sets creation")
            return
        
        # Create clear set: 4 reliable + 1 unreliable
        clear_reliable = random.sample(reliable_sources, 4)
        clear_unreliable = random.sample(unreliable_sources, 1)
        
        # Create unclear set: 1 reliable + 3 unreliable
        # Make sure we don't reuse the same sources
        remaining_reliable = [s for s in reliable_sources if s not in clear_reliable]
        remaining_unreliable = [s for s in unreliable_sources if s not in clear_unreliable]
        
        unclear_reliable = random.sample(remaining_reliable, 1)
        unclear_unreliable = random.sample(remaining_unreliable, 3)
        
        paired_sets = {
            "gold_url": self.gold_url,
            "topic": self.topic,
            "clear_set": {
                "description": "4 reliable sources + 1 unreliable source",
                "sources": clear_reliable + clear_unreliable
            },
            "unclear_set": {
                "description": "1 reliable source + 3 unreliable sources", 
                "sources": unclear_reliable + unclear_unreliable
            },
            "created_at": datetime.now().isoformat()
        }
        
        with open("paired_sets.json", "w") as f:
            json.dump(paired_sets, f, indent=2)
        
        self.logger.info("Paired sets created successfully")


# Example domain_trust_config.json structure:
example_config = {
    "health": {
        "reliable": ["who.int", "cdc.gov", "nih.gov", "pubmed.ncbi.nlm.nih.gov"],
        "unreliable": ["naturalnews.com", "mercola.com", "healthimpactnews.com"]
    },
    "finance": {
        "reliable": ["sec.gov", "federalreserve.gov", "reuters.com", "bloomberg.com"],
        "unreliable": ["zerohedge.com", "goldsilver.com"]
    }
}

# Save example config if it doesn't exist
import os
if not os.path.exists("domain_trust_config.json"):
    with open("domain_trust_config.json", "w") as f:
        json.dump(example_config, f, indent=2)