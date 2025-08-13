import scrapy
from scrapy.exceptions import CloseSpider
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
        self.results_for_calmrag = {"clear_set": [], "ambigous_set": []}
        # Updated targets to ensure enough sources for paired sets
        self.targets = {"reliable": 5, "unreliable": 4}
        
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
                self.unreliable_domains = set(topic_config.get("misleading", []))
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
                'government report', 'academic study', 'research paper',
                'university', 'institute', 'academic', 'research', 'study', 'trial',
                'scientific', 'evidence-based', 'publication', 'journal', 'scholarly',
                'peer review', 'academic journal', 'research institution', 'medical school',
                'health center', 'medical center', 'academic medical center', 'teaching hospital',
                'research center', 'laboratory', 'clinical research', 'medical research',
                'health research', 'public health research', 'epidemiology', 'biomedical',
                'medical education', 'health education', 'continuing education', 'professional development',
                'accreditation', 'certification', 'board certified', 'licensed', 'registered',
                'professional association', 'medical society', 'health organization', 'nonprofit',
                'foundation', 'trust', 'endowment', 'grant', 'funding', 'sponsored research'
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
            'public_health': [
                'medical', 'disease', 'treatment', 'therapy', 'clinical', 'patient', 'hospital', 'doctor',
                'health', 'medicine', 'nutrition', 'vitamin', 'supplement', 'wellness', 'fitness',
                'research', 'study', 'trial', 'academic', 'university', 'institute', 'center',
                'foundation', 'association', 'journal', 'publication', 'scientific', 'evidence',
                'prevention', 'diagnosis', 'symptoms', 'cure', 'recovery', 'rehabilitation',
                'pharmaceutical', 'drug', 'medication', 'prescription', 'dosage', 'side effect',
                'epidemiology', 'public health', 'community health', 'global health', 'mental health',
                'physical health', 'dental health', 'vision health', 'reproductive health',
                'pediatric', 'geriatric', 'obstetric', 'gynecologic', 'cardiology', 'oncology',
                'neurology', 'psychiatry', 'dermatology', 'orthopedics', 'radiology', 'pathology',
                'population health', 'health policy', 'healthcare', 'medical care', 'healthcare system',
                'health education', 'health promotion', 'disease prevention', 'health surveillance',
                'health statistics', 'mortality', 'morbidity', 'life expectancy', 'health disparities',
                'social determinants', 'environmental health', 'occupational health', 'infectious disease',
                'chronic disease', 'noncommunicable disease', 'vaccination', 'immunization',
                
                # General government/academic terms
                'government', 'federal', 'state', 'local', 'agency', 'department', 'office', 'bureau',
                'administration', 'service', 'program', 'initiative', 'campaign', 'guideline', 'recommendation',
                'policy', 'regulation', 'standard', 'protocol', 'procedure', 'practice', 'method',
                'approach', 'strategy', 'plan', 'framework', 'model', 'system', 'network',
                'organization', 'institution', 'establishment', 'facility', 'clinic', 'laboratory',
                'center', 'institute', 'academy', 'society', 'council', 'committee', 'board',
                'commission', 'task force', 'working group', 'advisory', 'expert', 'specialist',
                'professional', 'practitioner', 'provider', 'caregiver', 'therapist', 'counselor',
                
                # General health-related terms
                'wellness', 'wellbeing', 'lifestyle', 'behavior', 'habit', 'routine', 'activity',
                'exercise', 'workout', 'training', 'conditioning', 'strength', 'endurance', 'flexibility',
                'balance', 'coordination', 'mobility', 'stability', 'posture', 'movement', 'motion',
                'performance', 'capacity', 'ability', 'function', 'maintenance', 'improvement',
                'enhancement', 'optimization', 'maximization', 'development', 'growth', 'progress',
                'advancement', 'innovation', 'discovery', 'breakthrough', 'advance', 'improvement',
                
                # Common health page terms
                'information', 'resource', 'material', 'document', 'report', 'fact sheet', 'brochure',
                'guide', 'manual', 'handbook', 'toolkit', 'checklist', 'assessment', 'evaluation',
                'monitoring', 'tracking', 'measurement', 'analysis', 'review', 'summary', 'overview',
                'introduction', 'background', 'context', 'scope', 'purpose', 'objective', 'goal',
                'target', 'aim', 'intention', 'focus', 'priority', 'emphasis', 'importance',
                'significance', 'relevance', 'applicability', 'usefulness', 'value', 'benefit',
                'advantage', 'positive', 'good', 'better', 'best', 'optimal', 'ideal', 'recommended'
            ],
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
            self.logger.info("Targets met, stopping collection")
            raise CloseSpider('targets_met')
        
        domain = urlparse(response.url).netloc.replace("www.", "")
        
        # Skip if we've already processed this domain
        if domain in self.seen_domains:
            self.logger.debug(f"Skipping already seen domain: {domain}")
            return
        self.seen_domains.add(domain)

        # Skip the gold URL itself
        if response.url.rstrip('/') == self.gold_url.rstrip('/'):
            self.logger.info(f"Processing gold URL: {response.url}")
            # Still follow its links to find other sources
            for href in response.css("a::attr(href)").getall():
                if href and self._should_follow_link(href, response.url):
                    yield response.follow(href, callback=self.parse)
            return

        # Extract content
        title = self._extract_title(response)
        text = self._extract_text(response)
        
        # Debug content extraction
        self.logger.info(f"Domain {domain}: Title length: {len(title) if title else 0}, Text length: {len(text) if text else 0}")
        
        # Skip if content is too short
        if not text or len(text.strip()) < 100:
            self.logger.info(f"Domain {domain} rejected: Content too short ({len(text) if text else 0} chars)")
            return

        # Determine category
        category = self._categorize_source(domain, title, text)
        
        # Debug categorization result
        self.logger.info(f"Domain {domain} categorized as: {category}")
        
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
            yield article
            if self._collection_complete():
                self.logger.info("Targets met after adding article, stopping collection")
                raise CloseSpider('targets_met')
        else:
            self.logger.info(f"Domain {domain} rejected: Category {category} already has enough sources ({len(self.articles[category])})")

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
        
        # Debug text extraction
        self.logger.info(f"Text extraction for {response.url}: Raw length: {len(text)}, Clean length: {len(text.strip())}")
        if len(text.strip()) < 100:
            self.logger.info(f"Short text sample: '{text[:200]}...'")
        
        return text

    def _domain_matches(self, domain: str, candidates: set) -> bool:
        # exact match or subdomain-of
        return any(domain == d or domain.endswith("." + d) for d in candidates)
        
    def _categorize_source(self, domain, title, text):
        """Categorize source as reliable, unreliable, or offtopic"""
        
        content = f"{title} {text}".lower()
        
        # Check domain-based reliability FIRST (before topic relevance)
        if self._domain_matches(domain, self.reliable_domains):
            self.logger.info(f"Domain {domain} matched reliable domains list")
            return 'reliable'
        elif self._domain_matches(domain, self.unreliable_domains):
            self.logger.info(f"Domain {domain} matched unreliable domains list")
            return 'unreliable'
        
        # Only check topic relevance for unknown domains
        if not self._is_topic_relevant(content):
            self.logger.info(f"Domain {domain} marked as offtopic due to content relevance")
            return 'offtopic'
        
        # Check content-based reliability indicators for unknown domains
        reliable_score = sum(1 for keyword in self.reliability_indicators['reliable'] 
                           if keyword in content)
        unreliable_score = sum(1 for keyword in self.reliability_indicators['unreliable'] 
                             if keyword in content)
        
        self.logger.info(f"Domain {domain}: reliable_score={reliable_score}, unreliable_score={unreliable_score}")
        
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
            self.logger.info("No topic or keywords specified, considering everything relevant")
            return True  # If no topic specified, consider everything relevant
        
        # Check if any topic keywords appear in the content
        matched_keywords = [keyword for keyword in self.topic_keywords if keyword in content]
        
        # Debug topic relevance
        self.logger.info(f"Topic relevance check: Found {len(matched_keywords)} matching keywords: {matched_keywords[:5]}...")
        
        is_relevant = len(matched_keywords) > 0
        if not is_relevant:
            self.logger.info(f"Content failed topic relevance check. Sample content: '{content[:200]}...'")
        
        return is_relevant

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
        
        if any(pattern in href.lower() for pattern in skip_patterns):
            return False
        
        # For relative links, check if they might be health-related
        if href.startswith('/'):
            # Follow relative links on the same domain
            return True
        
        # For external links, be more selective
        try:
            parsed_href = urlparse(href)
            href_domain = parsed_href.netloc.replace("www.", "")
            
            # Skip obviously unrelated domains
            skip_domains = [
                'github.com', 'figma.com', 'atlassian.com', 'aws.amazon.com', 'linkedin.com'
            ]
            
            if any(skip_domain in href_domain for skip_domain in skip_domains):
                return False
            
            # Prefer health-related domains
            health_domains = [
                'health', 'medical', 'medicine', 'hospital', 'clinic',
                'doctor', 'patient', 'disease', 'treatment', 'therapy',
                'research', 'study', 'trial', 'clinical', 'academic',
                'edu', 'org'
            ]
            
            # If it contains health-related terms, follow it
            if any(term in href_domain.lower() for term in health_domains):
                return True
            
            # For other domains, be more conservative
            return False
            
        except:
            return False

    def _collection_complete(self):
        """Check if we have enough sources in each category"""
        current_counts = {k: len(self.articles[k]) for k in self.targets.keys()}
        is_complete = all(current_counts[k] >= v for k, v in self.targets.items())
        
        # Debug collection status
        self.logger.info(f"Collection status: {current_counts} (Targets: {self.targets}, Complete: {is_complete})")
        
        return is_complete
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
        paired = self._create_paired_sets()
        if paired:
            self.results_for_calmrag["clear_set"] = paired["clear_set"]["sources"]
            self.results_for_calmrag["ambigous_set"] = paired["ambigous_set"]["sources"]
        
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

        return paired_sets


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