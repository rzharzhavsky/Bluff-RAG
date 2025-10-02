import scrapy
from scrapy.exceptions import CloseSpider
import json
import random
from urllib.parse import urlparse, urljoin, parse_qs
import re
from datetime import datetime
from difflib import SequenceMatcher
import time

class SourceSpider(scrapy.Spider):
    name = "source_spider"
    
    # Settings with error handling -- edit if needed
    custom_settings = {
        'CONCURRENT_REQUESTS': 4,  # Reduced for better stability
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'DOWNLOAD_DELAY': 3,  # Increased delay
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 3,
        'AUTOTHROTTLE_MAX_DELAY': 15,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 0.8,
        'DOWNLOAD_TIMEOUT': 20,  # Increased timeout
        'RETRY_TIMES': 2,  # Reduced retries
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'HTTPERROR_ALLOW_ALL': False,
        'HTTPERROR_ALLOWED_CODES': [404],  # Only allow 404s
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': True,
        'COOKIES_ENABLED': False,
        'TELNETCONSOLE_ENABLED': False,
    }

    def __init__(self, gold_url=None, gold_question=None, topic=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold_url = gold_url
        self.gold_question = gold_question.lower() if gold_question else ""
        self.topic = topic.lower() if topic else ""
        
        # Extract keywords from the actual question
        self.question_keywords = self._extract_question_keywords()
        self.topic_keywords = self._get_topic_keywords(self.topic)
        
        # Generate starting URLs
        self.start_urls = self._generate_strategic_start_urls()
        
        self.results_for_bluffrag = {"clear_set": [], "ambiguous_set": []}
        self.targets = {"reliable": 6, "unreliable": 5}  # Slightly higher targets
        
        # Track domains and failed requests
        self.seen_domains = set()
        self.failed_domains = set()
        self.timeout_domains = set()
        
        # Store articles by category
        self.articles = {
            'reliable': [],
            'unreliable': [],
            'offtopic': []
        }
        
        self.target_per_category = 50
        self.max_total_articles = 200
        self.max_failed_requests = 20  # Stop after too many failures
        self.failed_request_count = 0
        
        # Load domain configuration
        try:
            with open("domain_trust_config.json", "r") as f:
                trust_config = json.load(f)
                topic_config = trust_config.get(self.topic, {})
                self.reliable_domains = set(topic_config.get("reliable", []))
                self.unreliable_domains = set(topic_config.get("misleading", []))
                self.logger.info(f"Loaded {len(self.reliable_domains)} reliable and {len(self.unreliable_domains)} unreliable domains")
        except FileNotFoundError:
            self.reliable_domains = set()
            self.unreliable_domains = set()
            self.logger.warning("domain_trust_config.json not found")
        
        # Reliability indicators with more specific terms
        self.reliability_indicators = {
            'reliable': [
                # Academic/Medical indicators
                'peer-reviewed', 'clinical trial', 'systematic review', 'meta-analysis',
                'randomized controlled', 'double-blind', 'placebo-controlled',
                'research study', 'academic study', 'medical research', 'clinical research',
                'published in', 'journal of', 'university of', 'institute of',
                'medical center', 'hospital', 'clinic', 'health department',
                'FDA approved', 'clinical evidence', 'scientific evidence',
                'peer review', 'methodology', 'statistics', 'data analysis',
                'evidence-based', 'cochrane review', 'pubmed', 'medline',
                # Government/Official sources
                'CDC', 'NIH', 'WHO', 'FDA', 'government', 'official',
                'department of health', 'public health', 'medical association',
                'board certified', 'licensed physician', 'medical degree'
            ],
            'unreliable': [
                # Alternative medicine red flags
                'big pharma conspiracy', 'doctors hate this', 'miracle cure',
                'natural cure they dont want', 'secret remedy', 'ancient wisdom',
                'pharmaceutical companies hide', 'mainstream medicine lies',
                'suppressed by', 'natural healing', 'detox', 'cleanse',
                'alternative medicine', 'holistic healing', 'energy healing',
                # Sensational language
                'shocking truth', 'exposed', 'revealed', 'hidden dangers',
                'cover-up', 'conspiracy', 'they don\'t want you to know',
                'breakthrough discovery', 'revolutionary', 'amazing results',
                # Anecdotal evidence
                'testimonial', 'personal experience', 'it worked for me',
                'my friend tried', 'I heard that', 'someone told me',
                'unverified report', 'rumor has it', 'word on the street'
            ]
        }

    def _extract_question_keywords(self):
        """Extract meaningful keywords from the question"""
        if not self.gold_question:
            return []
        
        # Stop words list
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'does', 'do', 'did', 'will', 'would', 'could', 'should', 'might', 'may',
            'can', 'must', 'shall', 'the', 'a', 'an', 'and', 'or', 'but', 'nor',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'very', 'too', 'more', 'most', 'such', 'no', 'not'
        }
        
        # Extract words and phrases
        words = re.findall(r'\b\w+\b', self.gold_question.lower())
        
        # Keep meaningful words (length > 2 and not stop words)
        keywords = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        # Also extract important phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:  # Avoid very short phrases
                    phrases.append(phrase)
        
        # Combine keywords and phrases
        all_terms = keywords + phrases
        
        self.logger.info(f"Extracted question terms: {all_terms[:10]}...")  # Show first 10
        return all_terms

    def _generate_strategic_start_urls(self):
        """Generate targeted starting URLs based on the question"""
        urls = []
        
        # Always include gold URL
        if self.gold_url:
            urls.append(self.gold_url)
        
        # Create search-based URLs for major health sites
        if self.question_keywords:
            # Use first few keywords for searches
            search_terms = " ".join(self.question_keywords[:3])
            encoded_terms = "+".join(self.question_keywords[:3])
            
            # Add direct searches on major health sites
            search_urls = [
                f"https://www.mayoclinic.org/search?q={encoded_terms}",
                f"https://www.webmd.com/search/search_results/default.aspx?query={encoded_terms}",
                f"https://www.healthline.com/search?q1={encoded_terms}",
                f"https://medlineplus.gov/search/?q={encoded_terms}",
                f"https://www.cdc.gov/search/?q={encoded_terms}",
            ]
            
            # Add some alternative medicine sites for unreliable sources
            alt_urls = [
                f"https://www.naturalnews.com/search.asp?query={encoded_terms}",
                f"https://www.mercola.com/search.aspx?q={encoded_terms}",
                f"https://www.greenmedinfo.com/search/site/{encoded_terms}",
            ]
            
            urls.extend(search_urls + alt_urls)
        
        # Add homepage fallbacks for crawling
        reliable_homepages = [
            "https://www.mayoclinic.org",
            "https://www.webmd.com", 
            "https://www.healthline.com",
            "https://medlineplus.gov",
            "https://www.nih.gov",
            "https://www.cdc.gov"
        ]
        
        unreliable_homepages = [
            "https://www.naturalnews.com",
            "https://www.mercola.com",
            "https://www.greenmedinfo.com"
        ]
        
        urls.extend(reliable_homepages + unreliable_homepages)
        
        self.logger.info(f"Generated {len(urls)} strategic URLs")
        return urls[:15]  # Limit to prevent too many starting points

    def _get_topic_keywords(self, topic):
        """Topic keywords"""
        topic_keyword_map = {
            'public_health': [
                # Medical terms
                'medical', 'medicine', 'health', 'healthcare', 'treatment', 'therapy',
                'disease', 'illness', 'condition', 'syndrome', 'disorder', 'infection',
                'symptoms', 'diagnosis', 'prognosis', 'prevention', 'cure', 'recovery',
                'patient', 'doctor', 'physician', 'nurse', 'hospital', 'clinic',
                # Research terms
                'research', 'study', 'trial', 'clinical', 'scientific', 'evidence',
                'data', 'statistics', 'analysis', 'results', 'findings', 'conclusion',
                'peer-reviewed', 'published', 'journal', 'academic', 'university',
                # Nutrition/supplements
                'nutrition', 'vitamin', 'supplement', 'mineral', 'nutrient', 'diet',
                'dietary', 'food', 'eating', 'consumption', 'intake', 'dosage',
                # Public health
                'public health', 'epidemiology', 'population', 'community', 'wellness',
                'fitness', 'exercise', 'lifestyle', 'behavior', 'risk factor'
            ]
        }
        
        return topic_keyword_map.get(topic, [topic]) if topic else []

    def parse(self, response):
        """Parsing with error handling and relevance checking"""
        
        # Handle failed requests
        if response.status >= 400:
            self.failed_request_count += 1
            domain = urlparse(response.url).netloc.replace("www.", "")
            self.failed_domains.add(domain)
            self.logger.warning(f"Failed request to {domain}: HTTP {response.status}")
            
            if self.failed_request_count >= self.max_failed_requests:
                self.logger.error("Too many failed requests, stopping")
                raise CloseSpider('too_many_failures')
            return
        
        # Check completion
        if self._collection_complete():
            raise CloseSpider('targets_met')
        
        domain = urlparse(response.url).netloc.replace("www.", "")
        
        # Skip problematic domains
        if domain in self.failed_domains or domain in self.timeout_domains:
            return
        
        # Skip already processed domains (but allow a few retries for different pages)
        domain_count = sum(1 for article in sum(self.articles.values(), []) if article['domain'] == domain)
        if domain_count >= 3:  # Max 3 articles per domain
            return
        
        # Skip gold URL itself but follow links
        if response.url.rstrip('/') == self.gold_url.rstrip('/'):
            yield from self._follow_strategic_links(response)
            return

        # Content extraction
        title = self._extract_title(response)
        text = self._extract_text_advanced(response)
        
        # Early content quality check
        if not self._is_content_quality_sufficient(title, text):
            self.logger.debug(f"Low quality content from {domain}")
            yield from self._follow_strategic_links(response)
            return

        # Relevance scoring
        relevance_score = self._score_question_relevance_advanced(title, text)
        
        self.logger.info(f"{domain}: Relevance score: {relevance_score:.3f}")
        
        # Higher relevance threshold
        if relevance_score < 0.25:
            self.logger.debug(f"Low relevance for {domain}: {relevance_score:.3f}")
            yield from self._follow_strategic_links(response)
            return

        # Categorize source
        category = self._categorize_source(domain, title, text, relevance_score)
        
        # Add to collection if needed
        if len(self.articles[category]) < self.target_per_category:
            article = {
                "url": response.url,
                "domain": domain,
                "category": category,
                "title": title,
                "text": text[:1500],  # Keep more text
                "relevance_score": relevance_score,
                "timestamp": datetime.now().isoformat()
            }
            
            self.articles[category].append(article)
            self.logger.info(f"✓ Added {category}: {domain} (score: {relevance_score:.3f})")
            yield article
        
        # Continue crawling
        if not self._collection_complete():
            yield from self._follow_strategic_links(response)

    def _is_content_quality_sufficient(self, title, text):
        """Check if content meets minimum quality standards"""
        if not title or not text:
            return False
        
        # Minimum length requirements
        if len(text.strip()) < 200:
            return False
        
        # Check for actual sentences (not just navigation text)
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
        
        if len(meaningful_sentences) < 3:
            return False
        
        # Avoid pages that are mostly navigation/menu items
        navigation_indicators = [
            'home', 'about', 'contact', 'privacy', 'terms', 'sitemap',
            'navigation', 'menu', 'footer', 'header', 'sidebar',
            'copyright', 'all rights reserved', 'cookies policy'
        ]
        
        nav_count = sum(1 for indicator in navigation_indicators if indicator in text.lower())
        if nav_count > len(meaningful_sentences) / 2:  # Too much navigation text
            return False
        
        return True

    def _extract_text_advanced(self, response):
        """Advanced text extraction with multiple strategies"""
        
        # Strategy 1: Look for main content areas
        main_content_selectors = [
            '//article[1]//text()[not(ancestor::script or ancestor::style or ancestor::nav or ancestor::header or ancestor::footer)]',
            '//main//text()[not(ancestor::script or ancestor::style or ancestor::nav)]',
            '//*[contains(@class, "content") or contains(@class, "article") or contains(@class, "post")]//text()[not(ancestor::script or ancestor::style)]',
            '//*[contains(@id, "content") or contains(@id, "article") or contains(@id, "main")]//text()[not(ancestor::script or ancestor::style)]',
        ]
        
        for selector in main_content_selectors:
            text_parts = response.xpath(selector).getall()
            if text_parts:
                text = " ".join(text_parts)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 200:  # Found substantial content
                    return text
        
        # Strategy 2: Fallback to paragraph extraction
        paragraphs = response.xpath('//p//text()[not(ancestor::script or ancestor::style)]').getall()
        if paragraphs:
            text = " ".join(paragraphs)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Strategy 3: Last resort - all visible text
        all_text = response.xpath('//text()[not(ancestor::script or ancestor::style or ancestor::nav)]').getall()
        text = " ".join(all_text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _score_question_relevance_advanced(self, title, text):
        """Advanced relevance scoring with multiple factors"""
        if not self.question_keywords:
            return 0.5
        
        content = f"{title} {text}".lower()
        
        # Factor 1: Direct keyword matches (weighted by importance)
        keyword_scores = []
        for keyword in self.question_keywords:
            if isinstance(keyword, str) and ' ' not in keyword:  # Single word
                count = content.count(keyword)
                # Boost for title appearances
                title_boost = 2 if keyword in title.lower() else 1
                keyword_scores.append(min(count * title_boost * 0.1, 1.0))
            elif isinstance(keyword, str):  # Phrase
                if keyword in content:
                    keyword_scores.append(0.3)  # Phrases get fixed boost
                else:
                    keyword_scores.append(0.0)
        
        keyword_score = sum(keyword_scores) / len(self.question_keywords) if keyword_scores else 0
        
        # Factor 2: Question phrase similarity
        question_phrase = " ".join(str(k) for k in self.question_keywords[:4])
        content_sample = content[:800]  # Larger sample
        similarity = SequenceMatcher(None, question_phrase, content_sample).ratio()
        
        # Factor 3: Topic relevance boost
        topic_score = 0
        if self.topic_keywords:
            topic_matches = sum(1 for keyword in self.topic_keywords if keyword in content)
            topic_score = min(topic_matches / len(self.topic_keywords), 0.3)
        
        # Factor 4: Content quality indicators
        quality_indicators = [
            'research', 'study', 'evidence', 'data', 'analysis', 'results',
            'clinical', 'trial', 'published', 'journal', 'scientific'
        ]
        quality_matches = sum(1 for indicator in quality_indicators if indicator in content)
        quality_score = min(quality_matches * 0.05, 0.2)
        
        # Combine all factors
        final_score = (
            keyword_score * 0.5 +
            similarity * 0.25 +
            topic_score * 0.15 +
            quality_score * 0.1
        )
        
        return min(final_score, 1.0)

    def _categorize_source(self, domain, title, text, relevance_score):
        """Source categorization"""
        content = f"{title} {text}".lower()
        
        # Minimum relevance for any category except offtopic
        if relevance_score < 0.2:
            return 'offtopic'
        
        # Domain-based classification (highest priority)
        if self._domain_matches(domain, self.reliable_domains):
            return 'reliable'
        elif self._domain_matches(domain, self.unreliable_domains):
            return 'unreliable'
        
        # Topic relevance check
        if not self._is_topic_relevant(content):
            return 'offtopic'
        
        # Content-based reliability analysis
        reliable_score = 0
        unreliable_score = 0
        
        # Score based on indicators
        for indicator in self.reliability_indicators['reliable']:
            if indicator in content:
                # Weight by length - longer phrases are more significant
                weight = len(indicator.split()) * 0.5 + 0.5
                reliable_score += weight
        
        for indicator in self.reliability_indicators['unreliable']:
            if indicator in content:
                weight = len(indicator.split()) * 0.5 + 0.5
                unreliable_score += weight
        
        # Domain-based hints for unknown domains
        reliable_domain_hints = ['.edu', '.gov', '.org', 'university', 'college', 'hospital', 'clinic']
        unreliable_domain_hints = ['blog', 'wordpress', 'tumblr', 'personal', 'alternative']
        
        domain_reliable_score = sum(1 for hint in reliable_domain_hints if hint in domain)
        domain_unreliable_score = sum(1 for hint in unreliable_domain_hints if hint in domain)
        
        total_reliable = reliable_score + domain_reliable_score
        total_unreliable = unreliable_score + domain_unreliable_score
        
        self.logger.debug(f"{domain}: reliable_score={total_reliable:.1f}, unreliable_score={total_unreliable:.1f}")
        
        # Classification logic
        if total_reliable > total_unreliable * 1.5:  # Clear reliable indication
            return 'reliable'
        elif total_unreliable > total_reliable * 1.2:  # Clear unreliable indication
            return 'unreliable'
        else:
            # For ambiguous cases, use relevance as tiebreaker
            # High relevance unreliable content is valuable for contrast
            if relevance_score > 0.4:
                return 'unreliable'  # Assume unreliable for high-relevance ambiguous content
            else:
                return 'reliable'  # Default to reliable for moderate relevance

    def _is_topic_relevant(self, content):
        """Topic relevance checking"""
        if not self.topic_keywords:
            return True
        
        # Count keyword matches with partial matching
        matches = 0
        for keyword in self.topic_keywords:
            if keyword in content:
                matches += 1
            # Also check for partial matches (root words)
            elif len(keyword) > 5:
                root = keyword[:5]
                if root in content:
                    matches += 0.5
        
        # Need at least 2 matches for health topics (they're broad)
        required_matches = 2 if self.topic == 'public_health' else 1
        return matches >= required_matches

    def _follow_strategic_links(self, response):
        """Strategic link following"""
        links_followed = 0
        all_links = []
        
        # Extract all links with context
        for link in response.css('a'):
            href = link.xpath('./@href').get()
            anchor_text = link.xpath('.//text()').getall()
            anchor_text = ' '.join(anchor_text).strip()
            
            if href and self._should_follow_link(href, anchor_text, response.url):
                relevance = self._score_link_relevance(href, anchor_text)
                all_links.append((href, relevance, anchor_text))
        
        # Sort by relevance and follow top links
        all_links.sort(key=lambda x: x[1], reverse=True)
        
        for href, relevance, anchor_text in all_links[:8]:  # Top 8 links
            yield response.follow(href, callback=self.parse)
            links_followed += 1
        
        self.logger.debug(f"Following {links_followed} strategic links from {response.url}")

    def _should_follow_link(self, href, anchor_text, base_url):
        """Link filtering"""
        if not href or not href.startswith(('http', '/')):
            return False
        
        # Expanded skip patterns
        skip_patterns = [
            # File types
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.css', '.js', '.xml', '.json',
            # Navigation
            '/tag/', '/category/', '/author/', '/search/', '/login/', '/register/',
            '/subscribe/', '/newsletter/', '/contact/', '/about/', '/privacy/',
            '/terms/', '/sitemap/', '/archive/', '/feed/', '/rss/',
            # Social/External
            'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
            'linkedin.com', 'pinterest.com', 'reddit.com',
            # Other
            'mailto:', 'tel:', '#', 'javascript:'
        ]
        
        href_lower = href.lower()
        if any(pattern in href_lower for pattern in skip_patterns):
            return False
        
        # For external links, be more selective
        if href.startswith('http'):
            try:
                parsed = urlparse(href)
                domain = parsed.netloc.replace('www.', '')
                
                # Skip known problematic domains
                if domain in self.failed_domains or domain in self.timeout_domains:
                    return False
                
                # For health topics, prefer health-related domains
                health_domains = [
                    'nih.gov', 'cdc.gov', 'who.int', 'mayoclinic.org', 'webmd.com',
                    'healthline.com', 'medlineplus.gov', 'pubmed.ncbi.nlm.nih.gov',
                    'naturalnews.com', 'mercola.com', 'greenmedinfo.com'
                ]
                
                if not any(health_domain in domain for health_domain in health_domains):
                    # For non-health domains, require very high relevance in anchor text
                    if not any(keyword in anchor_text.lower() for keyword in self.question_keywords[:5]):
                        return False
                
            except:
                return False
        
        return True

    def _score_link_relevance(self, href, anchor_text):
        """Link relevance scoring"""
        if not self.question_keywords:
            return 0.3
        
        link_text = f"{href} {anchor_text}".lower()
        
        # Keyword matching in link
        keyword_score = 0
        for keyword in self.question_keywords[:8]:  # Top 8 keywords
            if str(keyword) in link_text:
                # Boost for anchor text vs URL
                if str(keyword) in anchor_text.lower():
                    keyword_score += 0.15
                else:
                    keyword_score += 0.1
        
        # Content type indicators
        content_boosts = {
            'article': 0.2, 'research': 0.25, 'study': 0.25, 'health': 0.15,
            'medical': 0.15, 'news': 0.1, 'blog': 0.05, 'post': 0.05,
            'guide': 0.1, 'info': 0.05, 'fact': 0.15, 'evidence': 0.2
        }
        
        content_score = sum(boost for term, boost in content_boosts.items() 
                           if term in link_text)
        
        final_score = min(keyword_score + content_score, 1.0)
        return final_score

    def _extract_title(self, response):
        """Title extraction"""
        # Try multiple title sources
        title_selectors = [
            '//title/text()',
            '//h1//text()',
            '//h2//text()',
            '//*[@class="title" or @class="headline"]//text()',
            '//*[@id="title" or @id="headline"]//text()'
        ]
        
        for selector in title_selectors:
            title = response.xpath(selector).get()
            if title and len(title.strip()) > 5:
                return title.strip()
        
        return "No Title Found"

    def _domain_matches(self, domain: str, candidates: set) -> bool:
        """Domain matching"""
        return any(domain == d or domain.endswith("." + d) or d in domain for d in candidates)
        
    def _collection_complete(self):
        """Check collection completion"""
        current_counts = {k: len(self.articles[k]) for k in self.targets.keys()}
        is_complete = all(current_counts[k] >= v for k, v in self.targets.items())
        
        total_articles = sum(current_counts.values())
        self.logger.info(f"Collection: {current_counts} | Targets: {self.targets} | Complete: {is_complete}")
        
        # Also stop if we have too many total articles
        if total_articles >= self.max_total_articles:
            self.logger.info("Max total articles reached")
            return True
            
        return is_complete

    def closed(self, reason):
        """Closing with reporting"""
        
        # Compile all sources
        all_sources = []
        for category, articles in self.articles.items():
            all_sources.extend(articles)
        
        # Sort by relevance
        all_sources.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        scraped_data = {
            "gold_url": self.gold_url,
            "gold_question": self.gold_question,
            "topic": self.topic,
            "question_keywords": self.question_keywords[:10],  # Limit for readability
            "collection_stats": {
                "reliable": len(self.articles['reliable']),
                "unreliable": len(self.articles['unreliable']),
                "offtopic": len(self.articles['offtopic']),
                "total": len(all_sources),
                "failed_domains": len(self.failed_domains),
                "timeout_domains": len(self.timeout_domains)
            },
            "sources": all_sources,
            "failed_domains": list(self.failed_domains),
            "completion_reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("scraped_sources.json", "w") as f:
            json.dump(scraped_data, f, indent=2)
        
        # Create paired sets
        paired = self._create_paired_sets()
        if paired:
            self.results_for_bluffrag["clear_set"] = paired["clear_set"]["sources"]
            self.results_for_bluffrag["ambiguous_set"] = paired["ambiguous_set"]["sources"]
            
            with open("paired_sets.json", "w") as f:
                json.dump(paired, f, indent=2)
        
        # Reporting
        self._print_collection_summary()

    def _create_paired_sets(self):
        """Paired set creation with quality control"""
        
        # Filter for high-relevance sources only
        min_relevance = 0.3
        reliable_sources = [s for s in self.articles['reliable'] if s.get('relevance_score', 0) >= min_relevance]
        unreliable_sources = [s for s in self.articles['unreliable'] if s.get('relevance_score', 0) >= min_relevance]
        
        self.logger.info(f"High-relevance sources: Reliable={len(reliable_sources)}, Unreliable={len(unreliable_sources)}")
        
        # Fallback to lower threshold if needed
        if len(reliable_sources) < 5 or len(unreliable_sources) < 4:
            min_relevance = 0.2
            reliable_sources = [s for s in self.articles['reliable'] if s.get('relevance_score', 0) >= min_relevance]
            unreliable_sources = [s for s in self.articles['unreliable'] if s.get('relevance_score', 0) >= min_relevance]
            self.logger.warning(f"Lowered relevance threshold to {min_relevance}")
        
        # Final fallback
        if len(reliable_sources) < 5 or len(unreliable_sources) < 4:
            reliable_sources = self.articles['reliable']
            unreliable_sources = self.articles['unreliable']
            self.logger.warning("Using all sources regardless of relevance")
        
        if len(reliable_sources) < 5 or len(unreliable_sources) < 4:
            self.logger.error("Insufficient sources for paired set creation")
            return None
        
        # Sort by relevance score (highest first)
        reliable_sources.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        unreliable_sources.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Ensure domain diversity in selection
        reliable_selected = self._select_diverse_sources(reliable_sources, 5)
        unreliable_selected = self._select_diverse_sources(unreliable_sources, 4)
        
        # Create clear set: 4 reliable + 1 unreliable (highest relevance)
        clear_reliable = reliable_selected[:4]
        clear_unreliable = unreliable_selected[:1]
        
        # Create ambiguous set: 1 reliable + 3 unreliable
        remaining_reliable = [s for s in reliable_selected if s not in clear_reliable]
        remaining_unreliable = [s for s in unreliable_selected if s not in clear_unreliable]
        
        ambiguous_reliable = remaining_reliable[:1] if remaining_reliable else reliable_selected[4:5]
        ambiguous_unreliable = remaining_unreliable[:3] if len(remaining_unreliable) >= 3 else unreliable_selected[1:4]
        
        clear_set = clear_reliable + clear_unreliable
        ambiguous_set = ambiguous_reliable + ambiguous_unreliable
        
        paired_sets = {
            "gold_url": self.gold_url,
            "gold_question": self.gold_question,
            "topic": self.topic,
            "clear_set": {
                "description": "4 reliable sources + 1 unreliable source (highest relevance)",
                "sources": clear_set,
                "avg_relevance": sum(s.get('relevance_score', 0) for s in clear_set) / len(clear_set),
                "composition": {"reliable": 4, "unreliable": 1}
            },
            "ambiguous_set": {
                "description": "1 reliable source + 3 unreliable sources", 
                "sources": ambiguous_set,
                "avg_relevance": sum(s.get('relevance_score', 0) for s in ambiguous_set) / len(ambiguous_set),
                "composition": {"reliable": 1, "unreliable": 3}
            },
            "selection_criteria": {
                "min_relevance_threshold": min_relevance,
                "domain_diversity_enforced": True,
                "sorting": "by_relevance_score_desc"
            },
            "created_at": datetime.now().isoformat()
        }
        
        self.logger.info(f"Paired sets created:")
        self.logger.info(f"  Clear set avg relevance: {paired_sets['clear_set']['avg_relevance']:.3f}")
        self.logger.info(f"  Ambiguous set avg relevance: {paired_sets['ambiguous_set']['avg_relevance']:.3f}")
        
        return paired_sets

    def _select_diverse_sources(self, sources, count):
        """Select sources with domain diversity preference"""
        if len(sources) <= count:
            return sources
        
        selected = []
        used_domains = set()
        
        # First pass: select highest relevance from unique domains
        for source in sources:
            if len(selected) >= count:
                break
            if source['domain'] not in used_domains:
                selected.append(source)
                used_domains.add(source['domain'])
        
        # Second pass: fill remaining slots with highest relevance regardless of domain
        remaining_needed = count - len(selected)
        if remaining_needed > 0:
            remaining_sources = [s for s in sources if s not in selected]
            selected.extend(remaining_sources[:remaining_needed])
        
        return selected[:count]

    def _print_collection_summary(self):
        """Print detailed collection summary"""
        self.logger.info("="*60)
        self.logger.info("COLLECTION SUMMARY")
        self.logger.info("="*60)
        
        for category in ['reliable', 'unreliable', 'offtopic']:
            articles = self.articles[category]
            if articles:
                avg_relevance = sum(a.get('relevance_score', 0) for a in articles) / len(articles)
                top_domains = {}
                for article in articles:
                    domain = article['domain']
                    top_domains[domain] = top_domains.get(domain, 0) + 1
                
                self.logger.info(f"\n{category.upper()} ({len(articles)} articles):")
                self.logger.info(f"  Average relevance: {avg_relevance:.3f}")
                self.logger.info(f"  Top domains: {dict(list(sorted(top_domains.items(), key=lambda x: x[1], reverse=True))[:5])}")
                
                if articles:
                    best = max(articles, key=lambda x: x.get('relevance_score', 0))
                    self.logger.info(f"  Best source: {best['domain']} (relevance: {best.get('relevance_score', 0):.3f})")
        
        if self.failed_domains:
            self.logger.info(f"\nFailed domains ({len(self.failed_domains)}): {list(self.failed_domains)[:10]}")
        
        self.logger.info("="*60)