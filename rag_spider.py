import scrapy
import json
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import hashlib
import pandas as pd
import re
import csv

class RAGBenchmarkSpider(scrapy.Spider):
    name = 'rag_benchmark'
    
    def __init__(self, *args, **kwargs):
        super(RAGBenchmarkSpider, self).__init__(*args, **kwargs)
        
        # Initialize data containers
        self.reliable_sources = []
        self.unreliable_sources = []
        self.ambiguous_sources = []
        self.wrong_sources = []
        
        # Progress tracking
        self.pages_processed = 0
        self.save_interval = 50  # Save every 50 pages
        
        # Domain reliability mapping
        self.domain_reliability = {
            # Highly reliable sources
            'reliable': [
                # Public Health
                'who.int', 'cdc.gov', 'nih.gov', 'fda.gov', 'pubmed.ncbi.nlm.nih.gov',
                'bmj.com', 'thelancet.com', 'nejm.org', 'cochrane.org',
                # Finance
                'federalreserve.gov', 'sec.gov', 'reuters.com/business', 'bloomberg.com',
                'wsj.com', 'ft.com', 'morningstar.com', 'imf.org',
                # Current World Events
                'reuters.com', 'bbc.com', 'apnews.com', 'npr.org', 'pbs.org',
                'economist.com', 'foreignaffairs.com', 'cfr.org',
                # Historic Events
                'loc.gov', 'nara.gov', 'smithsonianmag.com', 'jstor.org',
                'history.com', 'britannica.com', 'nationalgeographic.com',
                # Sports
                'espn.com', 'si.com', 'nfl.com', 'nba.com', 'mlb.com',
                'fifa.com', 'olympics.com', 'usatoday.com/sports'
            ],
            # Unreliable sources
            'unreliable': [
                # Public Health misinformation
                'naturalnews.com', 'mercola.com', 'healthimpactnews.com',
                # Finance conspiracy/pump&dump
                'zerohedge.com', 'goldsilver.com', 'silverdoctors.com',
                # Current events conspiracy/fake news
                'infowars.com', 'breitbart.com', 'rt.com', 'sputniknews.com',
                'beforeitsnews.com', 'activistpost.com',
                # Historic revisionism
                'stormfront.org', 'vdare.com', 'americanrenaissance.com',
                # Sports tabloids/unreliable
                'tmz.com', 'thesun.co.uk', 'nypost.com/sports'
            ],
            # Ambiguous sources (could go either way)
            'ambiguous': [
                # Public Health - popular but not peer-reviewed
                'healthline.com', 'webmd.com', 'mayoclinic.org', 'medicalnewstoday.com',
                # Finance - popular/opinion based
                'yahoo.com/finance', 'marketwatch.com', 'fool.com', 'cnbc.com',
                'investopedia.com', 'seekingalpha.com',
                # Current events - mainstream but opinion-heavy
                'cnn.com', 'foxnews.com', 'msnbc.com', 'politico.com',
                'huffpost.com', 'buzzfeed.com', 'vox.com',
                # Historic events - popular sources
                'wikipedia.org', 'thoughtco.com', 'livescience.com',
                # Sports - opinion/entertainment focused
                'bleacherreport.com', 'deadspin.com', 'theringer.com',
                'sbnation.com', 'espn.com/fantasy'
            ],
            # Sources with known misinformation
            'wrong': [
                # Public Health - dangerous misinformation
                'vaccinechoice.com', 'thehealthwyze.com', 'greenmedinfo.com',
                # Finance - scams/ponzi promotion
                'bitcoinmagazine.com', 'coindesk.com', 'cryptonews.com',
                # Current events - conspiracy theories
                'wakingtimes.com', 'davidicke.com', 'prisonplanet.com',
                'globalresearch.ca', 'thetruthaboutcancer.com',
                # Historic events - holocaust denial/conspiracy
                'ihr.org', 'codoh.com', 'zundelsite.org',
                # Sports - fake injury reports/match fixing claims
                'sportsmole.co.uk', 'givemesport.com'
            ]
        }
        
        # Keywords that might indicate reliability level
        self.reliability_keywords = {
            'reliable': [
                # Public Health
                'peer-reviewed', 'clinical trial', 'systematic review', 'meta-analysis',
                'randomized controlled', 'double-blind', 'placebo-controlled',
                # Finance
                'SEC filing', 'audited financial', 'quarterly report', 'peer review',
                'regulatory filing', 'official statement', 'verified data',
                # Current/Historic Events
                'primary source', 'eyewitness account', 'official document',
                'government record', 'archived document', 'verified fact',
                # Sports
                'official league', 'verified statistics', 'game footage',
                'official announcement', 'league statement'
            ],
            'unreliable': [
                # Public Health
                'miracle cure', 'big pharma conspiracy', 'natural remedy',
                'doctors hate this', 'suppressed truth',
                # Finance
                'get rich quick', 'guaranteed returns', 'insider secret',
                'market manipulation', 'wall street conspiracy',
                # Current/Historic Events
                'mainstream media lies', 'government cover-up', 'hidden truth',
                'they dont want you to know', 'false flag',
                # Sports
                'inside scoop', 'locker room sources', 'unnamed insider',
                'exclusive rumor', 'behind the scenes drama'
            ],
            'ambiguous': [
                # General
                'some experts say', 'studies suggest', 'preliminary research',
                'early findings', 'further research needed', 'analysts believe',
                'sources close to', 'industry insiders', 'reports indicate',
                'speculation suggests', 'rumors suggest', 'unconfirmed reports'
            ],
            'wrong': [
                # Public Health
                'vaccines cause autism', 'covid is fake', '5g causes cancer',
                'chemtrails', 'population control', 'microchip implants',
                # Finance
                'federal reserve conspiracy', 'gold standard return', 'currency collapse',
                'illuminati controls markets', 'jewish bankers conspiracy',
                # Current/Historic Events
                'flat earth', 'moon landing fake', 'holocaust never happened',
                'lizard people', '9/11 inside job', 'sandy hook hoax',
                # Sports
                'all games are rigged', 'illuminati controls sports',
                'athletes are clones', 'steroids in water supply'
            ]
        }
    def splice_url(self, url):
            if not url or not isinstance(url, str):
                return None
            
            pattern = r'^(https?://[^/]+\.(com|org|co\.uk|blog|tv|net|us))'
            match = re.match(pattern, url)
            
            if match:
                return match.group(1)
            else:
                return None

    def load_unreliable(self): # returns list of unreliable source links
        unreliable_sources = []
        
        with open('source_lists/iffy_unreliable.csv', 'r', newline='') as csv_file: # Iffy
            iffy_csv_reader = csv.reader(csv_file)
            next(iffy_csv_reader)
            for row in iffy_csv_reader:
                unreliable_sources.append(row[0])
        
        df = pd.read_csv("source_lists/ad_fontes_media_bias.csv") # Ad Fontes

        df = df[df["Url"].notna() & (df["Url"] != "")] # Removes TV show entries without URLs
        df["Base_Url"] = df["Url"].apply(self.splice_url) # Takes cols of article links and changes to homepage links

        avg_scores = df.groupby("Source")[["Bias", "Quality"]].mean().reset_index()
        avg_scores["Bias"] = avg_scores["Bias"].round(2)
        avg_scores["Quality"] = avg_scores["Quality"].round(2)
        avg_scores["Base_Url"] = avg_scores["Source"].map(df.groupby("Source")["Base_Url"].first()) # Takes first link
        
        for _, row in avg_scores.iterrows():
            if row["Quality"] <= 16: # Quality of information is poor, bias is irrelevant in this scenario
                unreliable_sources.append(row["Base_Url"])
        return unreliable_sources
    
    def load_reliable(self): # returns list of reliable source links
        reliable_sources = []

        df = pd.read_csv("source_lists/ad_fontes_media_bias.csv") # Ad Fontes

        df = df[df["Url"].notna() & (df["Url"] != "")] # Removes TV show entries without URLs
        df["Base_Url"] = df["Url"].apply(self.splice_url)
        
        avg_scores = df.groupby("Source")[["Bias", "Quality"]].mean().reset_index()
        avg_scores["Bias"] = avg_scores["Bias"].round(2)
        avg_scores["Quality"] = avg_scores["Quality"].round(2)
        avg_scores["Base_Url"] = avg_scores["Source"].map(df.groupby("Source")["Base_Url"].first())
        
        for _, row in avg_scores.iterrows():
            if row["Quality"] >= 40 and -10 <= row["Bias"] <= 10: # Quality of information is high, balanced or slightly skewed bias
                reliable_sources.append(row["Base_Url"])
        return reliable_sources
    
    def load_ambiguous(self): # list of ambiguous source links
        ambiguous_sources = []

        df = pd.read_csv("source_lists/ad_fontes_media_bias.csv") # Ad Fontes

        df = df[df["Url"].notna() & (df["Url"] != "")] # Removes TV show entries without URLs
        df["Base_Url"] = df["Url"].apply(self.splice_url)
        
        avg_scores = df.groupby("Source")[["Bias", "Quality"]].mean().reset_index()
        avg_scores["Bias"] = avg_scores["Bias"].round(2)
        avg_scores["Quality"] = avg_scores["Quality"].round(2)
        avg_scores["Base_Url"] = avg_scores["Source"].map(df.groupby("Source")["Base_Url"].first())
        
        for _, row in avg_scores.iterrows():
            if 24 <= row["Quality"] < 40 and -24 <= row["Bias"] <= 24: # Quality of information varies, opinion-based writing, may be biased
                ambiguous_sources.append(row["Base_Url"])
        return ambiguous_sources

    def start_requests(self):
        """Define starting URLs for different types of sources"""
        
        start_urls = [
            # Public Health - Reliable
            'https://www.who.int/news',
            'https://www.cdc.gov/media/releases/',
            'https://www.nih.gov/news-events',
            'https://www.bmj.com/',
            
            # Public Health - Ambiguous
            'https://www.healthline.com/health-news',
            'https://www.webmd.com/news/',
            'https://www.mayoclinic.org/healthy-lifestyle',
            
            # Finance - Reliable
            'https://www.reuters.com/business/',
            'https://www.bloomberg.com/economics',
            'https://www.wsj.com/news/markets',
            'https://www.ft.com/markets',
            
            # Finance - Ambiguous
            'https://finance.yahoo.com/news/',
            'https://www.marketwatch.com/newsviewer',
            'https://www.cnbc.com/finance/',
            'https://www.investopedia.com/news/',
            
            # Current World Events - Reliable
            'https://www.reuters.com/world/',
            'https://www.bbc.com/news/world',
            'https://www.apnews.com/hub/world-news',
            'https://www.npr.org/sections/news/',
            
            # Current World Events - Ambiguous
            'https://www.cnn.com/world',
            'https://www.politico.com/news',
            'https://www.vox.com/world',
            
            # Historic Events - Reliable
            'https://www.loc.gov/collections/',
            'https://www.smithsonianmag.com/history/',
            'https://www.nationalgeographic.com/history/',
            'https://www.britannica.com/topic/history',
            
            # Historic Events - Ambiguous
            'https://en.wikipedia.org/wiki/Portal:History',
            'https://www.history.com/topics',
            'https://www.thoughtco.com/history-4133356',
            
            # Sports - Reliable
            'https://www.espn.com/nfl/',
            'https://www.si.com/',
            'https://www.nfl.com/news/',
            'https://www.nba.com/news/',
            'https://www.mlb.com/news',
            
            # Sports - Ambiguous
            'https://bleacherreport.com/',
            'https://www.theringer.com/sports',
            'https://www.sbnation.com/',
            
            # Note: Intentionally not including unreliable/wrong sources
            # Add them manually if needed for benchmarking purposes
        ] + self.load_reliable() + self.load_ambiguous()
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        """Main parsing method"""
        
        # Extract basic page info
        title = self.extract_title(response)
        url = response.url
        date = self.extract_date(response)
        text = self.extract_text(response)
        
        if not text or len(text.strip()) < 50:  # Skip if no meaningful content
            return
        
        # Determine reliability category
        reliability = self.categorize_reliability(url, title, text)
        
        # Create source excerpt
        source_excerpt = {
            "title": title,
            "url": url,
            "date": date,
            "text": text[:500] + "..." if len(text) > 500 else text  # Truncate long texts
        }
        
        # Add to appropriate category
        if reliability == 'reliable':
            self.reliable_sources.append(source_excerpt)
        elif reliability == 'unreliable':
            self.unreliable_sources.append(source_excerpt)
        elif reliability == 'ambiguous':
            self.ambiguous_sources.append(source_excerpt)
        elif reliability == 'wrong':
            self.wrong_sources.append(source_excerpt)
        
        # Increment counter and save periodically
        self.pages_processed += 1
        if self.pages_processed % self.save_interval == 0:
            self.save_progress()
            print(f"Progress saved: {self.pages_processed} pages processed")
        
        # Follow links to gather more content
        for link in response.css('a::attr(href)').getall():
            if link:
                absolute_url = urljoin(response.url, link)
                if self.should_follow_link(absolute_url):
                    yield scrapy.Request(url=absolute_url, callback=self.parse)

    def extract_title(self, response):
        """Extract page title"""
        title = response.css('title::text').get()
        if not title:
            title = response.css('h1::text').get()
        return title.strip() if title else "No Title"

    def extract_date(self, response):
        """Extract publication date"""
        # Try various date selectors
        date_selectors = [
            'meta[property="article:published_time"]::attr(content)',
            'meta[name="date"]::attr(content)',
            'meta[name="publish-date"]::attr(content)',
            '.date::text',
            '.published::text',
            'time::attr(datetime)'
        ]
        
        for selector in date_selectors:
            date = response.css(selector).get()
            if date:
                try:
                    # Parse and format date
                    if 'T' in date:
                        parsed_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    else:
                        parsed_date = datetime.strptime(date, '%Y-%m-%d')
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
        
        return datetime.now().strftime('%Y-%m-%d')  # Fallback to current date

    def extract_text(self, response):
        """Extract main text content"""
        # Try to get main content, avoiding navigation and ads
        content_selectors = [
            'article',
            '.content',
            '.article-body',
            '.post-content',
            'main',
            '.entry-content'
        ]
        
        text = ""
        for selector in content_selectors:
            elements = response.css(selector)
            if elements:
                text = ' '.join(elements.css('::text').getall())
                break
        
        if not text:
            # Fallback to all paragraph text
            text = ' '.join(response.css('p::text').getall())
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def categorize_reliability(self, url, title, text):
        """Categorize source reliability based on domain, keywords, and content"""
        
        domain = urlparse(url).netloc.lower()
        content = (title + " " + text).lower()
        
        # Check domain reliability first
        for category, domains in self.domain_reliability.items():
            if any(reliable_domain in domain for reliable_domain in domains):
                return category
        
        # Score based on keywords
        scores = {'reliable': 0, 'unreliable': 0, 'ambiguous': 0, 'wrong': 0}
        
        for category, keywords in self.reliability_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content:
                    scores[category] += 1
        
        # Return category with highest score, or 'ambiguous' if tied
        max_score = max(scores.values())
        if max_score == 0:
            return 'ambiguous'
        
        # Get categories with max score
        max_categories = [cat for cat, score in scores.items() if score == max_score]
        
        if len(max_categories) == 1:
            return max_categories[0]
        else:
            return 'ambiguous'  # If multiple categories tied, mark as ambiguous

    def should_follow_link(self, url):
        """Determine if we should follow a link"""
        domain = urlparse(url).netloc.lower()
        
        # Only follow links from domains we're interested in
        all_domains = []
        for domain_list in self.domain_reliability.values():
            all_domains.extend(domain_list)
        
        return any(target_domain in domain for target_domain in all_domains)

    def save_progress(self):
        """Save current progress to file"""
        results = {
            "reliable_sources": {
                "count": len(self.reliable_sources),
                "source_excerpts": self.reliable_sources
            },
            "unreliable_sources": {
                "count": len(self.unreliable_sources),
                "source_excerpts": self.unreliable_sources
            },
            "ambiguous_sources": {
                "count": len(self.ambiguous_sources),
                "source_excerpts": self.ambiguous_sources
            },
            "wrong_sources": {
                "count": len(self.wrong_sources),
                "source_excerpts": self.wrong_sources
            },
            "pages_processed": self.pages_processed,
            "last_saved": datetime.now().isoformat()
        }
        
        # Save progress file
        with open('rag_benchmark_progress.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def closed(self, reason):
        """Called when spider closes - save results"""
        
        # Save final results
        self.save_progress()  # Save one more time
        
        results = {
            "reliable_sources": {
                "count": len(self.reliable_sources),
                "source_excerpts": self.reliable_sources
            },
            "unreliable_sources": {
                "count": len(self.unreliable_sources),
                "source_excerpts": self.unreliable_sources
            },
            "ambiguous_sources": {
                "count": len(self.ambiguous_sources),
                "source_excerpts": self.ambiguous_sources
            },
            "wrong_sources": {
                "count": len(self.wrong_sources),
                "source_excerpts": self.wrong_sources
            },
            "final_stats": {
                "pages_processed": self.pages_processed,
                "completion_reason": reason,
                "completed_at": datetime.now().isoformat()
            }
        }
        
        # Save to final JSON file
        with open('rag_benchmark_data.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n=== RAG Benchmark Data Collection Complete ===")
        print(f"Completion reason: {reason}")
        print(f"Reliable sources: {len(self.reliable_sources)}")
        print(f"Unreliable sources: {len(self.unreliable_sources)}")
        print(f"Ambiguous sources: {len(self.ambiguous_sources)}")
        print(f"Wrong sources: {len(self.wrong_sources)}")
        print(f"Total pages processed: {self.pages_processed}")
        print(f"Final data saved to: rag_benchmark_data.json")
        print(f"Progress backups saved to: rag_benchmark_progress.json")


# Custom settings for the spider
custom_settings = {
    'USER_AGENT': 'RAG-Benchmark-Spider (+http://example.com/contact)',
    'ROBOTSTXT_OBEY': True,
    'DOWNLOAD_DELAY': 1,  # Be respectful to servers
    'RANDOMIZE_DOWNLOAD_DELAY': True,
    'CONCURRENT_REQUESTS': 16,
    'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
    'AUTOTHROTTLE_ENABLED': True,
    'AUTOTHROTTLE_START_DELAY': 1,
    'AUTOTHROTTLE_MAX_DELAY': 10,
    'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
    'CLOSESPIDER_TIMEOUT': 600,
}

# Example usage script
if __name__ == "__main__":
    """
    To run this scraper:
    
    1. Install Scrapy: pip install scrapy
    2. Save this code to a file like 'rag_spider.py'
    3. Run: scrapy runspider rag_spider.py
    
    Or create a proper Scrapy project:
    1. scrapy startproject rag_benchmark
    2. Add this spider to the spiders folder
    3. Run: scrapy crawl rag_benchmark
    """
    
    print("RAG Benchmark Spider ready!")
    print("Run with: scrapy runspider rag_spider.py")
    print("\nExample output format:")
    
    example_output = {
        "reliable_sources": {
            "count": 2,
            "source_excerpts": [
                {
                    "title": "NEJM Trial 2021",
                    "url": "https://doi.org/10.fake",
                    "date": "2021-05-10",
                    "text": "The Phase 3 trial of Drug X showed a 45% complete remission rate."
                },
                {
                    "title": "Nature Research Article",
                    "url": "https://nature.com/articles/example",
                    "date": "2021-03-15",
                    "text": "Peer-reviewed study demonstrates statistically significant improvement..."
                }
            ]
        },
        "unreliable_sources": {
            "count": 1,
            "source_excerpts": [
                {
                    "title": "Health Blog Opinion",
                    "url": "https://health.fake/blog",
                    "date": "2022-05-05",
                    "text": "Some analysts claim the remission rate could be lower than 30%."
                }
            ]
        }
    }
    
    print(json.dumps(example_output, indent=2))

    def load_trust_config(self, config_file="domain_trust_config.json"):
        with open(config_file, "r") as f:
            config = json.load(f)
        self.trusted_domains = set()
        self.ambiguous_domains = set(config.get("ambiguous", []))
        for topic, domains in config.items():
            if topic == "ambiguous":
                continue
            self.trusted_domains.update(domains.get("trusted", []))


    def start_requests(self):
        self.load_trust_config("domain_trust_config.json")
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        domain = urlparse(response.url).netloc.replace("www.", "")
        if domain in self.trusted_domains:
            trust_level = "credible"
        elif domain in self.ambiguous_domains:
            trust_level = "ambiguous"
        else:
            trust_level = "unknown"

        yield {
            "url": response.url,
            "title": response.xpath("//title/text()").get(),
            "text": " ".join(response.xpath("//p//text()").getall()).strip(),
            "trust_level": trust_level
        }



obj = RAGBenchmarkSpider()
print(obj.load_ambiguous())

