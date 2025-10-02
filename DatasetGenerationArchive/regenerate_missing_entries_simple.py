#!/usr/bin/env python3
"""
Simplified script to regenerate missing entries to reach 50 entries per domain.
The CalmRagEntry class automatically selects the least-used subdomain and updates usage files.
"""

import os
import json
import time
from datetime import datetime
from calmrag_dataset_generation import CalmRagEntry, save_query_history_to_file

# Target: 50 entries per domain
TARGET_ENTRIES_PER_DOMAIN = 50

def load_dataset():
    """Load the main dataset"""
    with open('../calmrag_dataset.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(dataset):
    """Save the updated dataset"""
    with open('../calmrag_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

def get_domain_counts(dataset):
    """Get current entry counts for each domain"""
    domain_counts = {}
    for entry in dataset:
        topic = entry['topic']
        domain_counts[topic] = domain_counts.get(topic, 0) + 1
    return domain_counts

def generate_entry(domain, entry_index):
    """Generate a new entry - CalmRagEntry will automatically pick the least-used subdomain"""
    entry_id = f"{domain}_{entry_index:03d}"
    print(f"Generating {entry_id}...")
    
    try:
        # Create CalmRagEntry - it automatically selects least-used subdomain
        entry = CalmRagEntry(entry_id, domain)
        
        # Build the entry
        entry_dict = entry.build()
        
        if entry_dict:
            print(f"  ✅ Generated {entry_id} (subdomain: {entry.current_subdomain})")
            return entry_dict
        else:
            print(f"  ❌ Failed to generate {entry_id}")
            return None
        
    except Exception as e:
        print(f"  ❌ Error generating {entry_id}: {e}")
        return None

def regenerate_missing_entries():
    """Main function to regenerate missing entries"""
    print("=== CALM-RAG Missing Entry Regeneration ===")
    
    # Load current dataset
    dataset = load_dataset()
    print(f"Loaded dataset with {len(dataset)} entries")
    
    # Get current domain counts
    domain_counts = get_domain_counts(dataset)
    print(f"Current domain counts: {domain_counts}")
    
    # Identify domains that need more entries
    domains_to_process = []
    for domain, count in domain_counts.items():
        if count < TARGET_ENTRIES_PER_DOMAIN:
            needed = TARGET_ENTRIES_PER_DOMAIN - count
            domains_to_process.append((domain, needed))
            print(f"{domain}: {count}/{TARGET_ENTRIES_PER_DOMAIN} entries (need {needed} more)")
    
    if not domains_to_process:
        print("All domains already have 50+ entries!")
        return
    
    print(f"\nProcessing {len(domains_to_process)} domains...")
    
    total_generated = 0
    
    for domain, needed_count in domains_to_process:
        print(f"\n=== Processing {domain} domain (need {needed_count} more entries) ===")
        
        generated_count = 0
        max_retries = 3  # Retry failed entries up to 3 times
        
        while generated_count < needed_count:
            # Find the next available entry index (sequential, no gaps)
            existing_ids = [entry['id'] for entry in dataset if entry['topic'] == domain]
            existing_indices = [int(id.split('_')[-1]) for id in existing_ids if id.startswith(f"{domain}_")]
            next_index = len(existing_indices) + 1  # Next sequential number
            
            # Generate the entry (CalmRagEntry will pick the least-used subdomain)
            retry_count = 0
            new_entry = None
            
            while retry_count < max_retries and not new_entry:
                if retry_count > 0:
                    print(f"  🔄 Retry attempt {retry_count}/{max_retries}")
                
                new_entry = generate_entry(domain, next_index)
                
                if not new_entry:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"  ⏳ Waiting 1 seconds before retry...")
                        time.sleep(1)
            
            if new_entry:
                # Add to dataset
                dataset.append(new_entry)
                generated_count += 1
                total_generated += 1
                
                # Save dataset after each successful generation
                save_dataset(dataset)
                print(f"  📊 Dataset saved with {len(dataset)} entries")
                
            else:
                print(f"  ❌ Failed after {max_retries} attempts, moving to next entry...")
                # Still increment to avoid infinite loop
                generated_count += 1
            
            # Rate limiting
            time.sleep(1)
            
            # Save query history periodically
            if generated_count % 5 == 0:
                save_query_history_to_file()
                print(f"  💾 Query history saved")
        
        print(f"\n✅ {domain} domain complete: Generated {generated_count}/{needed_count} entries")
    
    # Final saves
    save_dataset(dataset)
    save_query_history_to_file()
    
    print(f"\n🎉 Regeneration Complete!")
    print(f"📈 Total entries generated: {total_generated}")
    print(f"📊 Final dataset size: {len(dataset)} entries")
    
    # Show final domain counts
    final_counts = get_domain_counts(dataset)
    print(f"📋 Final domain counts: {final_counts}")

if __name__ == "__main__":
    regenerate_missing_entries()
