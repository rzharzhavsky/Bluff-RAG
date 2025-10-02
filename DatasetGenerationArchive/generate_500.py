import os
import json
import time
import random
from datetime import datetime
from bluffrag_dataset_generation import BluffRagEntry

# Query history is now managed directly by bluffrag_dataset_generation.py

def load_progress():
    """Load progress tracking from file"""
    progress_file = "generation_progress.json"
    try:
        with open(progress_file, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "total_generated": 0,
            "topic_progress": {},
            "failed_entries": [],
            "last_updated": datetime.now().isoformat()
        }

def save_progress(progress):
    """Save progress tracking to file"""
    progress_file = "generation_progress.json"
    progress["last_updated"] = datetime.now().isoformat()
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

def regenerate_failed_entries():
    """Regenerate only the failed entries from previous runs"""
    
    # Load progress to see what failed
    progress = load_progress()
    failed_entries = progress.get("failed_entries", [])
    
    if not failed_entries:
        print("No failed entries to regenerate!")
        return
    
    print(f"=== Regenerating {len(failed_entries)} Failed Entries ===")
    
    # Load existing dataset
    dataset_file = "bluffrag_dataset.json"
    try:
        with open(dataset_file, "r", encoding='utf-8') as f:
            existing_dataset = json.load(f)
            print(f"Loaded existing dataset with {len(existing_dataset)} entries")
    except FileNotFoundError:
        existing_dataset = []
        print("No existing dataset found, creating new one")
    
    # Track regeneration progress
    regenerated = 0
    still_failed = []
    
    for failed_entry in failed_entries:
        entry_id = failed_entry["entry_id"]
        topic = failed_entry["topic"]
        error = failed_entry["error"]
        
        print(f"\n--- Regenerating: {entry_id} (Topic: {topic}) ===")
        print(f"Previous error: {error}")
        
        try:
            # Create entry with persistent subdomain tracking
            entry = BluffRagEntry(entry_id, topic)
            
            # Generate the entry
            result = entry.build()
            
            if result:
                # Remove from failed entries
                progress["failed_entries"].remove(failed_entry)
                
                # Add to dataset
                existing_dataset.append(result)
                
                # Update progress
                progress["total_generated"] += 1
                progress["topic_progress"][topic] = progress["topic_progress"].get(topic, 0) + 1
                
                # Save progress
                save_progress(progress)
                
                # Save dataset
                with open(dataset_file, "w") as f:
                    json.dump(existing_dataset, f, indent=2)
                
                regenerated += 1
                print(f"  ✅ Successfully regenerated {entry_id}")
                
                # Small delay to be respectful to APIs
                time.sleep(2)
                
            else:
                print(f"  ❌ Failed to regenerate {entry_id}")
                still_failed.append(failed_entry)
                
        except Exception as e:
            print(f"  ❌ Error regenerating {entry_id}: {e}")
            still_failed.append(failed_entry)
            continue
    
    # Update failed entries list
    progress["failed_entries"] = still_failed
    save_progress(progress)
    
    # Final save
    with open(dataset_file, "w") as f:
        json.dump(existing_dataset, f, indent=2)
    
    print(f"\n=== Regeneration Complete ===")
    print(f"Successfully regenerated: {regenerated}")
    print(f"Still failed: {len(still_failed)}")
    print(f"Total entries in dataset: {len(existing_dataset)}")

def generate_dataset_batch_full():
    """Generate the complete BLUFF-RAG dataset with 500 QA pairs (50 per topic)"""
    
    # Define all 10 topics for full dataset
    topics = [
        "public_health", "current_events", "history", "finance", "sports",
        "climate", "technology", "astronomy", "law", "psychology"
    ]
    entries_per_topic = 50  # Full dataset: 50 per topic
    
    # Load existing data
    progress = load_progress()
    
    print(f"=== BLUFF-RAG Dataset Generation (FULL DATASET MODE) ===")
    print(f"Target: {len(topics)} topics with {entries_per_topic} entries each")
    print(f"Current progress: {progress['total_generated']} entries generated")
    print(f"Topics: {', '.join(topics)}")
    print(f"{'='*50}")
    
    # Load existing dataset
    dataset_file = "bluffrag_dataset.json"
    try:
        with open(dataset_file, "r", encoding='utf-8') as f:
            existing_dataset = json.load(f)
            print(f"Loaded existing dataset with {len(existing_dataset)} entries")
    except FileNotFoundError:
        existing_dataset = []
        print("Creating new dataset file")
    
    # Track progress for this session
    session_generated = 0
    session_failed = 0
    
    try:
        for topic in topics:
            print(f"\n--- Processing Topic: {topic.upper()} (FULL DATASET MODE) ---")
            
            # Get current progress for this topic
            topic_progress = progress.get("topic_progress", {}).get(topic, 0)
            remaining_for_topic = entries_per_topic - topic_progress
            
            if remaining_for_topic <= 0:
                print(f"  Topic {topic} already complete ({topic_progress}/{entries_per_topic})")
                continue
            
            print(f"  Progress: {topic_progress}/{entries_per_topic} entries")
            print(f"  Remaining: {remaining_for_topic} entries")
            
            for entry_num in range(remaining_for_topic):
                entry_id = f"{topic}_{entry_num:03d}"
                
                # Skip if this entry already exists
                if any(entry.get("id") == entry_id for entry in existing_dataset):
                    print(f"    Entry {entry_id} already exists, skipping")
                    continue
                
                print(f"\n    Generating entry {entry_id} ({entry_num + 1}/{remaining_for_topic})")
                
                try:
                    # Create entry with persistent subdomain tracking
                    entry = BluffRagEntry(entry_id, topic)
                    
                    # Generate the entry
                    result = entry.build()
                    
                    if result:
                        # Add to dataset
                        existing_dataset.append(result)
                        
                        # Update progress
                        progress["total_generated"] += 1
                        progress["topic_progress"][topic] = progress["topic_progress"].get(topic, 0) + 1
                        
                        # Save progress every entry
                        save_progress(progress)
                        
                        session_generated += 1
                        print(f"       Success! Total generated: {progress['total_generated']}")
                        
                        # Small delay to be respectful to APIs
                        time.sleep(2)
                        
                        # Save dataset checkpoint every 10 entries for safety
                        if progress["total_generated"] % 10 == 0:
                            print(f"       Checkpoint saved at {progress['total_generated']} entries")
                            with open(dataset_file, "w") as f:
                                json.dump(existing_dataset, f, indent=2)
                            print(f"       Dataset checkpoint saved to {dataset_file}")
                        
                    else:
                        print(f"       Failed to generate entry")
                        session_failed += 1
                        
                        # Log failed entry
                        failed_entry = {
                            "entry_id": entry_id,
                            "topic": topic,
                            "timestamp": datetime.now().isoformat(),
                            "error": "build() returned None"
                        }
                        progress["failed_entries"].append(failed_entry)
                        
                except Exception as e:
                    print(f"       Error generating entry: {e}")
                    session_failed += 1
                    
                    # Log failed entry
                    failed_entry = {
                        "entry_id": entry_id,
                        "topic": topic,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    progress["failed_entries"].append(failed_entry)
                    
                    # Continue with next entry
                    continue
                
                # Check if we've hit the target for this topic
                if progress["topic_progress"].get(topic, 0) >= entries_per_topic:
                    print(f"\n Target of {entries_per_topic} entries reached for topic {topic}!")
                    print(f"  Query history for {topic} managed by bluffrag_dataset_generation.py")
                    break
            
            # Continue to next topic (full dataset generation)
            continue
    
    except KeyboardInterrupt:
        print(f"\n  Generation interrupted by user")
        print(f"   Progress saved. You can resume later.")
    
    finally:
        # Final save
        print(f"\n--- Final Save ---")
        
        # Save dataset
        with open(dataset_file, "w") as f:
            json.dump(existing_dataset, f, indent=2)
        print(f"Dataset saved: {dataset_file} ({len(existing_dataset)} entries)")
        
        # Save progress
        save_progress(progress)
        print(f"Progress saved: generation_progress.json")
        
        # Save query history (now managed by bluffrag_dataset_generation.py)
        print(f"Query history managed by bluffrag_dataset_generation.py")
        
        # Summary
        print(f"\n=== Generation Summary (FULL DATASET MODE) ===")
        print(f"Total entries in dataset: {len(existing_dataset)}")
        print(f"Entries generated this session: {session_generated}")
        print(f"Failed entries this session: {session_failed}")
        print(f"Total failed entries: {len(progress['failed_entries'])}")
        
        # Topic breakdown
        print(f"\nTopic Breakdown:")
        for topic in topics:
            count = progress["topic_progress"].get(topic, 0)
            print(f"  {topic}: {count}/{entries_per_topic}")
        
        print(f"\n Full dataset generation complete! Check the generated files:")
        print(f"  - {dataset_file} (main dataset)")
        print(f"  - subdomain_usage_*.json (subdomain tracking for all topics)")
        print(f"  - query_history.json (managed by bluffrag_dataset_generation.py)")
        print(f"  - generation_progress.json (progress tracking)")

if __name__ == "__main__":
   #import sys
    """
    if len(sys.argv) > 1 and sys.argv[1] == "regenerate":
        print(f"=== REGENERATION MODE ===")
        regenerate_failed_entries()
    else:
        print(f"Starting FULL dataset generation for 500 QA pairs...")
        print(f"Target: 50 entries per topic across 10 topics")
        print(f"Total: 500 QA pairs")
        print(f"Press Ctrl+C to stop and save progress")
        print(f"Estimated time: 8-12 hours (can be left running overnight)")
        print(f"Topics: public_health, current_events, history, finance, sports, climate, technology, astronomy, law, psychology")
        print(f"\nTo regenerate failed entries only, run: python generate_500.py regenerate")
        
        generate_dataset_batch_full()
"""
    print("Regenerating============")
    regenerate_failed_entries()