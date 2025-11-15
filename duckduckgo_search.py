import os
import argparse
import json
import time
import glob

from langchain.utilities import DuckDuckGoSearchAPIWrapper

def load_existing_results(output_dir):
    """Load existing search results to enable resuming from where we left off."""
    docs = []
    completed_count = 0
    
    # Check for existing output files
    output_files = glob.glob(os.path.join(output_dir, "output_*.json"))
    if output_files:
        # Find the latest output file
        latest_file = max(output_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Found existing results: {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                docs = json.load(f)
            completed_count = len(docs)
            print(f"Resuming from {completed_count} completed searches...")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            docs = []
            completed_count = 0
    
    # Also check for final output.json
    final_output = os.path.join(output_dir, "output.json")
    if os.path.exists(final_output) and not docs:
        try:
            with open(final_output, 'r') as f:
                docs = json.load(f)
            completed_count = len(docs)
            print(f"Found completed results: {completed_count} searches already done")
        except Exception as e:
            print(f"Error loading final output: {e}")
    
    return docs, completed_count

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="", type=str, required=True, help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--data_name", default="ASQA.json", type=str, help="The name of the input data file.")
    parser.add_argument("--top_k", default=50, type=int, help="The number of top-k documents to retrieve.")
    parser.add_argument("--save_step", default=100, type=int, help="The number of steps to save the output.")
    parser.add_argument("--time", default=-1, type=float, help="The time to sleep between requests.")
    parser.add_argument("--debug", default=False, action="store_true", help="Whether to run in debug mode.")
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the output files will be written.")
    
    args = parser.parse_args()
    
    data = json.load(open(os.path.join(args.data_dir, args.data_name)))
    
    # Replace BingSearchAPIWrapper with DuckDuckGoSearchAPIWrapper
    search = DuckDuckGoSearchAPIWrapper()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load existing results to enable resuming
    docs, start_idx = load_existing_results(args.output_dir)
    
    # Get total number of items to process
    total_items = len(data['dev'])
    remaining_items = total_items - start_idx
    
    print(f"Total items: {total_items}")
    print(f"Completed: {start_idx}")
    print(f"Remaining: {remaining_items}")
    
    if start_idx >= total_items:
        print("All searches already completed!")
        return
    
    # Convert to list for easier indexing
    dev_items = list(data['dev'].items())
    
    for idx in range(start_idx, total_items):
        id, ins = dev_items[idx]
        
        # Updated logging to show progress and remaining items
        if idx % 10 == 0 or idx == start_idx:
            remaining = total_items - idx
            print(f"Processing item {idx + 1}/{total_items} (ID: {id}) - {remaining} remaining...")
        
        question = ins['ambiguous_question']
        prefix = "site:en.wikipedia.org"
        
        try:
            # DuckDuckGo search with error handling
            doc = search.results(prefix + " " + question, args.top_k)
            # print(f"  ✓ Found {len(doc) if doc else 0} results for: {question[:60]}...")
        except Exception as e:
            print(f"  ✗ Search failed for question {idx}: {e}")
            doc = []  # Empty results on failure
        
        docs.append(doc)
        
        # Save intermediate results
        if (idx + 1) % args.save_step == 0:
            output_file = os.path.join(args.output_dir, f"output_{str(idx+1)}.json")
            with open(output_file, 'w') as f:
                json.dump(docs, f, indent=4)
            print(f"💾 Saved intermediate results: output_{idx+1}.json ({len(docs)} searches)")
            if args.debug:
                print("Debug mode: stopping after first save step")
                break
        
        # Sleep between requests if specified
        if args.time > 0:
            time.sleep(args.time)
    
    # Save final results
    final_output = os.path.join(args.output_dir, "output.json")
    with open(final_output, 'w') as f:
        json.dump(docs, f, indent=4)
    
    print(f"🎉 Completed! Final results saved to output.json ({len(docs)} total searches)")
        
        
if __name__ == "__main__":
    main()
