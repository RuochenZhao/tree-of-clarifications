import os
import json
import dsp
import argparse
from tqdm import tqdm
import threading
import warnings

# Set environment variables to prevent threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from toc import (
    rerank,
    parse_disambig, make_str_disambig,
    ToC, Node,
    retrieve_passages,
    get_rac_template,
    check_unique, verify_with_evidence,
)
from utils import save_results


def get_dataset(args):
    data_path = os.path.join(args.data_dir, args.data_name) 
    data = json.load(open(data_path))
    
    qa_pairs = {}
    for split in ['train', 'dev']:
        n = 0
        qa_pairs[split] = []
        for idx, (id, ins) in enumerate(data[split].items()):
            question = ins['ambiguous_question']
            answers = [anns['long_answer'] for anns in ins['annotations']]
            
            entry = {'question' : question, 
                     'answer'   : answers,
                     'id'       : id,
            }
            
            str_disambigs = make_str_disambig(ins['qa_pairs']) if split == 'train' else ""
            entry.update({'disambig' : str_disambigs})
            
            qa_pairs[split] += [entry]        
            n+=1
    
    train = [dsp.Example(**kw_example) for kw_example in qa_pairs['train']]
    dev   = [dsp.Example(**kw_example) for kw_example in qa_pairs['dev']]

    return train, dev, data


def get_example(args, train, ins, passages,
                reranker=None, consolidation=False):
    
    question = ins.question
    
    # Use simple random sampling to avoid SentenceTransformers mutex issues
    import random
    n_dynamic = min(args.n_shot, len(train))
    demos = random.sample(train, n_dynamic) if len(train) > n_dynamic else train.copy()

    dic_example = {'question': question,
                   'demos': demos,
                   }
    
    # Use simple TF-IDF reranker instead of neural reranker
    if args.top_k_reranked > 0 and reranker is not None:
        top_k_reranked = min(args.top_k_reranked, len(passages))
        passages = rerank(reranker, question, passages, top_k_reranked)
        dic_example.update({'context' : passages})
    else:
        dic_example.update({'context' : passages})
        
    if consolidation:
        dic_example.update({'disambig' : ins.disambig})
        
    example = dsp.Example(**dic_example)
    
    return example


@dsp.transformation
def QD_predict(example: dsp.Example, qd_template, 
               sc=False, temperature=0.0):
    example, completions = dsp.generate(qd_template, temperature=temperature)(example, stage='qa')
    
    kw_out = { 
            'answer': completions.answer,
            'disambig': completions.disambig,
    }
    
    out_example = example.copy(**kw_out)
    
    return out_example, completions

def remove_demos(demos, target_ids):
    new_demos = []
    for demo in demos:
        if demo.id not in target_ids:
            new_demos += [demo]
    return new_demos

def remove_dup_demos(demos, lst_disambigs):
    answers = []
    for disambig in lst_disambigs:
        answers += [disambig['answer']]
    
    target_ids = []
    for demo in demos:
        demo_disambigs = parse_disambig(demo.disambig)
        for da in demo_disambigs:
            if dsp.metrics.F1(da['answer'], answers) > 0.8:
                target_ids += [demo.id]
    
    demos = remove_demos(demos, target_ids)
    
    return demos

def remove_dup_psgs(passages, contexts, lst_disambigs):
    answers = []
    for disambig in lst_disambigs:
        answers += [disambig['answer']]
    
    target_idxs = []
    for idx, passage in enumerate(passages):
        if passage in contexts and \
            dsp.passage_has_answers(passage, answers):
                target_idxs += [idx]
    
    passages = [passage for idx, passage in enumerate(passages) if idx not in target_idxs]
    
    return passages

def load_existing_outputs(output_dir):
    """Load existing output.json file and return the results and starting index."""
    preds = []
    outputs = []
    start_idx = 0

    output_file = os.path.join(output_dir, "output.json")

    if not os.path.exists(output_file):
        print("📝 No existing output.json found. Starting from scratch.")
        return preds, outputs, start_idx

    print(f"🔄 Found existing output.json")

    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)

        # Check if it's a dict (final format) or list (intermediate checkpoint)
        if isinstance(loaded_data, dict):
            # Final format - return as-is
            outputs = loaded_data
            # Try to load preds file too
            preds_file = os.path.join(output_dir, "preds.json")
            if os.path.exists(preds_file):
                with open(preds_file, 'r') as f:
                    preds = json.load(f)
            else:
                # Extract preds from outputs
                preds = {}
                for id, output in outputs.items():
                    if isinstance(output, dict) and 'answer' in output:
                        preds[id] = output['answer']
            start_idx = len(outputs)
            print(f"✅ Loaded {len(outputs)} existing results (final format). Resuming from index {start_idx}")
        else:
            # Intermediate checkpoint format (list)
            outputs = loaded_data
            # Extract predictions from outputs
            preds = []
            for output in outputs:
                if isinstance(output, dict) and 'answer' in output:
                    preds.append(output['answer'])
                else:
                    # Fallback: try to extract answer from the output structure
                    preds.append(str(output))

            start_idx = len(outputs)
            print(f"✅ Loaded {len(outputs)} existing results. Resuming from index {start_idx}")

    except Exception as e:
        print(f"⚠️  Error loading output.json: {e}")
        print("📝 Starting from scratch.")
        return [], [], 0

    return preds, outputs, start_idx

def get_argparser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument("--data_name", default="ASQA.json", type=str, help="The input data name.")
    # Line 130: Update argument name and help text
    parser.add_argument("--search_path", default=None, type=str, help="The search data path (DuckDuckGo results).")
    parser.add_argument("--prefix", default='', type=str, help="The prefix of output files.")
    parser.add_argument("--model_type", default='gemini-2.0-flash', type=str, help="The Gemini model type.")
    parser.add_argument("--colbert_url", default='', type=str, help= "The colbert server url (deprecated, use --pinecone_api_key instead).")
    parser.add_argument("--pinecone_api_key", default='pcsk_4UD9WH_BwHJtMkiLUuYb4zqTLkXdLHM1GNvXyULGwkcBk3nvG1BvNDYrTQjkWpz8n4HC5H', type=str, help="The Pinecone API key for Wikipedia retrieval.")
    parser.add_argument("--pinecone_index", default='wikipedia', type=str, help="The Pinecone index name.")
    parser.add_argument("--use_pinecone", default=False, action='store_true', help="Use Pinecone instead of ColBERT for retrieval.")
    parser.add_argument("--use_wikipedia", default=False, action='store_true', help="Use Wikipedia API for retrieval (no setup required).")
    parser.add_argument("--wikipedia_language", default='en', type=str, help="Wikipedia language code (default: en).")
    parser.add_argument("--temperature", default=0.7, type=float, help="The temperature for generation.")
    parser.add_argument("--n_shot", default=5, type=int, help="The number of few-shot examples for in-context learning.")
    parser.add_argument("--n_dev", default=-1, type=int, help="The number of dev examples to run.")
    parser.add_argument("--max_nodes", default=10, type=int, help="The maximum number of nodes in a tree.")
    parser.add_argument("--max_depth", default=3, type=int, help="The maximum depth of a tree.")
    parser.add_argument("--max_trials", default=3, type=int, help="The maximum number of restarts.")
    parser.add_argument("--top_k_docs", default=100, type=int, help="The maximum number of retrieved documents.")
    parser.add_argument("--top_k_reranked", default=5, type=int, help="The maximum number of reranked documents.")
    parser.add_argument("--save_steps", default="10", type=str, help="you can save intermediate results.")
    parser.add_argument("--save_step", default=10, type=int, help="Save intermediate results every N steps.")
    parser.add_argument("--verify", default=False, action='store_true',)
    parser.add_argument("--resume", default=True, action='store_true', help="Resume from existing output files if found.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where predictions will be written.",
    )
    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()
    
    print("🚀 Starting Tree-of-Clarifications pipeline...")
    print(f"📊 Data directory: {args.data_dir}")
    print(f"🔍 Search path: {args.search_path}")
    print(f"🤖 Model: {args.model_type}")
    print(f"📁 Output directory: {args.output_dir}")
    
    ## Set DSP configuration
    print("⚙️  Configuring DSP settings...")
    print("🧠 Initializing Gemini LM...")
    args.gemini_key = os.environ['GEMINI_KEY']
    lm = dsp.Gemini(model=args.model_type, api_key=args.gemini_key)
    
    # Initialize retriever based on configuration
    if args.use_wikipedia:
        print("🔍 Initializing Wikipedia API retriever...")
        rm = dsp.WikipediaColBERTv2(
            language=args.wikipedia_language,
            user_agent="TreeOfClarifications/1.0"
        )
    elif args.use_pinecone and args.pinecone_api_key:
        print("🔍 Initializing Pinecone retriever...")
        rm = dsp.PineconeColBERTv2(
            api_key=args.pinecone_api_key,
            index_name=args.pinecone_index
        )
    elif args.colbert_url:
        print("🔍 Initializing ColBERT retriever...")
        rm = dsp.ColBERTv2(url=args.colbert_url)
    else:
        raise ValueError("Either --use_wikipedia, --use_pinecone with --pinecone_api_key, or --colbert_url must be provided")

    # Configure DSP with reranker if available
    kw_config = {'lm' : lm, 'rm' : rm}

    if args.top_k_reranked > 0:
        kw_config['reranker'] = dsp.SentenceTransformersCrossEncoder()

    print("✅ Configuring DSP...")
    dsp.settings.configure(**kw_config)
    
    print("📚 Loading dataset...")
    train, dev, data = get_dataset(args)
    print(f"✅ Dataset loaded: {len(train)} train, {len(dev)} dev examples")
    
    print("📝 Getting RAC template...")
    rac_template = get_rac_template()
    print("✅ Template loaded")
    
    kw_args_ex = {}
    if args.top_k_reranked > 0:
        kw_args_ex['reranker'] = dsp.settings.reranker
    
    if args.search_path is not None:
        print(f"🔍 Loading search results from {args.search_path}...")
        search_results = json.load(open(args.search_path))
        assert len(search_results) == len(dev)
        print(f"✅ Search results loaded: {len(search_results)} entries")

    if args.n_dev < 0:
        n_dev = len(dev)
    else:
        n_dev = min(args.n_dev, len(dev))
    
    print(f"🎯 Processing {n_dev} examples")
        
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory created: {args.output_dir}")
    
    # Load existing outputs if resume is enabled
    if args.resume:
        print("🔄 Checking for existing output files...")
        preds, outputs, start_idx = load_existing_outputs(args.output_dir)

        # Convert dicts to lists if needed (when resuming from final format)
        if isinstance(outputs, dict):
            print("🔄 Converting final format to list for continued processing...")
            # Maintain order based on dev examples
            preds_list = []
            outputs_list = []
            for i in range(min(start_idx, len(dev))):
                example_id = dev[i].id
                if example_id in outputs:
                    outputs_list.append(outputs[example_id])
                    if isinstance(preds, dict) and example_id in preds:
                        preds_list.append(preds[example_id])
            preds = preds_list
            outputs = outputs_list
            print(f"✅ Converted {len(outputs)} results to list format")
    else:
        print("📝 Starting fresh (resume disabled)")
        preds, outputs, start_idx = [], [], 0
    
     
    lst_err = []
    search_passages = None
    
    # Adjust the range to start from where we left off
    remaining_examples = dev[start_idx:n_dev]
    if start_idx > 0:
        print(f"🔄 Resuming from example {start_idx + 1}/{n_dev}")
    else:
        print("🔄 Starting main processing loop...")
    
    for idx, ambig_ins in enumerate(tqdm(remaining_examples, desc="Processing examples", initial=start_idx, total=n_dev)):
        actual_idx = start_idx + idx
        print(f"\n📝 Processing example {actual_idx + 1}/{n_dev}: {ambig_ins.question[:100]}...")
        cur_demos = train.copy()
        
        if args.search_path is not None:
            search_passages = search_results[actual_idx]
            print(f"🔍 Using search passages: {len(search_passages) if search_passages else 0} passages")
        
        print("📚 Retrieving passages...")
        all_passages = retrieve_passages(args, ambig_ins, search_passages=search_passages)
        print(f"✅ Retrieved {len(all_passages)} passages")
        
        cur_passages = all_passages.copy()
        toc = ToC(root=Node(ambig_ins))
        do_pruning = args.verify == True
        n_restarts = 0 ; n_expansions = 0
        
        print(f"🌳 Building tree (max_nodes: {args.max_nodes}, max_depth: {args.max_depth})...")
        
        while n_restarts < args.max_trials and \
            toc.n_nodes < args.max_nodes and \
            n_expansions <= 15:
            
            n_expansions += 1
            print(f"  🔄 Expansion {n_expansions}, Restarts: {n_restarts}, Nodes: {toc.n_nodes}")
            
            if toc.leaf_nodes == []:
                toc.leaf_nodes = [toc.root]
                toc.leaf_nodes += toc.valid_nodes
                n_restarts += 1
                print(f"  🔄 Restarting (attempt {n_restarts})")

            cur_node = toc.leaf_nodes.pop(0)
            cur_ins = cur_node.ins
            if cur_node.depth > args.max_depth:
                print(f"  ⏭️  Skipping node at depth {cur_node.depth} (max: {args.max_depth})")
                continue
            
            print(f"  🤖 Generating QD prediction for depth {cur_node.depth}...")
            qd_example = get_example(args, cur_demos, cur_ins, cur_passages, **kw_args_ex)
            toc.slt_psgs += qd_example.context
            qd_result, qd_completions = QD_predict(qd_example, rac_template, sc=False, temperature=args.temperature)
            try:
                lst_disambigs = parse_disambig(qd_result.disambig)
                if lst_disambigs == []:
                    lst_disambigs = parse_disambig(qd_result.answer.split("\nAnswer:")[0])
            except:
                lst_err += [[actual_idx, toc, qd_result.disambig]]
            
            if args.verify:
                cur_demos = remove_dup_demos(cur_demos, lst_disambigs)
                cur_passages = remove_dup_psgs(cur_passages, qd_example.context, lst_disambigs)
                
                if do_pruning:
                    valid_disambigs = []
                    for disambig in lst_disambigs:
                        if check_unique(toc.valid_qas, disambig):
                            ver_completion = verify_with_evidence(dsp.settings.lm,
                                                                toc,
                                                                disambig,
                                                                dsp.settings.reranker)
                            if "True" in ver_completion[0]:
                                valid_disambigs += [disambig]
                    lst_disambigs = valid_disambigs.copy()
            
            if len(lst_disambigs) > 0:
                toc.add_nodes(lst_disambigs, depth=cur_node.depth+1)
                continue
            
            if do_pruning:
                if n_restarts >= args.max_trials or n_expansions >= 10:    
                    n_restarts = 0
                    do_pruning = False
                    continue
        
        print(f"🌳 Tree construction complete. Final nodes: {toc.n_nodes}")
        tree_ins = toc._get_tree(args.max_nodes)
        
        print("🔄 Generating final consolidation...")
        kw_args_ex.update({'consolidation': True})
        ac_example = get_example(args, cur_demos, tree_ins, all_passages, **kw_args_ex)
        
        ac_result, ac_completions = QD_predict(ac_example, rac_template, sc=False)
        
        preds   += [ac_result.answer]
        outputs += [ac_completions.data[0]]
        
        print(f"✅ Example {idx + 1} completed!")

        # Save intermediate results - overwrite output.json each time
        if (actual_idx + 1) % args.save_step == 0:
            output_file = os.path.join(args.output_dir, "output.json")
            with open(output_file, 'w') as f:
                json.dump(outputs, f, indent=4)
            print(f"💾 Saved intermediate results to output.json ({len(outputs)} examples)")
    
    print("\n🔍 Inspecting LM history...")
    lm.inspect_history(n=1)
    
    print("💾 Saving final results...")
    save_results(args, data, preds, outputs)
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
    