import dsp

def retrieve_passages(args, ins, search_passages=None):
    question = ins.question
    passages = dsp.retrieve(question, k=args.top_k_docs)
        
    if search_passages is not None:
        passages += search_passages
    
    return passages
