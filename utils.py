import os
import json

def save_results(args, data, preds, outputs):
    preds_w_ids = {}
    outputs_w_ids = {}

    # Handle case where outputs is already a dict (resumed from final format)
    if isinstance(outputs, dict):
        outputs_w_ids = outputs
        # Extract preds from outputs if not provided
        if not preds or isinstance(preds, dict):
            preds_w_ids = preds if isinstance(preds, dict) else {}
        else:
            # This shouldn't happen, but handle it
            for idx, (id, entry) in enumerate(data['dev'].items()):
                if idx >= len(preds):
                    break
                preds_w_ids[id] = preds[idx]
    else:
        # Normal case: outputs is a list
        for idx, (id, entry) in enumerate(data['dev'].items()):
            if idx >= len(preds):
                break

            preds_w_ids[id] = preds[idx]
            outputs_w_ids[id] = outputs[idx]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save both to the same directory, using output.json (not outputs.json)
    with open(os.path.join(args.output_dir, args.prefix + "preds.json"), 'w') as f:
        json.dump(preds_w_ids, f, indent=4)

    with open(os.path.join(args.output_dir, args.prefix + "output.json"), 'w') as f:
        json.dump(outputs_w_ids, f, indent=4)


