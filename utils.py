import torch

def batchify(data, heads, bsz, device, pad):
    """
    Groups data into batches, sorts by length to minimize padding,
    and uses pad_sequence for efficient padding.
    """
    if not data:
        return [], []
    
    valid_data = [(d, h) for d, h in zip(data, heads) if len(d) > 0]
    if not valid_data:
        return [], []

    # Sort data by sentence length for efficient padding
    valid_data.sort(key=lambda x: len(x[0]))
    sorted_sents = [x[0] for x in valid_data]
    sorted_heads = [x[1] for x in valid_data]

    batched_sents = []
    batched_heads = []
    for i in range(0, len(sorted_sents), bsz):
        batch_sents = sorted_sents[i:i + bsz]
        
        # Use the standard PyTorch utility to pad sequences
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            batch_sents, 
            batch_first=True, 
            padding_value=pad
        )
        batched_sents.append(padded_batch.to(device))
        
        # Batch the corresponding heads
        batched_heads.append(sorted_heads[i:i + bsz])

    return batched_sents, batched_heads