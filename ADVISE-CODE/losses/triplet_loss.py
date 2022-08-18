import torch
import torch.nn as nn

def _mine_hard_examples(distances, top_k):
    """Mines hard examples.

    Mine hard returns examples with smallest values in the following masked matrix:

    / 0, 1, 1, 1 \
    | 1, 0, 1, 1 |
    | 1, 1, 0, 1 |
    \ 1, 1, 1, 0 /
        
    Args:
        distances: a [batch, batch] float tensor, in which distances[i, j] is the
        distance between i-th item and j-th item.
        top_k: number of negative examples to choose per each row.

    Returns:
        pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
        neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
    """
    batch_size = list(distances.size())[0]
    top_k = min(top_k, batch_size - 1)


    pos_indices = torch.unsqueeze(torch.arange(batch_size, dtype=torch.int32), 1)
    pos_indices = torch.tile(pos_indices, (1, 1 + top_k))

    _, neg_indices = torch.topk(-distances, k= 1 + top_k)
    
    pos_indices = pos_indices.to(neg_indices.device)   
    
    masks =  ~pos_indices.eq(neg_indices)
    pos_indices = torch.masked_select(pos_indices, masks)
    neg_indices = torch.masked_select(neg_indices, masks)
    

    return pos_indices, neg_indices

def compute_triplet_loss(anchors, positives, negatives, distance_fn, alpha):
    """Computes triplet loss.

    Args:
        anchors: a [batch, embedding_size] tensor.
        positives: a [batch, embedding_size] tensor.
        negatives: a [batch, embedding_size] tensor.
        distance_fn: a function using to measure distance between two [batch,
        embedding_size] tensors
        alpha: a float value denoting the margin.

    Returns:
        loss: the triplet loss tensor.
        summary: a dict mapping from summary names to summary tensors.
    """
    batch_size = list(anchors.size())[0]
    batch_size = max(1e-12, batch_size)

    dist1 = distance_fn(anchors, positives)
    dist2 = distance_fn(anchors, negatives)

    losses = torch.maximum(dist1 - dist2 + alpha, torch.tensor(0))
    losses = torch.masked_select(losses, losses > 0)

    loss = torch.where(torch.tensor(list(losses.size())[0] > 0).to(losses.device), torch.mean(losses),  torch.tensor(0.0).to(losses.device))

    # Gather statistics.
    loss_examples = torch.count_nonzero(dist1 + alpha >= dist2).float()
    return loss, { 'loss_examples': loss_examples}

def triplet_loss_wrap_func(anchors, positives, distance_fn, mining_fn, refine_fn, margin, tag=None):
    """Wrapper function for triplet loss.

    Args:
        anchors: a [batch, common_dimensions] tf.float32 tensor.
        positives: a [batch, common_dimensions] tf.float32 tensor.
        similarity_matrx: a [common_dimensions, common_dimensions] tf.float32 tensor.
        distance_fn: a callable that takes two batch of vectors as input.
        mining_fn: a callable that takes distance matrix as input.
        refine_fn: a callable that takes pos_indices and neg_indices as inputs.
        margin: margin alpha of the triplet loss.

    Returns:
        loss: the loss tensor.
    """ 
      
    distances = torch.multiply(torch.unsqueeze(anchors, 1), torch.unsqueeze(positives, 0))
    distances = 1 - torch.sum(distances, 2)


    pos_indices, neg_indices = mining_fn(distances)

    if not refine_fn is None:
        pos_indices, neg_indices = refine_fn(pos_indices, neg_indices)
    

    loss, summary = compute_triplet_loss(
        anchors= anchors[pos_indices.long()], 
        positives= positives[pos_indices.long()], 
        negatives= positives[neg_indices.long()],
        distance_fn=distance_fn,
        alpha=margin)


    return loss, summary

def compute_loss(predictions, config, is_training = True):
    loss_dict = {}

    def _mine_hard_examples_wrap(distances):
        return _mine_hard_examples(distances, config['triplet_mining']['mine_hard']['top_k'])

    mining_fn = _mine_hard_examples_wrap

    image_id = predictions['image_id']
    img_encoded = predictions['img_encoded']
    stmt_encoded = predictions['stmt_encoded']
    ocr_encoded = predictions['ocr_encoded']

    margin = config['triplet_margin']
    keep_prob = config['joint_emb_dropout_keep_prob']

    def distance_fn(x, y):
        """Distance function."""
        distance = nn.functional.dropout(torch.multiply(x, y), p =  (1 - keep_prob), training = is_training)
        distance = 1 - torch.sum(distance, 1)
        return distance

    def refine_fn(pos_indices, neg_indices):
        """Refine function."""
        pos_ids = [image_id[i] for i in pos_indices.tolist()]
        neg_ids = [image_id[i] for i in neg_indices.tolist()]
        
        masks = torch.tensor([pos_ids[i] != neg_ids[i] for i in range(len(pos_ids))]).to(pos_indices.device).bool()
          
        pos_indices = torch.masked_select(pos_indices, masks)
        neg_indices = torch.masked_select(neg_indices, masks)
        return pos_indices, neg_indices


    loss_img_stmt, summary = triplet_loss_wrap_func(
        img_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin, 'img_stmt')
    loss_stmt_img, summary = triplet_loss_wrap_func(
        stmt_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin, 'stmt_img')

    loss_dict = {
      'triplet_img_stmt': loss_img_stmt,
      'triplet_stmt_img': loss_stmt_img,
    }
    
    loss_ocr_stmt, summary = triplet_loss_wrap_func(
        ocr_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin, 'ocr_stmt')
    loss_stmt_ocr, summary = triplet_loss_wrap_func(
        stmt_encoded, ocr_encoded, distance_fn, mining_fn, refine_fn, margin, 'stmt_ocr')

    loss_dict.update({
      'triplet_ocr_stmt': loss_ocr_stmt,
      'triplet_stmt_ocr': loss_stmt_ocr,
    })   

    # For optional constraints.
    if config['densecap_loss_weight'] > 0:
        dense_encoded = predictions['dense_encoded']
        loss_dense_img, summary = triplet_loss_wrap_func(
            dense_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin,
            'dense_img')
        loss_dense_stmt, summary = triplet_loss_wrap_func(
            dense_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin,
            'dense_stmt')
        loss_dict.update({
            'triplet_dense_img': loss_dense_img * config['densecap_loss_weight'],
            'triplet_dense_stmt': loss_dense_stmt * config['densecap_loss_weight'],
        })

    if config['symbol_loss_weight'] > 0:
        number_of_symbols = predictions['number_of_symbols']
        symb_encoded = predictions['symb_encoded']

        # Since not all images have symbol annotations. Mask them out.
        indices = torch.squeeze(torch.nonzero(torch.greater(number_of_symbols, 0), as_tuple=False), axis=1)
        symb_encoded = symb_encoded[indices]
        img_encoded = img_encoded[indices]
        stmt_encoded = stmt_encoded[indices]

        loss_symb_img, summary = triplet_loss_wrap_func(
            symb_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin,
            'symb_img')
        loss_symb_stmt, summary = triplet_loss_wrap_func(
            symb_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin,
            'symb_stmt')
        loss_dict.update({
            'triplet_symb_img': loss_symb_img * config['symbol_loss_weight'],
            'triplet_symb_stmt': loss_symb_stmt * config['symbol_loss_weight'],
        })

    return loss_dict



    