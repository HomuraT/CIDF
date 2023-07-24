import os
import json
import tqdm
import torch

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict

from config import args
from doc import load_data, Example
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from rerank import rerank_by_graph
from logger_config import logger

import torch.nn as nn

def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256, **kwargs) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    
    if len(kwargs) > 0:
        hr_tensor_global = hr_tensor.mean(0)
        ent_mean = entities_tensor.mean(0)
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        # ### hmt

        if 'cfw' in kwargs:
            sim_tensor = (batch_score.softmax(1) @ entities_tensor)
            batch_score = torch.mm(
                hr_tensor[start:end, :] - kwargs['cfw'] * nn.functional.normalize(sim_tensor.tanh(),
                                                                        dim=1) - 1 * ent_mean - 1 * hr_tensor_global,
                entities_tensor.t())
        if 'hr_mean_w' in kwargs:
            batch_score = compute_score(hr_tensor[start:end, :] - kwargs['hr_mean_w'] * hr_tensor_global.tanh(),
                                        entities_tensor.t(), mode=args.score_mode)

        # ### end hmt
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks

def predict_by_split(wandb_logger=None):
    assert os.path.exists(args.test_path)
    assert os.path.exists(args.train_path)
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    batch_size = 256
    
    if args.task == 'wiki5m_trans':
        from eval_wiki5m_trans import _dump_entity_embeddings, _load_entity_embeddings
        predictor.load(ckt_path=args.eval_model_path, use_data_parallel=True)
        _dump_entity_embeddings(predictor)
        entity_tensor = _load_entity_embeddings().cuda()
        batch_size = 32
        args.valid_path = args.test_path
    else:
        entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True, batch_size=batch_size)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False, batch_size=batch_size)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))
    
    ### hmt
    if wandb_logger:
        def add_prefix(prefix, d):
            new_d = {}
            for k, v in d.items():
                new_d[f"{prefix}_{k}"] = v
            return new_d
        wandb_logger.log(add_prefix('forward', forward_metrics))
        wandb_logger.log(add_prefix('backward', backward_metrics))
        wandb_logger.log(add_prefix('average', metrics))

    ### end hmt
    ###
    from tqdm import tqdm
    best_key = 'hit@1'
    best_metric = {best_key:-1}
    best_record = None
    all_record = []
    fms = []
    bms = []
    save_name = 'all_record.json'

    for i in tqdm(range(0, 10)):
        forward_metrics = eval_single_direction(predictor,
                                                    entity_tensor=entity_tensor,
                                                    eval_forward=True,batch_size=batch_size, cfw=i/10)

        backward_metrics = eval_single_direction(predictor,
                                                     entity_tensor=entity_tensor,
                                                     eval_forward=False,batch_size=batch_size, cfw=i/10)
        fms.append([i/10, forward_metrics])
        bms.append([i/10, backward_metrics])

    for (fw, forward_metrics)  in tqdm(fms):
        for (bw, backward_metrics) in bms:
            metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
            all_record.append({'fw':fw, 'bw':bw, 'forward_metrics':forward_metrics, 'backward_metrics':backward_metrics, 'metrics':metrics})
            if metrics[best_key] > best_metric[best_key]:
                best_metric = metrics
                best_record = {'fw':fw, 'bw':bw, 'forward_metrics':forward_metrics, 'backward_metrics':backward_metrics, 'metrics':metrics}

    with open(os.path.join(args.model_dir, save_name), 'w', encoding='utf-8') as f:
        json.dump({
            'best_record':best_record,
            'all_record':all_record
        }, f)
    forward_metrics = eval_single_direction(predictor,
                                                    entity_tensor=entity_tensor,
                                                    eval_forward=True,batch_size=batch_size, cfw=best_record['fw'])

    backward_metrics = eval_single_direction(predictor,
                                                     entity_tensor=entity_tensor,
                                                     eval_forward=False,batch_size=batch_size, cfw=best_record['bw'])
    print('结束了')
    ###


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256, **kwargs) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size, **kwargs)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics


if __name__ == '__main__':
    predict_by_split()
