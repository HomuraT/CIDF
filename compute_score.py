from torch import Tensor
from config import args
import torch
from tqdm import trange


def compute_score(hr_vectors: Tensor, tail_vectors: Tensor, mode: str = 'cos-sim', one2one=False, max_num=4096) -> Tensor:
    '''
    计算得分

    :param hr_vectors: 头+关系的表示
    :param tail_vectors: 尾表示
    :param mode: 模式
    :return: 得分
    '''
    if mode == 'cos-sim':
        if one2one:
            return (hr_vectors * tail_vectors).sum(1)
        else:
            return hr_vectors @ tail_vectors
    elif mode == 'minus-distance':
        if one2one:
            return 1 / ((hr_vectors - tail_vectors).norm(2, -1) + 1)
        else:
            if tail_vectors.shape[1] > max_num:
                total_step = tail_vectors.shape[1] // max_num
                scores = []
                for step in trange(total_step + 1):
                    start_index = step * max_num
                    end_index = (step + 1) * max_num
                    scores.append(compute_score(hr_vectors, tail_vectors[:, start_index:end_index], mode=mode))
                return torch.cat(scores, dim=-1)
            else:
                hr_vectors = hr_vectors.unsqueeze(1).expand(hr_vectors.shape[0], tail_vectors.shape[-1],
                                                            hr_vectors.shape[-1])
                return 1 / ((hr_vectors - tail_vectors.t()).norm(2, -1) + 1)
