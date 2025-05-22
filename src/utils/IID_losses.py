import sys

import torch


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  # p_i_j = compute_joint(x_out, x_tf_out)




  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
  su_temp1 = x_out.unsqueeze(2)
  su_temp2 = x_tf_out.unsqueeze(1)
  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k #这两个相乘有什么用？
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise 为什么要对称
  p_i_j = p_i_j / p_i_j.sum()  # normalise




  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
  #                           - torch.log(p_j) \
  #                           - torch.log(p_i))

  # loss_no_lamb = loss_no_lamb.sum()

  return loss


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j


def dynamic_q_entropy(p):
  prob_dist = p.mean(dim=0)

  entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9))

  max_entropy = torch.log(torch.tensor(prob_dist.shape[0], dtype=torch.float))

  imbalance_score = entropy / max_entropy

  q = imbalance_score
  return q.item()


def tsallis_log(p, q, EPS=sys.float_info.epsilon):
  p = torch.clamp(p, min=EPS)

  if abs(q - 1.0) < EPS:
    return -torch.sum(p * torch.log(p + EPS)).item()
  else:
    return (p ** (1 - q) - 1) / (1 - q)


def tsallis_mutual_info(x_out, x_tf_out, q_value, beta=0.9, lamb=1.0, EPS=sys.float_info.epsilon):
  _, k = x_out.size()
  bn_, k_ = x_out.size()

  if q_value is None:
    q_value = 0.9
  else:
    q_value = beta * q_value + (1 - beta) * dynamic_q_entropy(x_out)
  # q_value = dynamic_q_entropy(x_tf_out)
  # q_value = dynamic_q_pseudo_gini(x_tf_out, k_)

  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)
  p_i_j = p_i_j.sum(dim=0)
  p_i_j = (p_i_j + p_i_j.t()) / 2.
  p_i_j = p_i_j / p_i_j.sum()

  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

  p_i_j = torch.clamp(p_i_j, min=EPS)
  p_i = torch.clamp(p_i, min=EPS)
  p_j = torch.clamp(p_j, min=EPS)

  loss = - p_i_j * (tsallis_log(p_i_j, q_value) \
                    - lamb * tsallis_log(p_j, q_value) \
                    - lamb * tsallis_log(p_i, q_value))
  # loss = - p_i_j * (tsallis_log(p_i_j, q_value) \
  #                   - lamb * torch.log(p_j) \
  #                   - lamb * torch.log(p_i))

  loss = loss.sum()
  return loss, q_value