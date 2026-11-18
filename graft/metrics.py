import torch


def _mask(labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    return torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)


def _reduce(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val):
    return _reduce((preds - labels) ** 2, _mask(labels, null_val))


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    return _reduce(torch.abs(preds - labels), _mask(labels, null_val))


def masked_mape(preds, labels, null_val):
    return _reduce(torch.abs(preds - labels) / labels, _mask(labels, null_val))


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse
