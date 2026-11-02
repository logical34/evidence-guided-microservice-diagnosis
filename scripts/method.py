"""Numerically guarded, label-free score normalization and coherent release rules.

Only arrays, observable candidate names, observable quality and an LLM prior enter
these functions. Reference labels belong in the evaluator, never this module.
"""
import numpy as np


def normalize_scores(raw, logprior=None):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all() or (raw < 0).any() or not (raw > 0).any():
        return np.zeros_like(raw), False
    prior = np.zeros(raw.shape[0]) if logprior is None else np.asarray(logprior, dtype=float)
    if prior.shape != (raw.shape[0],) or not np.isfinite(prior).all():
        return np.zeros_like(raw), False
    positive = raw > 0
    z = np.full_like(raw, -np.inf)
    z[positive] = np.log(raw[positive])
    z += prior[:, None]
    z -= np.max(z)
    q = np.exp(z)
    q /= q.sum()
    return q, True


def service_prior(services, nominated='', valid=False, gamma=0.):
    prior = np.zeros(len(services))
    if valid and nominated in services:
        prior[services.index(nominated)] = gamma
    return prior


def summarize(q, services, faults, variant='selected', fixed_service=None):
    q = np.asarray(q, dtype=float)
    supported = bool(q.ndim == 2 and q.shape == (len(services), len(faults)) and q.size and
                     np.isfinite(q).all() and (q >= 0).all() and q.sum() > 0)
    if not supported:
        return dict(service='', fault='', service_score=0., joint_score=0.,
                    service_margin=-np.inf, global_margin=0., mismatch=False, supported=False)
    # Sorted observable candidate order makes ties deterministic and label-blind.
    marginal = q.sum(axis=1)
    li, lj = np.unravel_index(np.argmax(q), q.shape)
    mi = int(np.argmax(marginal))
    if fixed_service is not None:
        if fixed_service not in services:
            return dict(service='', fault='', service_score=0., joint_score=0.,
                        service_margin=-np.inf, global_margin=0., mismatch=li != mi, supported=False)
        i = services.index(fixed_service)
        j = int(np.argmax(q[i]))
    elif variant == 'service_first':
        i, j = mi, int(np.argmax(q[mi]))
    else:
        i, j = li, lj
    others = np.delete(marginal, i)
    signed = float(marginal[i] - (others.max() if len(others) else 0.))
    order = np.sort(marginal)
    global_margin = float(order[-1] - order[-2]) if len(order) > 1 else float(order[-1])
    return dict(service=services[i], fault=faults[j], service_score=float(marginal[i]),
                joint_score=float(q[i, j]), service_margin=global_margin if variant == 'global' else signed,
                global_margin=global_margin, selected_margin=signed, mismatch=li != mi,
                marginal_service=services[mi], supported=True)


def release(record, rule):
    fields = ['service_score', 'joint_score', 'service_margin']
    if not record.get('valid', False) or not record.get('supported', True) or not all(np.isfinite(record[k]) for k in fields):
        return 'refer'
    if record['service_score'] < rule['service_threshold'] or record['service_margin'] < rule['margin']:
        return 'refer'
    return 'joint' if record['joint_score'] >= rule['joint_threshold'] else 'service'
