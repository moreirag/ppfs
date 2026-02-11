import math
from dataclasses import dataclass

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


EPS = 1e-12


@dataclass
class ROIConfig:
    axis: np.ndarray
    theta: float


def _ensure_axis(axis, n_obj):
    if axis is None:
        axis = np.ones(n_obj, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or axis.shape[0] != n_obj:
        raise ValueError(f"axis must have shape ({n_obj},), got {axis.shape}")
    norm = np.linalg.norm(axis)
    if norm <= EPS:
        raise ValueError("axis vector cannot be zero")
    return axis


def _crowding_distance(F):
    n_points, n_obj = F.shape
    if n_points == 0:
        return np.array([])
    if n_points <= 2:
        return np.full(n_points, np.inf)

    cd = np.zeros(n_points)
    for m in range(n_obj):
        order = np.argsort(F[:, m])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf

        f_min = F[order[0], m]
        f_max = F[order[-1], m]
        if abs(f_max - f_min) <= EPS:
            continue

        scale = f_max - f_min
        for i in range(1, n_points - 1):
            if np.isinf(cd[order[i]]):
                continue
            prev_f = F[order[i - 1], m]
            next_f = F[order[i + 1], m]
            cd[order[i]] += (next_f - prev_f) / scale

    return cd


def _roi_angles(F, axis):
    axis_norm = np.linalg.norm(axis)
    f_norm = np.linalg.norm(F, axis=1)
    denom = np.maximum(f_norm * axis_norm, EPS)
    cos_vals = np.sum(F * axis[None, :], axis=1) / denom
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    return np.arccos(cos_vals)


def _roi_front_penalty(front_no, angles, theta):
    penalized = front_no.copy()
    penalty_mask = angles > theta
    penalty = np.floor(np.exp(0.3 * (angles - theta))).astype(int)
    penalized[penalty_mask] += penalty[penalty_mask]
    return penalized, penalty_mask


def _dominance_info(F):
    n = F.shape[0]
    D = np.zeros((n, n), dtype=bool)

    for i in range(n - 1):
        fi = F[i]
        for j in range(i + 1, n):
            fj = F[j]
            i_better = np.any(fi < fj)
            i_worse = np.any(fi > fj)
            if i_better and not i_worse:
                D[i, j] = True
            elif i_worse and not i_better:
                D[j, i] = True

    count_dom = D.sum(axis=1)
    return D.T @ count_dom


def _dwu_pairwise_distance(X_dec, X_w, Y_dec, Y_w, Y_penalty, Y_angle, theta):
    dist = np.linalg.norm(X_dec - Y_dec[None, :], axis=1)
    weight_gap = np.abs(X_w - Y_w)
    score = dist / (weight_gap + 1.0)
    if Y_penalty:
        score = score - math.exp(Y_angle - theta)
    return score


def _replacement_uniformity(X, n_survive, front_no, weights, penalty_mask, angles, theta):
    n = len(X)
    if n_survive >= n:
        return np.arange(n)

    pool = list(range(n))
    selected = []

    min_front = front_no.min()
    aux = [i for i in pool if front_no[i] == min_front]

    if len(aux) == 1:
        other = [i for i in pool if front_no[i] != min_front]
        if other:
            next_front = min(front_no[i] for i in other)
            aux.extend([i for i in other if front_no[i] == next_front])

    if len(aux) < 2:
        aux = pool[: min(2, len(pool))]

    best_score = -np.inf
    best_pair = (aux[0], aux[min(1, len(aux) - 1)])
    for j in aux:
        scores = _dwu_pairwise_distance(
            X[aux],
            weights[aux],
            X[j],
            weights[j],
            bool(penalty_mask[j]),
            angles[j],
            theta,
        )
        local_i = int(np.argmax(scores))
        candidate_pair = (j, aux[local_i])
        if scores[local_i] > best_score and candidate_pair[0] != candidate_pair[1]:
            best_score = scores[local_i]
            best_pair = candidate_pair

    for idx in best_pair:
        if idx in pool and len(selected) < n_survive:
            selected.append(idx)
            pool.remove(idx)

    while len(selected) < n_survive and pool:
        best_idx = None
        best_min_dist = -np.inf

        for j in pool:
            scores = _dwu_pairwise_distance(
                X[selected],
                weights[selected],
                X[j],
                weights[j],
                bool(penalty_mask[j]),
                angles[j],
                theta,
            )
            min_dist = float(np.min(scores))
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = j

        selected.append(best_idx)
        pool.remove(best_idx)

    return np.asarray(selected, dtype=int)


class ROISelection(Selection):
    def __init__(self, criterion="crowding"):
        super().__init__()
        self.criterion = criterion

    def _do(self, problem, pop, n_select, n_parents=2, **kwargs):
        n_random = n_select * n_parents
        P = np.random.randint(0, len(pop), size=(n_random, 2))
        chosen = np.empty(n_random, dtype=int)

        rank = pop.get("roi_rank")
        metric = pop.get("roi_metric")

        for i, (a, b) in enumerate(P):
            if rank[a] < rank[b]:
                winner = a
            elif rank[b] < rank[a]:
                winner = b
            else:
                if self.criterion == "crowding":
                    if metric[a] > metric[b]:
                        winner = a
                    elif metric[b] > metric[a]:
                        winner = b
                    else:
                        winner = a if np.random.random() < 0.5 else b
                else:
                    if metric[a] < metric[b]:
                        winner = a
                    elif metric[b] < metric[a]:
                        winner = b
                    else:
                        winner = a if np.random.random() < 0.5 else b

            chosen[i] = winner

        return chosen.reshape(n_select, n_parents)


class ROINSGA2Survival(Survival):
    def __init__(self, axis=None, theta=0.3):
        super().__init__(filter_infeasible=True)
        self.axis = axis
        self.theta = float(theta)

    def _do(self, problem, pop, n_survive, **kwargs):
        F = pop.get("F")
        n_obj = F.shape[1]

        axis = _ensure_axis(self.axis, n_obj)
        fronts = NonDominatedSorting().do(F)

        base_front = np.empty(len(pop), dtype=int)
        for i, front in enumerate(fronts, start=1):
            base_front[front] = i

        angles = _roi_angles(F, axis)
        roi_front, _ = _roi_front_penalty(base_front, angles, self.theta)

        crowding = np.zeros(len(pop), dtype=float)
        for fr in np.unique(roi_front):
            idx = np.where(roi_front == fr)[0]
            crowding[idx] = _crowding_distance(F[idx])

        order = np.lexsort((-crowding, roi_front))
        survivors = order[:n_survive]

        pop.set("roi_rank", roi_front)
        pop.set("roi_metric", crowding)
        pop.set("roi_angle", angles)

        return pop[survivors]


class ROIDWUSurvival(Survival):
    def __init__(self, axis=None, theta=0.3):
        super().__init__(filter_infeasible=True)
        self.axis = axis
        self.theta = float(theta)

    def _do(self, problem, pop, n_survive, **kwargs):
        F = pop.get("F")
        X = pop.get("X")
        n_obj = F.shape[1]

        axis = _ensure_axis(self.axis, n_obj)
        fronts = NonDominatedSorting().do(F)

        base_front = np.empty(len(pop), dtype=int)
        for i, front in enumerate(fronts, start=1):
            base_front[front] = i

        angles = _roi_angles(F, axis)
        roi_front, penalty_mask = _roi_front_penalty(base_front, angles, self.theta)
        dweight = _dominance_info(F).astype(float)

        survivors = _replacement_uniformity(
            X=X,
            n_survive=n_survive,
            front_no=roi_front,
            weights=dweight,
            penalty_mask=penalty_mask,
            angles=angles,
            theta=self.theta,
        )

        pop.set("roi_rank", roi_front)
        pop.set("roi_metric", dweight)
        pop.set("roi_angle", angles)

        return pop[survivors]


class ROINSGA2(NSGA2):
    def __init__(
        self,
        pop_size=100,
        axis=None,
        theta=0.3,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=ROISelection(criterion="crowding"),
            crossover=crossover,
            mutation=mutation,
            survival=ROINSGA2Survival(axis=axis, theta=theta),
            eliminate_duplicates=eliminate_duplicates,
            **kwargs,
        )


class ROIDWUMOEA(NSGA2):
    def __init__(
        self,
        pop_size=100,
        axis=None,
        theta=0.3,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=ROISelection(criterion="dweight"),
            crossover=crossover,
            mutation=mutation,
            survival=ROIDWUSurvival(axis=axis, theta=theta),
            eliminate_duplicates=eliminate_duplicates,
            **kwargs,
        )
