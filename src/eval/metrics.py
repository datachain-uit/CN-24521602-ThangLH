from sklearn.metrics import homogeneity_score, normalized_mutual_info_score

def best_community_f1(
    communities: list[set[str]],
    ground_truth: set[str],
) -> tuple[float, int]:
    best_f1 = 0.0
    best_index = -1

    for index, community in enumerate(communities):
        tp = len(community & ground_truth)
        fp = len(community - ground_truth)
        fn = len(ground_truth - community)

        denominator = 2 * tp + fp + fn

        f1 = (
            2 * tp / denominator
            if denominator > 0
            else 0.0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_index = index

    return best_f1, best_index

def weighted_precision(
    communities: list[set[str]],
    ground_truth: set[str],
) -> float:
    numerator = 0.0
    denominator = 0.0

    for community in communities:
        size = len(community)

        if size == 1:
            p = 0.0
        elif size > 1:
            p = len(community & ground_truth) / size
        else:
            continue

        numerator += size * p**2
        denominator += size * p

    return numerator / denominator if denominator > 0 else 0.0

def clustering_homogeneity(
    communities: list[set[str]],
    ground_truth: set[str],
) -> float:
    gt_labels = []
    comm_labels = []

    for community_id, members in enumerate(communities):
        for user_id in sorted(members):
            gt_labels.append(
                1 if user_id in ground_truth else 0
            )
            comm_labels.append(community_id)

    return float(
        homogeneity_score(gt_labels, comm_labels)
    )


def best_cluster_scores(communities: list[set[str]], ground_truth: set[str]) -> tuple[float, float, float]:
    """F1*, Precision*, Recall* của cụm có F1 lớn nhất như scores.py:1690-1733; hòa thì giữ cụm đầu tiên (so sánh f1 > best_f1)."""
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for community in communities:
        tp = len(community & ground_truth)
        precision = tp / len(community) if community else 0.0
        recall = tp / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, precision, recall
    return best_f1, best_p, best_r


def _labels(communities: list[set[str]], ground_truth: set[str]) -> tuple[list[int], list[int], list[int]]:
    """gt_labels, comm_labels (scores.py:1735-1738) và nhãn share >= 0.5 cho cụm >= 2 phần tử (scores.py:1740-1754)."""
    gt, comm, share = [], [], []
    for community_id, members in enumerate(communities):
        tp = len(members & ground_truth)
        flag = 1 if len(members) > 1 and tp / len(members) >= 0.5 else 0
        for user_id in sorted(members):
            gt.append(1 if user_id in ground_truth else 0)
            comm.append(community_id)
            share.append(flag)
    return gt, comm, share


def clustering_nmi(communities: list[set[str]], ground_truth: set[str]) -> tuple[float, float]:
    """(NMI theo nhãn cụm, scores.py:1761; NMI50 theo nhãn share >= 0.5, scores.py:1772)."""
    gt, comm, share = _labels(communities, ground_truth)
    return float(normalized_mutual_info_score(gt, comm)), float(normalized_mutual_info_score(gt, share))


def partition_metrics(partition: list[set[str]], population: set[str], ground_truth: set[str]) -> dict:
    """Giới hạn phân hoạch và nhãn vào tập đánh giá tại tau (bỏ cụm rỗng, giữ thứ tự cụm) rồi tính mọi chỉ số."""
    restricted = [community & population for community in partition if community & population]
    truth = ground_truth & population
    f1, precision, recall = best_cluster_scores(restricted, truth)
    nmi, nmi50 = clustering_nmi(restricted, truth)
    return dict(f1_star=f1, precision_star=precision, recall_star=recall, wp=weighted_precision(restricted, truth), homogeneity=clustering_homogeneity(restricted, truth), nmi=nmi, nmi50=nmi50)


def expected_random_wp(n_io: int, n_ctrl: int, n_clusters: int) -> float:
    """Kỳ vọng giải tích của WP chuẩn với nhãn ngẫu nhiên đều K cụm: E[WP] = π + (1 − π)(K − 1)/(N − 1), π = N_IO/N.

    Mẫu số Σ nₖpₖ = N_IO là hằng; E[nₖpₖ²] = nₖπ² + π(1−π)(N−nₖ)/(N−1). Bỏ qua cụm rỗng/cụm 1 phần tử (p = 0). N_IO = 0 thì WP = 0 theo định nghĩa.
    """
    n = n_io + n_ctrl
    if n < 2 or n_clusters < 1 or n_io < 0 or n_ctrl < 0:
        raise ValueError("Cần N >= 2, K >= 1 và số tài khoản không âm.")
    if n_io == 0:
        return 0.0
    pi = n_io / n
    return pi + (1 - pi) * (n_clusters - 1) / (n - 1)

