from sklearn.metrics import homogeneity_score

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