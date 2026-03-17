import functools
import numbers
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._scaling_factor import scaling_factor
from ._single_column_transformer import SingleColumnTransformer
from ._to_str import ToStr
from ._utils import import_optional_dependency, unique_strings
from .datasets._utils import get_data_dir


class ModelNotFound(ValueError):
    pass


class ClassPriorTextEncoder(SingleColumnTransformer):
    """Encode string features by applying a pretrained language model \
        downloaded from the HuggingFace Hub.

    This is a thin wrapper around :class:`~sentence_transformers.SentenceTransformer`
    that follows the scikit-learn API, making it usable within a scikit-learn pipeline.

    .. warning::

        To use this class, you need to install the optional ``transformers``
        dependencies for skrub. See the "deep learning dependencies" section
        in the :ref:`installation_instructions` guide for more details.

    Parameters
    ----------
    model_name : str, default="intfloat/e5-small-v2"

        - If a filepath on disk is passed, this class loads the model from that path.
        - Otherwise, it first tries to download a pre-trained
          :class:`~sentence_transformers.SentenceTransformer` model.
          If that fails, tries to construct a model from Huggingface models repository
          with that name.

        The following models have a good performance/memory usage tradeoff:

        - ``intfloat/e5-small-v2``
        - ``all-MiniLM-L6-v2``
        - ``all-mpnet-base-v2``

        You can find more options on the `sentence-transformers documentation
        <https://www.sbert.net/docs/pretrained_models.html#model-overview>`_.

        The default model is a shrunk version of e5-v2, which has shown good
        performance in the benchmark of [1]_.

    n_components : int or None, default=30,
        The number of embedding dimensions. As the number of dimensions is different
        across embedding models, this class uses a :class:`~sklearn.decomposition.PCA`
        to set the number of embedding to ``n_components`` during ``transform``.
        Set ``n_components=None`` to skip the PCA dimension reduction mechanism.

        See [1]_ for more details on the choice of the PCA and default
        ``n_components``.

    device : str, default=None
        Device (e.g. "cpu", "cuda", "mps") that should be used for computation.
        If None, checks if a GPU can be used.
        Note that macOS ARM64 users can enable the GPU on their local machine
        by setting ``device="mps"``.

    batch_size : int, default=32
        The batch size to use during ``transform``.

    token_env_variable : str, default=None
        The name of the environment variable which stores your HuggingFace
        authentication token to download private models.
        Note that we only store the name of the variable but not the token itself.

    cache_folder : str, default=None
        Path to store models. By default ``~/skrub_data``.
        See :func:`skrub.datasets._utils.get_data_dir`.
        Note that when unpickling ``TextEncoder`` on another machine,
        the ``cache_folder`` path needs to be accessible to store the downloaded model.

    store_weights_in_pickle : bool, default=False
        Whether or not to keep the loaded sentence-transformers model
        in the ``TextEncoder`` when pickling.

        - When set to False, the ``_estimator`` property is removed from
          the object to pickle, which significantly reduces the size of
          the serialized object. Note that when the serialized object is
          unpickled on another machine, the ``TextEncoder`` will try to download
          the sentence-transformer model again from HuggingFace Hub.
          This process could fail if, for example, the machine doesn't have
          internet access. Additionally, if you use weights stored on disk
          that are *not* on the HuggingFace Hub (by passing a path to
          ``model_name``), these weights will not be pickled either.
          Therefore you would need to copy them to the machine where you
          unpickle the ``TextEncoder``.
        - When set to True, the ``_estimator`` property is included in
          the serialized object. Users deploying fine-tuned models stored on
          disk are recommended to use this option. Note that the machine
          where the ``TextEncoder`` is unpickled must have the same device than
          the machine where it was pickled.

    random_state : int, RandomState instance or None, default=None
        Used when the PCA dimension reduction mechanism is used, for reproducible
        results across multiple function calls.

    verbose : bool, default=True
        Verbose level, controls whether to show a progress bar or not during
        ``transform``.

    Attributes
    ----------
    input_name_ : str
        The name of the fitted column, or "text_enc" if the column has no name.

    pca_ : sklearn.decomposition.PCA
        A fitted PCA to reduce the embedding dimensionality (either PCA or truncation,
        see the ``n_components`` parameter).

    n_components_ : int
        The number of dimensions of the embeddings after dimensionality
        reduction.

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    GapEncoder :
        Encode string columns by constructing latent topics.
    StringEncoder
        Fast n-gram encoding of string columns.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.

    Notes
    -----
    This class uses a pre-trained model, so calling ``fit`` or ``fit_transform``
    will not train or fine-tune the model. Instead, the model is loaded from disk,
    and a PCA is fitted to reduce the dimension of the language model's output,
    if ``n_components`` is not None.

    When PCA is disabled, this class is essentially stateless, with loading the
    pre-trained model from disk being the only difference between ``fit_transform``
    and ``transform``.

    Be aware that parallelizing this class (e.g., using
    :class:`~skrub.TableVectorizer` with ``n_jobs`` > 1) may be computationally
    expensive. This is because a copy of the pre-trained model is loaded into memory
    for each thread. Therefore, we recommend you to let the default n_jobs=None
    (or set to 1) of the TableVectorizer and let pytorch handle parallelism.

    If memory usage is a concern, check the characteristics of your selected model.

    References
    ----------
    .. [1]  L. Grinsztajn, M. Kim, E. Oyallon, G. Varoquaux
            "Vectorizing string entries for data processing on tables: when are larger
            language models better?", 2023.
            https://hal.science/hal-04345931

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import TextEncoder

    Let's encode video comments using only 2 embedding dimensions:

    >>> enc = TextEncoder(
    ...    model_name='intfloat/e5-small-v2', n_components=2
    ... )
    >>> X = pd.Series([
    ...   "The professor snatched a good interview out of the jaws of these questions.",
    ...   "Bookmarking this to watch later.",
    ...   "When you don't know the lyrics of the song except the chorus",
    ... ], name='video comments')

    Fitting does not train the underlying pre-trained deep-learning model,
    but ensure various checks and enable dimension reduction.

    >>> enc.fit_transform(X) # doctest: +SKIP
       video comments_0  video comments_1
    0          0.411395          0.096504
    1         -0.105210         -0.344567
    2         -0.306184          0.248063
    """

    def __init__(
        self,
        model_name="intfloat/e5-small-v2",
        n_components=30,
        device=None,
        batch_size=32,
        token_env_variable=None,
        cache_folder=None,
        store_weights_in_pickle=False,
        random_state=None,
        verbose=False,
        prior_strength=0.2,
        unknown_category="global",
        train_projection_head=True,
        projection_n_epochs=30,
        projection_lr=1e-2,
        projection_weight_decay=0.0,
        projection_output_dim=None,
        separation_margin=0.5,
        separation_weight=1.0,
        pair_loss_top_m=32,
        pair_loss_prefilter_factor=4,
        pair_loss_warmup_epochs=3,
        pair_loss_eps=1e-8,
        pull_level_weights=None,
        sibling_repulsion_weight=1.0,
        inter_repulsion_weight=0.2,
        inter_pair_loss_top_m=None,
        sibling_parent_variance_beta=1.0,
    ):
        self.model_name = model_name
        self.n_components = n_components
        self.device = device
        self.batch_size = batch_size
        self.token_env_variable = token_env_variable
        self.cache_folder = cache_folder
        self.store_weights_in_pickle = store_weights_in_pickle
        self.random_state = random_state
        self.verbose = verbose
        self.prior_strength = prior_strength
        self.unknown_category = unknown_category
        self.train_projection_head = train_projection_head
        self.projection_n_epochs = projection_n_epochs
        self.projection_lr = projection_lr
        self.projection_weight_decay = projection_weight_decay
        self.projection_output_dim = projection_output_dim
        self.separation_margin = separation_margin
        self.separation_weight = separation_weight
        self.pair_loss_top_m = pair_loss_top_m
        self.pair_loss_prefilter_factor = pair_loss_prefilter_factor
        self.pair_loss_warmup_epochs = pair_loss_warmup_epochs
        self.pair_loss_eps = pair_loss_eps
        self.pull_level_weights = pull_level_weights
        self.sibling_repulsion_weight = sibling_repulsion_weight
        self.inter_repulsion_weight = inter_repulsion_weight
        self.inter_pair_loss_top_m = inter_pair_loss_top_m
        self.sibling_parent_variance_beta = sibling_parent_variance_beta

    def fit_transform(self, column, y=None):
        """Fit the TextEncoder from ``column``.

        In practice, it loads the pre-trained model from disk and returns
        the embeddings of the column.

        Parameters
        ----------
        column : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        for attr in (
            "class_priors_",
            "global_prior_",
            "projection_weight_",
            "projection_bias_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

        self.to_str = ToStr(convert_category=True)
        column = self.to_str.fit_transform(column)

        self._check_params()

        self.input_name_ = sbd.name(column) or "text_enc"

        X_out = self._vectorize(column)

        if self.projection_output_dim is None and self.n_components is not None:
            if (min_shape := min(X_out.shape)) >= self.n_components:
                self.pca_ = PCA(
                    n_components=self.n_components,
                    copy=False,
                    random_state=self.random_state,
                )
                X_out = self.pca_.fit_transform(X_out)
            else:
                warnings.warn(
                    f"The matrix shape is {(X_out.shape)}, and its minimum is "
                    f"{min_shape}, which is too small to fit a PCA with "
                    f"n_components={self.n_components}. "
                    "The embeddings will be truncated by keeping the first "
                    f"{self.n_components} dimensions instead. "
                    "Set n_components=None to keep all dimensions and remove "
                    "this warning."
                )
                # self.n_components can be greater than the number
                # of dimensions of X_out.
                # Therefore, self.n_components_ below stores the resulting
                # number of dimensions of X_out.
                X_out = X_out[:, : self.n_components]

        # block normalize
        self.scaling_factor_ = scaling_factor(X_out)
        X_out /= self.scaling_factor_

        if y is None:
            if self.projection_output_dim is not None:
                raise ValueError(
                    "projection_output_dim is set but no class labels were provided in y. "
                    "This supervised projection head requires y during fit."
                )
            if self.prior_strength > 0:
                warnings.warn(
                    "No class categories were provided (y=None), so no class "
                    "prior was applied."
                )
        else:
            category_levels = self._normalize_category_levels(
                categories=y,
                n_samples=X_out.shape[0],
                arg_name="y",
            )
            classes = category_levels[-1]
            if self.projection_output_dim is not None and not self.train_projection_head:
                raise ValueError(
                    "projection_output_dim is set but train_projection_head=False. "
                    "Please enable train_projection_head to learn the projection."
                )
            if self.train_projection_head:
                X_out = self._fit_projection_head(X_out, category_levels)
            self._fit_class_priors(X_out, classes)
            X_out = self._apply_class_prior(X_out, classes)

        self.n_components_ = X_out.shape[1]
        cols = self.get_feature_names_out()
        X_out = sbd.make_dataframe_like(column, dict(zip(cols, X_out.T)))
        X_out = sbd.copy_index(column, X_out)

        return X_out

    def transform(self, column, class_categories=None):
        """Transform ``column`` using the TextEncoder.

        This method uses the embedding model loaded in memory during ``fit``
        or ``fit_transform``.

        Parameters
        ----------
        column : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        check_is_fitted(self, "_estimator")

        # Error checking at fit time is done by the ToStr transformer,
        # but after ToStr is fitted it does not check the input type anymore,
        # while we want to ensure that the input column is a string or categorical
        # so we need to add the check here.
        if not (sbd.is_string(column) or sbd.is_categorical(column)):
            raise ValueError("Input column does not contain strings.")
        column = self.to_str.transform(column)

        X_out = self._vectorize(column)

        if self.projection_output_dim is None and hasattr(self, "pca_"):
            X_out = self.pca_.transform(X_out)
        elif self.projection_output_dim is None and self.n_components is not None:
            X_out = X_out[:, : self.n_components]

        # block scale
        X_out /= self.scaling_factor_
        X_out = self._apply_projection_head(X_out)
        if self.prior_strength != 0 and hasattr(self, "class_priors_"):
            if class_categories is None:
                warnings.warn(
                    "No class categories were provided at transform time, so no class "
                    "prior was applied."
                )
            else:
                category_levels = self._normalize_category_levels(
                    categories=class_categories,
                    n_samples=X_out.shape[0],
                    arg_name="class_categories",
                )
                classes = category_levels[-1]
                X_out = self._apply_class_prior(X_out, classes)

        cols = self.get_feature_names_out()
        X_out = sbd.make_dataframe_like(column, dict(zip(cols, X_out.T)))
        X_out = sbd.copy_index(column, X_out)

        return X_out

    def _normalize_categories(self, categories, n_samples, arg_name):
        classes = np.asarray(categories, dtype=object).ravel()
        if classes.shape[0] != n_samples:
            raise ValueError(
                f"`{arg_name}` has length {classes.shape[0]} but expected {n_samples}."
            )
        return classes

    def _normalize_category_levels(self, categories, n_samples, arg_name):
        arr = np.asarray(categories, dtype=object)
        if arr.ndim == 1:
            levels = [arr.ravel()]
        elif arr.ndim == 2 and arr.shape[1] >= 1:
            levels = [arr[:, idx].ravel() for idx in range(arr.shape[1])]
        else:
            raise ValueError(
                f"`{arg_name}` must be 1D labels or 2D hierarchical labels. "
                f"Got shape {arr.shape}."
            )
        for level in levels:
            if level.shape[0] != n_samples:
                raise ValueError(
                    f"`{arg_name}` has length {level.shape[0]} but expected {n_samples}."
                )
        return levels

    def _fit_class_priors(self, X_out, classes):
        X_np = np.asarray(X_out, dtype=float)
        null_mask = self._is_null(classes)

        self.class_priors_ = {}
        self.global_prior_ = X_np.mean(axis=0)

        if null_mask.all():
            warnings.warn(
                "All class categories are missing, so no class-dependent prior "
                "could be learned."
            )
            return

        valid_classes = classes[~null_mask]
        for class_name in dict.fromkeys(valid_classes.tolist()):
            mask = classes == class_name
            self.class_priors_[class_name] = X_np[mask].mean(axis=0)

    def _fit_projection_head(self, X_out, category_levels):
        torch = import_optional_dependency(
            "torch",
            extra=(
                "ClassPriorTextEncoder with train_projection_head=True requires "
                "PyTorch."
            ),
        )

        X_np = np.asarray(X_out, dtype=np.float32)
        _, n_features = X_np.shape
        output_dim = (
            int(self.projection_output_dim)
            if self.projection_output_dim is not None
            else int(n_features)
        )
        leaf_classes = category_levels[-1]
        null_mask = self._is_null(leaf_classes)
        if null_mask.all():
            warnings.warn(
                "All class categories are missing, projection-head training was skipped."
            )
            return X_np

        valid_mask = ~null_mask
        X_train = X_np[valid_mask]
        levels_train = [level[valid_mask] for level in category_levels]

        level_ids = []
        level_n_classes = []
        for level_values in levels_train:
            ids = np.full(level_values.shape[0], -1, dtype=np.int64)
            mapping = {}
            for i, class_name in enumerate(level_values):
                if class_name not in mapping:
                    mapping[class_name] = len(mapping)
                ids[i] = mapping[class_name]
            level_ids.append(ids)
            level_n_classes.append(len(mapping))

        # Build leaf->parent ids for sibling selection at the deepest level.
        if len(level_ids) >= 2:
            leaf_to_parent = np.full(level_n_classes[-1], -1, dtype=np.int64)
            for row_idx, leaf_id in enumerate(level_ids[-1]):
                if leaf_to_parent[leaf_id] == -1:
                    leaf_to_parent[leaf_id] = level_ids[-2][row_idx]
        else:
            leaf_to_parent = np.full(level_n_classes[-1], -1, dtype=np.int64)

        level_ids_t = [torch.from_numpy(ids) for ids in level_ids]
        pull_weights = self._resolve_pull_level_weights(len(level_ids))

        X_t = torch.from_numpy(X_train)
        y_t = level_ids_t[-1]
        head = torch.nn.Linear(n_features, output_dim)
        with torch.no_grad():
            head.weight.zero_()
            diag_size = min(n_features, output_dim)
            head.weight[:diag_size, :diag_size] = torch.eye(diag_size)
            head.bias.zero_()

        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=float(self.projection_lr),
            weight_decay=float(self.projection_weight_decay),
        )

        sep_weight = float(self.separation_weight)
        eps = float(self.pair_loss_eps)
        sibling_weight = float(self.sibling_repulsion_weight)
        inter_weight = float(self.inter_repulsion_weight)
        parent_var_beta = float(self.sibling_parent_variance_beta)
        progress_iter = range(int(self.projection_n_epochs))
        progress_bar = None
        if self.verbose:
            tqdm_auto = import_optional_dependency(
                "tqdm.auto",
                extra=(
                    "ClassPriorTextEncoder with verbose=True requires tqdm for "
                    "training progress display."
                ),
            )
            progress_bar = tqdm_auto.tqdm(
                progress_iter,
                desc="Projection head training",
                leave=False,
            )
            progress_iter = progress_bar

        for epoch in progress_iter:
            optimizer.zero_grad()
            Z = head(X_t)

            level_centroids = []
            pull_loss = torch.tensor(0.0, dtype=Z.dtype, device=Z.device)
            for level_idx, ids_t in enumerate(level_ids_t):
                n_classes_level = level_n_classes[level_idx]
                centroids = []
                for class_id in range(n_classes_level):
                    class_mask = ids_t == class_id
                    centroids.append(Z[class_mask].mean(dim=0))
                C_level = torch.stack(centroids, dim=0)
                level_centroids.append(C_level)
                z_targets = C_level[ids_t]
                level_pull = torch.mean(torch.sum((Z - z_targets) ** 2, dim=1))
                pull_loss = pull_loss + float(pull_weights[level_idx]) * level_pull

            C = level_centroids[-1]

            sep_loss = torch.tensor(0.0, dtype=Z.dtype)
            n_classes = C.shape[0]
            if n_classes > 1:
                pair_idx = torch.triu_indices(n_classes, n_classes, offset=1)
                left_idx = pair_idx[0]
                right_idx = pair_idx[1]
                pair_diff = C[left_idx] - C[right_idx]
                pair_dist = torch.linalg.norm(pair_diff, dim=1)
                n_pairs = int(pair_dist.shape[0])

                use_all_pairs = (
                    epoch < int(self.pair_loss_warmup_epochs)
                    or int(self.pair_loss_top_m) <= 0
                    or int(self.pair_loss_top_m) >= n_pairs
                )
                candidate_idx = torch.arange(n_pairs, device=pair_dist.device)
                if not use_all_pairs:
                    pre_n = min(
                        n_pairs,
                        max(
                            int(self.pair_loss_top_m),
                            int(self.pair_loss_prefilter_factor)
                            * int(self.pair_loss_top_m),
                        ),
                    )
                    # Closest centroid pairs are a cheap hardness proxy.
                    candidate_idx = torch.topk(
                        pair_dist, k=pre_n, largest=False
                    ).indices

                sibling_scores = []
                inter_scores = []
                for idx in candidate_idx:
                    i = left_idx[idx]
                    j = right_idx[idx]
                    diff = C[i] - C[j]
                    dist = torch.linalg.norm(diff) + eps
                    direction = diff / dist

                    centered_i = Z[y_t == i] - C[i]
                    centered_j = Z[y_t == j] - C[j]
                    proj_i = centered_i @ direction
                    proj_j = centered_j @ direction
                    std_i = torch.sqrt(torch.mean(proj_i * proj_i) + eps)
                    std_j = torch.sqrt(torch.mean(proj_j * proj_j) + eps)
                    base_score = (std_i + std_j) / dist

                    i_leaf = int(i.item())
                    j_leaf = int(j.item())
                    same_parent = (
                        len(level_ids) >= 2
                        and leaf_to_parent[i_leaf] >= 0
                        and leaf_to_parent[i_leaf] == leaf_to_parent[j_leaf]
                    )
                    if same_parent:
                        parent_id = int(leaf_to_parent[i_leaf])
                        parent_ids = level_ids_t[-2]
                        parent_center = level_centroids[-2][parent_id]
                        parent_centered = Z[parent_ids == parent_id] - parent_center
                        parent_proj = parent_centered @ direction
                        parent_var = torch.mean(parent_proj * parent_proj)
                        mod = 1.0 / (1.0 + parent_var_beta * parent_var)
                        sibling_scores.append(base_score * mod)
                    else:
                        inter_scores.append(base_score)

                sibling_loss = self._aggregate_hard_pair_scores(
                    torch.stack(sibling_scores) if sibling_scores else None,
                    top_m=int(self.pair_loss_top_m),
                    use_all_pairs=use_all_pairs,
                )
                if sibling_loss is None:
                    sibling_loss = Z.new_tensor(0.0)
                inter_top_m = (
                    int(self.inter_pair_loss_top_m)
                    if self.inter_pair_loss_top_m is not None
                    else max(1, int(self.pair_loss_top_m) // 4)
                )
                inter_loss = self._aggregate_hard_pair_scores(
                    torch.stack(inter_scores) if inter_scores else None,
                    top_m=inter_top_m,
                    use_all_pairs=use_all_pairs,
                )
                if inter_loss is None:
                    inter_loss = Z.new_tensor(0.0)
                sep_loss = sibling_weight * sibling_loss + inter_weight * inter_loss

            loss = pull_loss + sep_weight * sep_loss
            loss.backward()
            optimizer.step()
            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    pull=f"{float(pull_loss.item()):.4f}",
                    sep=f"{float(sep_loss.item()):.4f}",
                )

        if progress_bar is not None:
            progress_bar.close()

        with torch.no_grad():
            projected = head(torch.from_numpy(X_np)).cpu().numpy().astype(float, copy=False)
            self.projection_weight_ = (
                head.weight.detach().cpu().numpy().astype(float, copy=True)
            )
            self.projection_bias_ = (
                head.bias.detach().cpu().numpy().astype(float, copy=True)
            )
        return projected

    def _apply_projection_head(self, X_out):
        if not hasattr(self, "projection_weight_"):
            return X_out
        X_np = np.asarray(X_out, dtype=float)
        return X_np @ self.projection_weight_.T + self.projection_bias_

    def _aggregate_hard_pair_scores(self, scores, top_m, use_all_pairs):
        if scores is None or int(scores.shape[0]) == 0:
            if scores is not None:
                return scores.new_tensor(0.0)
            return None
        if use_all_pairs or top_m <= 0 or top_m >= int(scores.shape[0]):
            return scores.mean()
        return scores.topk(k=top_m, largest=True).values.mean()

    def _resolve_pull_level_weights(self, n_levels):
        if self.pull_level_weights is None:
            return np.arange(1.0, n_levels + 1.0, dtype=float)
        weights = np.asarray(self.pull_level_weights, dtype=float).ravel()
        if weights.size == 1:
            return np.repeat(weights, n_levels)
        if weights.size != n_levels:
            raise ValueError(
                f"pull_level_weights has length {weights.size}, expected 1 or {n_levels}."
            )
        return weights

    def _apply_class_prior(self, X_out, classes):
        alpha = float(self.prior_strength)
        if alpha == 0:
            return X_out

        out = np.asarray(X_out, dtype=float).copy()
        null_mask = self._is_null(classes)
        for idx, class_name in enumerate(classes):
            if null_mask[idx] or class_name not in self.class_priors_:
                if self.unknown_category == "ignore":
                    continue
                if self.unknown_category == "zero":
                    prior = np.zeros(out.shape[1], dtype=out.dtype)
                else:
                    prior = self.global_prior_
            else:
                prior = self.class_priors_[class_name]
            out[idx] = (1.0 - alpha) * out[idx] + alpha * prior
        return out

    def _is_null(self, values):
        null_mask = []
        for value in values:
            if value is None:
                null_mask.append(True)
                continue
            try:
                is_self_unequal = value != value
                if isinstance(is_self_unequal, (bool, np.bool_)):
                    null_mask.append(bool(is_self_unequal))
                    continue
            except Exception:
                pass
            null_mask.append(False)
        return np.asarray(null_mask, dtype=bool)

    def _vectorize(self, column):
        is_null = sbd.to_numpy(sbd.is_null(column))
        column = sbd.to_numpy(column)
        unique_x, indices_x = unique_strings(column, is_null)

        # sentence-transformers deals with converting a torch tensor
        # to a numpy array, on CPU.
        return self._estimator.encode(
            unique_x,
            normalize_embeddings=False,
            batch_size=self.batch_size,
            show_progress_bar=self.verbose,
        )[indices_x]

    @functools.cached_property
    def _estimator(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*IProgress not found.*",
            )
            st = import_optional_dependency(
                "sentence_transformers",
                extra=(
                    "The TextEncoder requires sentence-transformers and its"
                    " dependencies. Please see"
                    " https://skrub-data.org/stable/install.html#deep-learning-dependencies"
                    " for the installation guide."
                ),
            )

        self._cache_folder = get_data_dir(
            name=self.model_name, data_home=self.cache_folder
        )

        if self.token_env_variable is not None:
            token = os.getenv(self.token_env_variable)
        else:
            token = None

        try:
            estimator = st.SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self._cache_folder,
                token=token,
            )
        except OSError as e:
            raise ModelNotFound(
                f"{self.model_name} is not a local folder and is not a valid "
                "model identifier listed on 'https://huggingface.co/models'.\n "
                "If this is a private repository, make sure to pass a token having "
                "permission to this repo by setting this token as an environment "
                "variable, and passing this variable to the TextEncoder as "
                "`token_env_variable=<your_token_env_variable>`"
            ) from e
        return estimator

    def _check_params(self):
        # XXX: Use sklearn _parameter_constraints instead?
        if self.n_components is not None and not isinstance(
            self.n_components, numbers.Integral
        ):
            raise ValueError(
                f"Got n_components={self.n_components!r} but expected an integer "
                "or None."
            )

        if not (isinstance(self.batch_size, numbers.Integral) and self.batch_size > 0):
            raise ValueError(
                f"Got batch_size={self.batch_size} but expected a positive integer"
            )

        if self.cache_folder is not None and not isinstance(
            self.cache_folder, (str, bytes, Path)
        ):
            raise ValueError(
                f"Got cache_folder={self.cache_folder} but expected a "
                "str, bytes or Path type."
            )

        if not isinstance(self.model_name, (str, Path)):
            raise ValueError(
                f"Got model_name={self.model_name} but expected a str or a Path type."
            )
        self._check_prior_params()
        return

    def _check_prior_params(self):
        if not isinstance(self.prior_strength, numbers.Real):
            raise ValueError(
                f"Got prior_strength={self.prior_strength!r} but expected a float "
                "in [0, 1]."
            )
        if not (0.0 <= float(self.prior_strength) <= 1.0):
            raise ValueError(
                f"Got prior_strength={self.prior_strength!r} but expected a float "
                "in [0, 1]."
            )
        if self.unknown_category not in {"global", "zero", "ignore"}:
            raise ValueError(
                "Got unknown_category="
                f"{self.unknown_category!r} but expected one of "
                "{'global', 'zero', 'ignore'}."
            )
        if not isinstance(self.train_projection_head, bool):
            raise ValueError(
                "Got train_projection_head="
                f"{self.train_projection_head!r} but expected a boolean."
            )
        if not (
            isinstance(self.projection_n_epochs, numbers.Integral)
            and self.projection_n_epochs > 0
        ):
            raise ValueError(
                "Got projection_n_epochs="
                f"{self.projection_n_epochs!r} but expected a positive integer."
            )
        if not (isinstance(self.projection_lr, numbers.Real) and self.projection_lr > 0):
            raise ValueError(
                f"Got projection_lr={self.projection_lr!r} but expected a positive float."
            )
        if not (
            isinstance(self.projection_weight_decay, numbers.Real)
            and self.projection_weight_decay >= 0
        ):
            raise ValueError(
                "Got projection_weight_decay="
                f"{self.projection_weight_decay!r} but expected a non-negative float."
            )
        if self.projection_output_dim is not None and not (
            isinstance(self.projection_output_dim, numbers.Integral)
            and self.projection_output_dim > 0
        ):
            raise ValueError(
                "Got projection_output_dim="
                f"{self.projection_output_dim!r} but expected a positive integer or None."
            )
        if not (
            isinstance(self.separation_margin, numbers.Real)
            and self.separation_margin >= 0
        ):
            raise ValueError(
                f"Got separation_margin={self.separation_margin!r} but expected a non-negative float."
            )
        if not (
            isinstance(self.separation_weight, numbers.Real)
            and self.separation_weight >= 0
        ):
            raise ValueError(
                f"Got separation_weight={self.separation_weight!r} but expected a non-negative float."
            )
        if not isinstance(self.pair_loss_top_m, numbers.Integral):
            raise ValueError(
                f"Got pair_loss_top_m={self.pair_loss_top_m!r} but expected an integer."
            )
        if not (
            isinstance(self.pair_loss_prefilter_factor, numbers.Integral)
            and self.pair_loss_prefilter_factor > 0
        ):
            raise ValueError(
                "Got pair_loss_prefilter_factor="
                f"{self.pair_loss_prefilter_factor!r} but expected a positive integer."
            )
        if not (
            isinstance(self.pair_loss_warmup_epochs, numbers.Integral)
            and self.pair_loss_warmup_epochs >= 0
        ):
            raise ValueError(
                "Got pair_loss_warmup_epochs="
                f"{self.pair_loss_warmup_epochs!r} but expected a non-negative integer."
            )
        if not (isinstance(self.pair_loss_eps, numbers.Real) and self.pair_loss_eps > 0):
            raise ValueError(
                f"Got pair_loss_eps={self.pair_loss_eps!r} but expected a positive float."
            )
        if self.pull_level_weights is not None:
            weights = np.asarray(self.pull_level_weights, dtype=float).ravel()
            if weights.size == 0:
                raise ValueError("pull_level_weights must not be empty.")
            if np.any(~np.isfinite(weights)):
                raise ValueError(
                    "pull_level_weights must contain only finite numeric values."
                )
        if not (
            isinstance(self.sibling_repulsion_weight, numbers.Real)
            and self.sibling_repulsion_weight >= 0
        ):
            raise ValueError(
                "Got sibling_repulsion_weight="
                f"{self.sibling_repulsion_weight!r} but expected a non-negative float."
            )
        if not (
            isinstance(self.inter_repulsion_weight, numbers.Real)
            and self.inter_repulsion_weight >= 0
        ):
            raise ValueError(
                "Got inter_repulsion_weight="
                f"{self.inter_repulsion_weight!r} but expected a non-negative float."
            )
        if self.inter_pair_loss_top_m is not None and not (
            isinstance(self.inter_pair_loss_top_m, numbers.Integral)
            and self.inter_pair_loss_top_m > 0
        ):
            raise ValueError(
                "Got inter_pair_loss_top_m="
                f"{self.inter_pair_loss_top_m!r} but expected a positive integer or None."
            )
        if not (
            isinstance(self.sibling_parent_variance_beta, numbers.Real)
            and self.sibling_parent_variance_beta >= 0
        ):
            raise ValueError(
                "Got sibling_parent_variance_beta="
                f"{self.sibling_parent_variance_beta!r} but expected a non-negative float."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Always dump self._cache_folder because it is overwritten when the model
        # is loaded, and it shows an absolute path on the user machine.
        # However, we have to include self.cache_folder in the serialized object
        # because that is a parameter provided by the user.
        remove_props = ["_cache_folder"]
        if not self.store_weights_in_pickle:
            remove_props.append("_estimator")

        for prop in remove_props:
            if prop in state:
                del state[prop]

        return state

    def get_feature_names_out(self, input_features=None):
        """Return a list of features generated by the transformer.

        Each feature has format ``{input_name}_{n_component}`` where ``input_name``
        is the name of the input column, or a default name for the encoder, and
        ``n_component`` is the idx of the specific feature.

        Parameters
        ----------
        input_features : None
            The input features. Ignored, only here for compatibility.

        Returns
        -------
        list of str
            The list of feature names.
        """
        check_is_fitted(self, "n_components_")
        num_digits = len(str(self.n_components_ - 1))
        return [
            f"{self.input_name_}_{str(i).zfill(num_digits)}"
            for i in range(self.n_components_)
        ]
