from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRecord:
    algorithm: str
    family: str
    dataset_scope: str
    goal: str
    status: str
    milestone: str
    note: str = ""


EXPERIMENTS = [
    ExperimentRecord("Popularity", "baseline", "all", "sanity-check baseline", "ready", "phase_1"),
    ExperimentRecord("TruncatedSVD", "matrix factorization", "all", "compact latent factor baseline", "ready", "phase_1"),
    ExperimentRecord("EASE", "linear CF", "all", "closed-form co-occurrence benchmark", "ready", "phase_1"),
    ExperimentRecord("TFIDF_Content", "content-based", "all", "metadata and text retrieval baseline", "ready", "phase_1"),
    ExperimentRecord("SequentialTransition", "sequential", "all", "lightweight order-aware baseline without heavy deep stack", "ready", "phase_1"),
    ExperimentRecord(
        "HybridFeatureRerank",
        "hybrid",
        "MovieLens 20M / Google Local South Carolina / Goodreads Fantasy 10K",
        "feature-engineered rerank with collaborative, content, recency and tag overlap signals",
        "ready",
        "phase_1",
    ),
    ExperimentRecord(
        "Hybrid_NoContentAblation",
        "hybrid ablation",
        "MovieLens 20M / Google Local South Carolina / Goodreads Fantasy 10K",
        "quantify the contribution of content and profile-alignment signals",
        "ready",
        "phase_1",
    ),
    ExperimentRecord(
        "Hybrid_NoCollaborativeAblation",
        "hybrid ablation",
        "MovieLens 20M / Google Local South Carolina / Goodreads Fantasy 10K",
        "quantify the contribution of collaborative candidate signals",
        "ready",
        "phase_1",
    ),
    ExperimentRecord("NCF", "deep", "MovieLens 20M / Goodreads Fantasy 10K", "secondary deep reference, not main contribution", "planned", "phase_2"),
    ExperimentRecord("SASRec", "sequential", "MovieLens 20M / Google Local South Carolina / Goodreads Fantasy 10K", "single selective sequential benchmark if budget allows", "planned", "phase_2"),
    ExperimentRecord("GRU4Rec", "sequential RNN", "Google Local South Carolina / Goodreads Fantasy 10K", "fallback sequential alternative to SASRec, not parallel mandatory scope", "planned", "phase_2"),
    ExperimentRecord("RBM", "legacy deep", "MovieLens 20M", "historical notebook reference only", "draft", "legacy"),
]
