# agents/ml_dl_comparison_agent.py

def ml_dl_comparison_agent(
    best_ml_model,
    best_ml_rmse: float,
    best_dl_model,
    best_dl_rmse: float,
    improvement_threshold: float = 0.05
):
    """
    Selects between ML and DL models using a rational decision rule.

    Rules:
    - Prefer ML for stability and interpretability
    - Switch to DL only if it improves RMSE by a meaningful margin
    """

    print("\n⚖️ ML vs DL Decision Agent")

    improvement = (best_ml_rmse - best_dl_rmse) / best_ml_rmse

    if improvement > improvement_threshold:
        print(
            f"🧠 DL selected | RMSE improvement: {improvement:.2%}"
        )
        decision = "DL"
        selected_model = best_dl_model
    else:
        print(
            f"📊 ML selected | DL improvement insufficient ({improvement:.2%})"
        )
        decision = "ML"
        selected_model = best_ml_model

    return {
        "selected_model": selected_model,
        "selected_type": decision,
        "ml_rmse": best_ml_rmse,
        "dl_rmse": best_dl_rmse,
        "improvement_ratio": improvement
    }
