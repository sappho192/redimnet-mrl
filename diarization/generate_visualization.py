import optuna

study = optuna.load_study(
    study_name="hierarchical_mrl_diarization_2025-12-15",
    storage="sqlite:///optuna_study.db"
)

from optuna.visualization import plot_pareto_front
fig = plot_pareto_front(study, target_names=["DER", "Time", "Speaker MAE"])
fig.write_html("pareto_front.html")
