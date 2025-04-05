from .plottable_data import (
    field_histograms,
    make_video,
    plot_all_time_metrics,
    plot_power_spectrum_by_field,
)
from .spatial import MAE, MSE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity, PearsonR
from .spectral import binned_spectral_mse

__all__ = [
    "MAE",
    "NRMSE",
    "RMSE",
    "MSE",
    "NMSE",
    "LInfinity",
    "VMSE",
    "VRMSE",
    "binned_spectral_mse",
    "PearsonR",
]

long_time_metrics = ["VRMSE", "RMSE", "binned_spectral_mse", "PearsonR"]
validation_metric_suite = [
    RMSE(),
    NRMSE(),
    LInfinity(),
    VRMSE(),
    binned_spectral_mse(),
    PearsonR(),
]
validation_plots = [plot_power_spectrum_by_field, field_histograms]
time_plots = [plot_all_time_metrics]
time_space_plots = [make_video]
