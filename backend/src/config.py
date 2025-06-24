from datetime import timedelta

MAX_DF = 0.7
MIN_DF = 1
EPS = 1.17
MIN_SAMPLES = 1

CLUSTER_TTL = timedelta(seconds=10)

PARSING_INTERVAL = 3600

SUMMARY_SIZE = 3

LOG_LEVEL = "DEBUG"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default_console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "app": {
            "handlers": ["default_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["default_console"],
        "level": "WARNING",
    },
}

SUMM_MODELS_FILEPATHS = {
    "dt": "ml_models/best_dt.joblib",
    "dt_smote": "ml_models/best_smote_dt.joblib",
    "rf": "ml_models/best_rf.joblib",
    "xgb": "ml_models/best_xgb.json",
    "lgbm": "ml_models/best_lgbm.joblib",
}
