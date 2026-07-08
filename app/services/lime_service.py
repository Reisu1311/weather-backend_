import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from .preprocessor import FEATURES, HUMAN_LABELS, CLASS_NAMES

class LimeService:
    def __init__(self, model_service):
        self.model_service = model_service
        self.explainer     = None
        self._init_explainer()

    def _init_explainer(self):
        n     = len(FEATURES)
        dummy = np.random.rand(300, n)
        self.explainer = LimeTabularExplainer(
            training_data        = dummy,
            feature_names        = FEATURES,
            class_names          = list(self.model_service.le.classes_),
            mode                 = "classification",
            discretize_continuous= True,
        )
        print("✅ LIME explainer (classification) initialized")

    def explain(
        self,
        features          : np.ndarray,
        feature_values_raw: dict,
        predicted_label    : str,
        top_n              : int = 8,
    ) -> list:
        try:
            X_scaled = self.model_service.scaler.transform(features)

            # Index kelas yang diprediksi
            label_idx = list(
                self.model_service.le.classes_).index(predicted_label)

            exp = self.explainer.explain_instance(
                data_row    = X_scaled[0],
                predict_fn  = self.model_service.model.predict_proba,
                num_features= top_n,
                labels      = (label_idx,),
                num_samples = 500,
            )

            result = []
            for feat_desc, weight in exp.as_list(label=label_idx):
                feat_name = self._extract_feat_name(feat_desc)
                raw_val   = feature_values_raw.get(feat_name, 0)
                result.append({
                    "feature"    : feat_name,
                    "weight"     : round(float(weight), 4),
                    "human_label": HUMAN_LABELS.get(feat_name, feat_name),
                    "value"      : str(round(float(raw_val), 2)),
                })
            return result
        except Exception as e:
            print(f"LIME error: {e}")
            return []

    def _extract_feat_name(self, desc: str) -> str:
        # Urutkan dari nama terpanjang agar tidak salah match
        # misalnya "temperature_2m" vs "temperature_2m_lag1"
        sorted_feats = sorted(FEATURES, key=len, reverse=True)
        for f in sorted_feats:
            if f in desc:
                return f
        return desc.split(" ")[0]