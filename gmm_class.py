import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import BayesianGaussianMixture
from typing import Dict, List, Tuple


from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
)

from emitter_data.my_settings import (
    DUTY_CYCLE,
    FREQUENCY_RANGE,
    PRI_RANGE,
    TIME_BANDWIDTH_PRODUCT,
)

SEED = 42

# Fasta skalningsintervall per feature, harledda ur samma my_settings-
# granser som transformer-tokenizerns globala grid (med samma padding).
# Skalningen ar darmed en ren funktion av EN signal: inget fittas pa
# datasetet, sa ingen information lacker fran "framtida" signaler in i
# den sekventiella metoden, och jamforelsen mot transformern sker pa
# samma villkor.
_PW_LOW = PRI_RANGE[0] * DUTY_CYCLE[0]
_PW_HIGH = PRI_RANGE[1] * DUTY_CYCLE[1]
FEATURE_RANGES = {
    "pri": (PRI_RANGE[0] / 2.0, PRI_RANGE[1] * 4.0),
    "freq": (FREQUENCY_RANGE[0] * 0.85, FREQUENCY_RANGE[1] * 1.15),
    "pw": (_PW_LOW / 2.0, _PW_HIGH * 2.0),
    "bw": (
        TIME_BANDWIDTH_PRODUCT[0] / _PW_HIGH / 2.0,
        TIME_BANDWIDTH_PRODUCT[1] / _PW_LOW * 2.0,
    ),
}


class Entry:
    def __init__(
        self,
        model: BayesianGaussianMixture,
        max_length: int,
        n_samples: int = 10000,
        percentile: float = 1.0,
    ):
        """
        Initierar en Entry och förberäknar omedelbart
        tröskeltabellen för alla längder upp till max_length.
        """
        self.model = model
        self.N = n_samples
        self.percentile = percentile
        self.max_length = max_length

        # self.threshold_table[L-1] kommer att vara tröskeln för längd L
        self.threshold_table = self._create_threshold_table()

    def _create_threshold_table(self) -> np.ndarray:
        """
        Skapar en uppslagstabell för log-likelihood-trösklar för
        alla längder från 1 till self.max_length, enligt
        handledarens metod (cumsum).
        """
        # print(f"Skapar tröskeltabell (max_length={self.max_length})...") # För felsökning
        np.random.seed(SEED)  # För reproducerbara tabeller

        # 1. Sampla N * max_length punkter
        all_samples, _ = self.model.sample(n_samples=self.N * self.max_length)

        # 2. Beräkna log-likelihood för *varje* individuell punkt
        log_likelihoods_per_point = self.model.score_samples(all_samples)

        if False:
            # --- DEBUG START: Kontrollera variation ---
            min_val = log_likelihoods_per_point.min()
            max_val = log_likelihoods_per_point.max()
            std_val = log_likelihoods_per_point.std()

            print(f"DEBUG CHECK:")
            print(f"  Min Log Score: {min_val:.4f}")
            print(f"  Max Log Score: {max_val:.4f}")
            print(f"  Standardavvikelse: {std_val:.4f}")

        # 3. Omforma till (N, max_length)
        ll_matrix = log_likelihoods_per_point.reshape(
            self.N, self.max_length
        )  # innehåller log_liklelihood scoren för varje pdw för varje samplad signal, alla samplade signaler har längden max_length

        # 4. Beräkna den kumulativa summan (cumsum) längs varje rad
        cumulative_sums = np.cumsum(
            ll_matrix, axis=1
        )  # Här beräknas den kumulativa summan ut för varje rad.
        # 5. Skapa en array med längder: [1, 2, 3, ..., max_length]
        lengths = np.arange(1, self.max_length + 1)

        # 6. Beräkna medelvärdet för varje längd L (Score)
        #    Vi delar varje kolumn L-1 med L
        mean_ll_matrix = cumulative_sums / lengths

        # 7. Skapa den slutgiltiga tröskeltabellen
        #    Beräkna percentilen för varje kolumn (axis=0)
        threshold_table = np.percentile(mean_ll_matrix, self.percentile, axis=0)

        # print("...Tröskeltabell skapad.") # För felsökning
        return threshold_table

    def check_signal(self, signal):
        length = signal.shape[0]
        if length == 0:
            return np.inf  # Kan inte hantera tom signal

        log_likelihood = self.model.score(signal)

        # Hämta tröskeln från vår förberäknade tabell
        if length > self.max_length:
            # Fall 1: Använd tröskeln för max_length
            # (Antagandet är att scoren stabiliseras)
            threshold = self.threshold_table[-1]
            # Fall 2: Vägra (om du föredrar det)
            # return np.inf
        else:
            # Kom ihåg 0-indexering: längd L finns på index L-1
            threshold = self.threshold_table[length - 1]

        difference = threshold - log_likelihood
        return difference

    # Ta bort _set_threshold och _set_threshold2, de behövs inte längre


class Library:
    def __init__(
        self,
        n_samples,
        percentile=1.0,
        model=None,
        threshold=0.9,
        columns=["pri", "freq", "pw", "bw"],
        n_bins=16,
        device="cpu",
    ):
        """
        Initierar biblioteket. Kräver att få veta den
        maximala signallängden som kommer att bearbetas.
        data: List av tuples (DataFrame, label) för evaluering
        """
        self.models: List[Entry] = []
        self.predictions: List[int] = []
        self.max_signal_length = -1

        self.percentile = percentile
        self.N = n_samples
        self.columns = columns

    def process(self, signal):
        # Best-match: utvardera ALLA modeller och ta den med bast marginal
        # bland dem som passerar troskeln. First-match gjorde tilldelningen
        # beroende av i vilken ordning biblioteket rakade byggas, vilket
        # blir godtyckligt nar tva snarlika entries bada passerar.
        if self.models:
            differences = [model.check_signal(signal) for model in self.models]
            best = int(np.argmin(differences))
            if differences[best] < 0:
                self.predictions.append(best)
                return best

        # Om ingen modell matchar, skapa en ny
        np.random.seed(SEED)
        model = BayesianGaussianMixture(
            n_components=20,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.1,
            max_iter=1000,
        )
        model.fit(signal)

        # *** HÄR ÄR ÄNDRINGEN ***
        # Skicka med max_signal_length när en ny Entry skapas
        entry = Entry(
            model,
            max_length=self.max_signal_length,
            percentile=self.percentile,
            n_samples=self.N,
        )

        self.models.append(entry)
        pred = len(self.models) - 1
        self.predictions.append(pred)
        return pred

    def _scale_data(self, data):
        """Deterministisk skalning till [-1, 1] per feature utifran de
        fasta intervallen i FEATURE_RANGES. Anvander self.columns, sa
        eventuella extra kolumner i signalen (t.ex. toa) ignoreras."""
        scaled_data = []
        for signal_df in data:
            scaled = np.empty((len(signal_df), len(self.columns)))
            for j, col in enumerate(self.columns):
                low, high = FEATURE_RANGES[col]
                center = (low + high) / 2.0
                half_span = (high - low) / 2.0
                values = signal_df[col].to_numpy(dtype=float)
                scaled[:, j] = (values - center) / half_span
            scaled_data.append(scaled)
        return scaled_data

    def fit_predict(self, signals_list):
        self.max_signal_length = max(len(signal) for signal in signals_list)
        scaled_data = self._scale_data(signals_list)
        preds = []
        for signal in tqdm(scaled_data, desc="Processing Signals"):
            pred = self.process(signal)
            preds.append(pred)
        return preds

    def evaluate(self, true_labels, predicted_labels, verbose=False):
        """
        Beräknar Homogeneity och Completeness.
        """
        hom = homogeneity_score(true_labels, predicted_labels)
        com = completeness_score(true_labels, predicted_labels)
        v_meas = v_measure_score(true_labels, predicted_labels)
        if verbose:
            print("\n--- Utvärdering ---")
            print(f"Antal hittade kluster: {len(set(predicted_labels))}")
            print(f"Antal sanna kluster:   {len(set(true_labels))}")
            print(
                f"Homogeneity:  {hom:.4f} (Varje kluster innehåller bara medlemmar av en klass)"
            )
            print(
                f"Completeness: {com:.4f} (Alla medlemmar av en klass är tilldelade samma kluster)"
            )
            print(f"V-Measure:    {v_meas:.4f} (Harmoniskt medelvärde)")

        return {
            "homogeneity": hom,
            "completeness": com,
            "v_measure": v_meas,
            "num_predicted_clusters": len(set(predicted_labels)),
            "num_true_clusters": len(set(true_labels)),
            "threshold": self.percentile,
        }


from tqdm import tqdm


class MyDataCollection(BaseModel):
    data_points: List[Dict[str, int]]
