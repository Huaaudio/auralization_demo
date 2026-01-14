import numpy as np
class SQmetrics:
    def __init__(self):
        from mosqito.sq_metrics import loudness_zwst, loudness_ecma, tnr_ecma_st
        from mosqito.sq_metrics import sharpness_din_st
        from mosqito.sq_metrics import sii_ansi, roughness_ecma, pr_ecma_st
        # removed: import src.secondment.fluctuation_strength as fs
        self.loudness_zwst = loudness_zwst
        self.loudness_ecma = loudness_ecma
        self.sharpness_din_st = sharpness_din_st
        self.sii_ansi = sii_ansi
        self.roughness_ecma = roughness_ecma
        self.pr_ecma_st = pr_ecma_st
        self.tnr_ecma_st = tnr_ecma_st
        #self.shm_loudness_ecma = shm_loudness_ecma
        #self.shm_tonality_ecma = shm_tonality_ecma

    def compute_metrics(self, convolved_data, sample_rate):
        """
        Compute sound quality metrics for given convolved audio data.
        Parameters:
        convolved_data (array-like): Audio data to compute metrics on.
        sample_rate (int): Sample rate of the audio data.
        Returns:
        dict: Dictionary with computed metrics for each audio file.
        """
        metrics = {}
        data = convolved_data
        # Compute Loudness (Zwicker method)
        metrics['loudness_zwst'] = self.loudness_zwst(data, sample_rate)[0]
        # Compute Loudness (ECMA method)
        metrics['loudness_ecma'] = self.loudness_ecma(data, sample_rate)[0]
        # Compute Sharpness (DIN St method)
        metrics['sharpness_din_st'] = self.sharpness_din_st(data, sample_rate)
        # Compute SII (ANSI method)
        metrics['sii_ansi'] = self.sii_ansi(data, sample_rate, method='third_octave', speech_level='normal')[0]
        # Compute Loudness (Sottek Hearing Model - ECMA method)
        #metrics['shm_loudness_ecma'] = self.shm_loudness_ecma(data, sample_rate)
        # Compute Tonality (Sottek Hearing Model - ECMA method)
        #metrics['shm_tonality_ecma'] = self.shm_tonality_ecma(data, sample_rate)
        # Compute Roughness (ECMA method)
        metrics['roughness_ecma'] = self.roughness_ecma(data, sample_rate)[0]
        # Compute Prominence Ratio (ECMA method)
        metrics['pr_ecma_st'] = self.pr_ecma_st(data, sample_rate)[0]
        # Compute Tone to Noise Ratio (ECMA method)
        metrics['tnr_ecma_st'] = self.tnr_ecma_st(data, sample_rate)[0]
        # Compute Fluctuation Strength
        # specific_loudness = self.loudness_zwst(data, sample_rate)[1]
        # from . import fluctuation_strength as fs
        # metrics['fluctuation_strength'] = fs.acousticFluctuation(specific_loudness, fs.fmoddetection(specific_loudness))
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print computed sound quality metrics.
        Parameters:
        metrics (dict): Dictionary with computed metrics
        """
        def _fmt(v):
            # numpy arrays
            if isinstance(v, np.ndarray):
                return '[' + ', '.join(f"{float(x):.4f}" for x in v) + ']'
            # lists / tuples
            if isinstance(v, (list, tuple)):
                parts = []
                for x in v:
                    if isinstance(x, np.ndarray):
                        parts.append('[' + ', '.join(f"{float(y):.4f}" for y in x) + ']')
                    else:
                        try:
                            parts.append(f"{float(x):.4f}")
                        except Exception:
                            parts.append(str(x))
                return '(' + ', '.join(parts) + ')'
            # scalars
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {_fmt(value)}")