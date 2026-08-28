import yaml
from cascades.ext.spkdia.model.diarization.feat_utils import Wav2Feat
from cascades.ext.spkdia.model.diarization.dvec_utils import load_dvec_model, DvecCfg
from cascades.ext.spkdia.inference.tsvad_inference import extract_dvec_inference


MODEL_DIR = "/mnt/tsstd01wus2/data/princetondata/princetondata/Models/SpeakerDia/"
dvec_cfg_dict = {
    "MODEL_ROOT": MODEL_DIR + "fast_diarization_v1_241118_compliance/release_package",
    "model_cfg_file": "spkid/config.yaml",
    "model_file": "spkid/model.checkpoint",
    "fbank_mean_invstd_file": "spkid/mean_invstd.vec",
}
dvec_cfg = DvecCfg(**dvec_cfg_dict)


class DvecExtractor:
    """Extract d-vector embeddings from audio."""
    def __init__(self, device="cuda"):
        self.device = device
        self._wav2feat = Wav2Feat(
            mean_invstd_file=dvec_cfg.fbank_mean_invstd_file,
            input_sampling_rate=16000,
            wav_downsample_prob=0,
        )
        self._load_dvec_model(dvec_cfg.model_cfg_file, dvec_cfg.model_file)

    def _load_dvec_model(self, dvec_model_cfg_file, dvec_model_file):
        with open(dvec_model_cfg_file, encoding="utf-8") as file:
            dvec_training_config = yaml.safe_load(file)

        # set the model file path
        dvec_model_config = dvec_training_config["model"]
        dvec_model_config["init_model"] = {"checkpoint": dvec_model_file}

        # load model
        dvec_model = load_dvec_model(dvec_model_config)
        dvec_model.to(self.device)
        dvec_model.eval()

        self.dvec_model = dvec_model
        self.dvec_cfg = dvec_cfg

    def get_dvec_shift(self):
        """Get d-vector shift in seconds."""
        return self.dvec_cfg.dvec_shift

    def __call__(self, wav):
        feature = self._wav2feat(wav).to(self.device)
        dvec = extract_dvec_inference(feature, self.dvec_model, dvec_hop_size=4, dvec_model_version=4)

        return feature, dvec
