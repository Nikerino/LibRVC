import os

import wget

from librvc.modules.vc.modules import VC


def get_models(model_dir: str):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    hubert_model_path = os.path.join(model_dir, 'hubert_base.pt')
    rmvpe_model_path = os.path.join(model_dir, 'rmvpe.pt')

    if not os.path.exists(hubert_model_path):
        wget.download('https://huggingface.co/IAHispano/Applio/resolve/main/Resources/hubert_base.pt?download=true', out=hubert_model_path)

    if not os.path.exists(rmvpe_model_path):
        wget.download('https://huggingface.co/IAHispano/Applio/resolve/main/Resources/rmvpe.pt?download=true', out=rmvpe_model_path)

    return hubert_model_path, rmvpe_model_path

class RVC():

    def __init__(self, model_dir: str = '.checkpoints'):
        hubert_model_path, rvmpe_model_path = get_models(model_dir)
        self.vc = VC(hubert_model_path, rvmpe_model_path)
    
    def convert(self, speaker_model_path: str, speaker_index_path: str, input_file: str, pitch_shift: int = 0):
        self.vc.get_vc(speaker_model_path, speaker_index_path)
        fs, data, times, _ = self.vc.vc_inference(1, input_file, f0_up_key=pitch_shift)
        return data, fs