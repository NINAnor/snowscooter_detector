import glob
import numpy as np
import librosa

import warnings
warnings.filterwarnings("ignore")

def openAudioFile(path, sample_rate=44100, offset=0.0, duration=None):    
    try:
        sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')
    except:
        sig, rate = [], sample_rate
    return sig, rate

if __name__ == "__main__":

    FOLDERS = glob.glob("/data/Prosjekter3/16784000_sats_22_51_rosten/Model_training_data/*")

    for folder in FOLDERS:

        length_files = []
        FILES = glob.glob(folder + "/*.wav")

        for file in FILES:

            sig, rate = openAudioFile(file)
            duration = len(sig) / rate
            length_files.append(duration)

        total_length = sum(np.array(length_files))
        print("Total time of audio for folder {} is {}".format(folder.split("/")[-1], total_length))

# docker run --rm -it -v $pwd:/app -v ~/Data/:/Data audioclip:latest poetry run python length_audio.py
# docker run --rm -it -v $pwd:/app -v /data/:/data audioclip:latest poetry run python length_audio.py

# Total time of audio for folder Snowscooter_passages is 14628.809841269802 // 4 hours --> 4800
# Total time of audio for folder Norwegian_soundscape is 120667.35780045352 // 33 hours --> 39600
