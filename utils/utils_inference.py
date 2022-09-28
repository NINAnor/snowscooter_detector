from utils.audio_processing import openAudioFile, openCachedFile, splitSignal, noise

# Difference with testing is that it uses pyfilesystem to fetch the files
# to predict on.

class AudioList():

    def __init__(self, length_segments = 3, overlap = 0, sample_rate=44100):
        self.sample_rate = sample_rate
        self.length_segments = length_segments
        self.overlap = overlap
        
    def read_audio(self, filesystem, audio_path):
        """Read the audio, change the sample rate and randomly pick one channel"""
        if filesystem is False:
            sig, _ = openAudioFile(audio_path, sample_rate=self.sample_rate)
        else:
            sig, _ = openCachedFile(filesystem, audio_path, sample_rate=self.sample_rate)
        return sig

    def split_segment(self, array):
        splitted_array = splitSignal(array, rate=self.sample_rate, seconds=self.length_segments, overlap=self.overlap, minlen=3)
        return splitted_array

    def get_processed_list(self, filesystem, audio_path):
        track = self.read_audio(filesystem, audio_path)      
        list_divided = self.split_segment(track)
        return list_divided