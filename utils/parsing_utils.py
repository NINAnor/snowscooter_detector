import glob
import os
from pathlib import Path


def remove_extension(input):
    """Removes filename extension AND directory path"""
    filename = Path(input)
    filename = filename.with_suffix("")
    filename = os.path.basename(filename)
    return filename


def parseFolders(apath, rpath):

    audio_files = [
        f for f in glob.glob(apath + "/**/*", recursive=True) if os.path.isfile(f)
    ]
    audio_no_extension = []
    for audio_file in audio_files:
        audio_file_no_extension = remove_extension(audio_file)
        audio_no_extension.append(audio_file_no_extension)

    result_files = [
        f for f in glob.glob(rpath + "/**/*", recursive=True) if os.path.isfile(f)
    ]

    flist = []
    for result in result_files:
        result_no_extension = remove_extension(result)
        result_no_extension.split(".Table")
        is_in = result_no_extension.split(".Table")[0] in audio_no_extension

        if is_in:
            audio_idx = audio_no_extension.index(result_no_extension.split(".Table")[0])
            pair = {"audio": audio_files[audio_idx], "result": result}
            flist.append(pair)
        else:
            continue

    print("Found {} audio files with valid result file.".format(len(flist)))

    return flist
