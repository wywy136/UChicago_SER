import os


def getfilepaths(zone: int, date: str) -> list:
    filepath = "/project/graziul/data/Zone" + str(zone) + "/" + date
    infile_path = []
    infiles = os.listdir(filepath)
    for file in infiles:
        if file.endswith(".mp3") and file.startswith("2018"):
            infile_path.append(os.path.join(filepath, file))
    
    return infile_path
