import pathlib
import tempfile
import zipfile


def retrieve_train_data_path(zip_file):
    zpath = pathlib.Path(__file__).resolve().parent / "data" / zip_file
    # unpack
    arc = zipfile.ZipFile(str(zpath))
    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=zpath.name)
    arc.extractall(edest)
    path = pathlib.Path(edest)
    return path
