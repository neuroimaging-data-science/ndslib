
import numpy as np
import json
import requests
import shutil
import tempfile
import os
import os.path as op
import pandas as pd
import nibabel as nib


def _read_afq():
    subjects = pd.read_csv(
        "https://yeatmanlab.github.io/AFQBrowser-demo/data/subjects.csv",
        usecols=[1, 2, 3, 4, 5, 6, 7],
        na_values="NaN", index_col=0)

    subjects["age_less_than_10"] = subjects["Age"] < 10

    nodes = pd.read_csv(
        'https://yeatmanlab.github.io/AFQBrowser-demo/data/nodes.csv',
        index_col=0)
    joined = pd.merge(subjects, nodes, how="inner",
                      left_index=True, right_index=True)

    return joined, joined["tractID"].unique()


def load_data(dataset, fname=None):
    """
    Load data for use in examples

    Parameters
    ----------
    dataset : str
        The name of a dataset. Can be one of:
        "bold_numpy" : Read a BOLD time-series as a numpy array.
        "bold_volume" : Read a single volume of a BOLD time-series
                        a numpy array
        "afq" : Read AFQ data
        "age_groups_fa" : Read AFQ data and return dataframe divided
                          by age-groups
        "abide2_saggitals": Read ABIDE2 mid-saggitals as numpy arrays.

    fname : str, optional.
        If provided, data will be cached to this local path and retrieved
        from there on future calls with the same value.
    """

    if dataset.lower() == "bold_numpy":
        return load_npy("https://ndownloader.figshare.com/files/20182988",
                        fname)
    if dataset.lower() == "bold_volume":
        return load_npy("https://ndownloader.figshare.com/files/20182988",
                        fname)[..., 0]
    if dataset.lower() == "afq":
        return _read_afq()
    if dataset.lower() == "age_groups_fa":
        afq_data, tracts = load_data("afq")
        age_groups = afq_data.groupby(["age_less_than_10", "tractID", "nodeID"])
        younger_fa = age_groups.mean()["fa"][True]
        older_fa = age_groups.mean()["fa"][False]
        return younger_fa, older_fa, tracts
    if dataset.lower() == "abide2":
        abide = pd.read_csv(
            'https://figshare.com/ndownloader/files/31374262', sep='\t')
        abide.insert(6, 'temp', abide['age_resid'])
        abide.drop('age_resid', axis=1, inplace=True)
        abide.rename(columns={'temp': 'age_resid'}, inplace=True)
        return abide
    if dataset.lower() == "hcp-mmp1":
        download_file(
            "https://github.com/neurohackademy/nh2021-curriculum/blob/main/machine-learning/data/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz?raw=true",
            "HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz")
        img = nib.load("HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz")
        return img
    if dataset.lower() == "abide2_saggitals":
        download_file("https://figshare.com/ndownloader/files/34050380",
                      "abide2_saggitals_X.npy")
        X = np.load("abide2_saggitals_X.npy")
        download_file("https://figshare.com/ndownloader/files/34050383",
                      "abide2_saggitals_group.npy")
        group = np.load("abide2_saggitals_group.npy")
        download_file(
            "https://figshare.com/ndownloader/files/34050377",
            "abide2_saggitals_subject.npy")
        subject = np.load("abide2_saggitals_subject.npy")
        return X, group, subject

    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def download_file(url, fname):
    """
    Download a file from a URL and save it locally

    Parameters
    ----------
    url : str
        The URL where the file is stored.
    fname : str
        The local filename to save to.
    """
    path, fname = op.split(fname)
    os.makedirs(path, exist_ok=True)
    response = requests.get(url, stream=True)
    with open(op.join(path, fname), 'wb') as f:
        shutil.copyfileobj(response.raw, f)


def make_description(path, fname='dataset_description.json',
                     BIDSVersion="1.4.0", **kwargs):
    """Creates a BIDS-compatible dataset description file.

    Parameters
    ----------

    path : str
        Location of the BIDS dataset.

    fname : str, optional
        Name of the description file (default: `'dataset_description.json'`).

    BIDSVersion : str, optional
        Choose the version of BIDS to which this dataset complies.

    **kwargs : dict, optional
        Other fields to put in the file.
    """
    kwargs["Name"] = kwargs.get("Name", path)
    kwargs.update({"BIDSVersion": BIDSVersion})
    desc_file = op.join(path, fname)
    with open(desc_file, 'w') as outfile:
        json.dump(kwargs, outfile)


def download_bids_dataset():
    """
    Makes a minimal BIDS dataset with one fMRI subject from OpenNeuro ds001233
    """
    download_file("https://openneuro.org/crn/datasets/ds001233/snapshots/00003/files/sub-17:ses-pre:func:sub-17_ses-pre_task-cuedSFM_run-01_bold.nii.gz",
                "ds001233/sub-17/ses-pre/func/sub-17_ses-pre_task-cuedSFM_run-01_bold.nii.gz")

    download_file("https://openneuro.org/crn/datasets/ds001233/snapshots/00003/files/task-cuedSFM_bold.json",
                "ds001233/sub-17/ses-pre/func/sub-17_ses-pre_task-cuedSFM_run-01_bold.json")

    download_file("https://openneuro.org/crn/datasets/ds001233/snapshots/00003/files/sub-17:ses-pre:anat:sub-17_ses-pre_T1w.nii.gz",
                "ds001233/sub-17/ses-pre/anat/sub-17_ses-pre_T1w.nii.gz")

    download_file("https://openneuro.org/crn/datasets/ds001233/snapshots/00003/files/sub-17:ses-pre:anat:sub-17_ses-pre_T1w.json",
                "ds001233/sub-17/ses-pre/anat/sub-17_ses-pre_T1w.json")

    make_description("ds001233",
                    BIDSVersion="1.0.2",
                     **{"Name":"singleFingerRSA",
                        "Authors":["Patrick Beukema","Timothy Verstynen"],
                        "Funding":["NSF Career Award 1351748, Pennsylvania Department of Health Formula Award SAP4100062201, Multimodal Neuroimaging Training Program NIH T90 DA022761"],
                        "License":"Creative Commons CC0 1.0",
                        "HowToAcknowledge":"Please cite the following manuscript Patrick Beukema, Jörn Diedrichsen, Timothy Verstynen. bioRxiv 255794; doi: https://doi.org/10.1101/255794 in addition to this dataset"})


def load_npy(url, fname=None):
    """
    Loads a numpy file from a URL

    Parameters
    ---------
    url : string
        The URL at which the file is stored.

    fname : string, optional
        A full path to a local file, in which the file will get stored.
        Default: store the data in a temporary and ephemeral location.
    """
    if fname is None:
        fname = tempfile.NamedTemporaryFile().name

    if op.exists(fname):
        return np.load(fname)

    download_file(url, fname)

    return np.load(fname)
