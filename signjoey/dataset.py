# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import os
import numpy as np

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(

        self,
        path: str,
        path_posestimation : str,
        fields: Tuple[RawField, RawField, Field, Field, Field, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
                ("keypoints_face", fields[5]),
                ("keypoints_body", fields[6]),
                ("keypoints_hand", fields[7])
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []

        for s in samples:
            sample = samples[s]
            n_timesteps = sample["sign"].size()[0]
            sample_name = sample["name"].split('/')[1]
            keypoints_face = np.load(os.path.join(path_posestimation, sample_name, 'face.npy'))
            keypoints_body = np.load(os.path.join(path_posestimation, sample_name, 'body.npy'))
            keypoints_hand = np.load(os.path.join(path_posestimation, sample_name, 'hand.npy'))
            print(type(keypoints_face), keypoints_face.shape, print(type(sample["sign"])))
            #print(type(keypoints_face), type(keypoints_face[0]), keypoints_face[0])
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                        #torch.from_numpy(keypoints_face),
                        #torch.from_numpy(keypoints_body),
                        #torch.from_numpy(keypoints_hand)

                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
