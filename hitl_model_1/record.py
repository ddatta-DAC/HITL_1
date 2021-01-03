# ====================================
# Object to store input records
# ====================================
from sklearn.preprocessing import normalize
import glob
import numpy as np
import os
import sklearn

class record_class:
    embedding = None
    serialID_to_entityID = None

    @staticmethod
    def __setup_embedding__(embedding_path, serialID_to_entityID, _normalize=True):
        record_class.embedding = {}
        record_class.serialID_to_entityID = serialID_to_entityID
        files = glob.glob(os.path.join(embedding_path, '**.npy'))
        for f in sorted(files):
            emb = np.load(f)
            domain = f.split('/')[-1].split('_')[-1].split('.')[0]
            if _normalize:
                emb = normalize(emb, axis=1)
            record_class.embedding[domain] = emb
        return

    def __init__(self, _record, _label):
        id_col = 'PanjivaRecordID'
        self.id = _record[id_col]
        domains = list(record_class.embedding.keys())
        self.x = []
        self.label = _label
        for d, e in _record.items():
            if d == id_col: continue
            non_serial_id = record_class.serialID_to_entityID[e]
            self.x.append(record_class.embedding[d][non_serial_id])
        self.x = np.array(self.x)
