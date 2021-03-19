# ====================================
# Object to store input records
# ====================================
from sklearn.preprocessing import normalize
import glob
import numpy as np
import os
import sklearn
from collections import OrderedDict


class record_class:
    embedding = None
    serialID_to_entityID = None

    @staticmethod
    def __setup_embedding__(
            embedding_path,
            serialID_to_entityID,
            _normalize=True
    ):
        record_class.embedding = {}
        record_class.serialID_to_entityID = serialID_to_entityID
        files = glob.glob(os.path.join(embedding_path, '**.npy'))
        
        for f in sorted(files):
            emb = np.load(f)
            domain = f.split('/')[-1].split('_')[-1].split('.')[0]
            if _normalize:
                emb = normalize(emb, axis=1)
            record_class.embedding[domain] = emb
        record_class.domains = list(record_class.embedding.keys())
        return

    def __init__(self, _record, _label, interaction_type='concat'):
        _record = OrderedDict(_record)
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
        self.interaction_type = interaction_type

    def calc_features(self):
        num_domains = len(record_class.domains)
        terms = []
        interaction_type = self.interaction_type
        if interaction_type is not None:
            for i in range(num_domains):
                for j in range(i + 1, num_domains):
                    if interaction_type == 'concat':
                        x1x2 = np.concatenate([self.x[i], self.x[j]])
                    elif interaction_type == 'mul':
                        x1x2 = self.x[i] * self.x[j]
                    terms.append(x1x2)
            self.features = np.array(terms)
        else:
            self.features = np.concatenate(self.x)
      