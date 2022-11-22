import numpy as np
from sklearn.ensemble import RandomForestClassifier

class HierarchicalForest:
    def __init__(self, n_estimators_coarse=25, n_estimators_fine=10, max_depth=10, fine_to_coarse=None, mod=None, transformations=None, honest=False):
        self.n_estimators_coarse=n_estimators_coarse
        self.coarse_forest=None
        
        self.n_estimators_fine=n_estimators_fine
        self.fine_forests={}
        
        self.max_depth=max_depth

        self.fine_to_coarse = fine_to_coarse

        self.mod = mod

        self.transformations = transformations
        
        self.fitted=False

        if honest:
            self.forest_class = UncertaintyForest
        else:
            self.forest_class = RandomForestClassifier
        
        
    def fit(self, X, y):

        if self.fine_to_coarse is None:
            raise ValueError("fine_to_coarse is None. please give a valid mapping in the constructor.")

        self.classes = np.unique(y)

        if self.mod == None:
            self.mod = len(self.classes)
            
        self.mod_classes = np.unique(y % self.mod)
        self.coarse_labels = np.unique(self.fine_to_coarse)

        if self.transformations is not None:
            if isinstance(self.transformations, list):
                if len(self.transformations) == len(self.coarse_labels):
                    self.use_transforms = True
                else:
                    raise ValueError('number of transformations (%i) must equal the number of coarse classes (%i)'%(len(self.transformations), len(self.coarse_labels)))
            else:
                raise ValueError('not implemented error. to use transformations please provide a list of fitted transformations of length %i'%(len(self.coarse_labels)))
        else:
            self.use_transforms = False

        y_coarse=np.zeros(len(y))
        for coarse_label in self.coarse_labels:
            temp_fine_indices = np.where(self.fine_to_coarse == coarse_label)[0]
            temp_indices = np.concatenate([np.where(y == self.classes[tfi])[0] for tfi in temp_fine_indices])
            
            y_coarse[temp_indices] = coarse_label
            
        self._fit_coarse(X, y_coarse)
        self._fit_fine(X, y, y_coarse)
        
        self.fitted=True

        return self
        
        
    def _fit_coarse(self, X, y_coarse):
        self.coarse_forest = self.forest_class(n_estimators=self.n_estimators_coarse, max_depth=self.max_depth).fit(X, y_coarse)
    
    
    def _fit_fine(self, X, y, y_coarse):
        for i, coarse_label in enumerate(self.coarse_labels):
            temp_indices = np.where(y_coarse == coarse_label)[0]

            if self.use_transforms:
                self.fine_forests[coarse_label] = self.forest_class(n_estimators=self.n_estimators_fine, max_depth=self.max_depth
                                                ).fit(self.transformations[i].transform(X[temp_indices]), y[temp_indices] % self.mod)

            else:
                self.fine_forests[coarse_label] = self.forest_class(n_estimators=self.n_estimators_fine, max_depth=self.max_depth
                                                ).fit(X[temp_indices], y[temp_indices] % self.mod)
            

    def predict_proba(self, X):
        posteriors = np.zeros((X.shape[0], len(self.mod_classes)))
        coarse_posteriors = self.coarse_forest.predict_proba(X)

    
        #- Hierarchical posteriors & prediction
        for i, coarse_label in enumerate(self.coarse_labels):
            temp_fine_label_indices = np.where(self.fine_to_coarse == coarse_label)[0]

            if self.use_transforms:
                temp_fine_posteriors = self.fine_forests[coarse_label].predict_proba(self.transformations[i].transform(X))
            else:
                temp_fine_posteriors = self.fine_forests[coarse_label].predict_proba(X)

            if self.mod == len(self.classes):
                posteriors[:, temp_fine_label_indices] = np.multiply(coarse_posteriors[:, i],
                                                                             temp_fine_posteriors.T
                                                                            ).T
            else:
                posteriors += np.multiply(coarse_posteriors[:, i],
                                                temp_fine_posteriors.T).T
        
        return posteriors
    

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)