import numpy as np
from proglearn import UncertaintyForest as UF

class HierarchicalForest:
    def __init__(self, n_estimators_coarse=25, n_estimators_fine=10, max_depth=10):
        self.n_estimators_coarse=n_estimators_coarse
        self.coarse_forest=None
        
        self.n_estimators_fine=n_estimators_fine
        self.fine_forests={}
        
        self.max_depth=max_depth
        
        self.fitted=False
        
    def fit(self, X, y, fine_to_coarse):
        self.fine_to_coarse = fine_to_coarse
        self.classes = np.unique(y)
        self.coarse_labels = np.unique(fine_to_coarse)
        
        
        y_coarse=np.zeros(len(y))
        for coarse_label in self.coarse_labels:
            temp_fine_indices = np.where(fine_to_coarse == coarse_label)[0]
            temp_indices = np.concatenate([np.where(y == self.classes[tfi])[0] for tfi in temp_fine_indices])
            
            y_coarse[temp_indices] = coarse_label
            
        self._fit_coarse(X, y_coarse)
        self._fit_fine(X, y, y_coarse)
        
        self.fitted=True
        
        
    def _fit_coarse(self, X, y_coarse):
        self.coarse_forest = UF(n_estimators=self.n_estimators_coarse, max_depth=self.max_depth).fit(X, y_coarse)
    
    
    def _fit_fine(self, X, y, y_coarse):
        for coarse_label in self.coarse_labels:
            temp_indices = np.where(y_coarse == coarse_label)[0]
            self.fine_forests[coarse_label] = UF(n_estimators=self.n_estimators_fine, max_depth=self.max_depth
                                                ).fit(X[temp_indices], y[temp_indices])
            
    def predict_proba(self, X):
        posteriors = np.zeros((X.shape[0], len(self.classes)))
        coarse_posteriors = self.coarse_forest.predict_proba(X, 0)
        

        #- Hierarchical posteriors & prediction
        for i, coarse_label in enumerate(self.coarse_labels):
            temp_fine_label_indices = np.where(self.fine_to_coarse == coarse_label)[0]

            temp_fine_posteriors = self.fine_forests[coarse_label].predict_proba(X, 0)
            posteriors[:, temp_fine_label_indices] = np.multiply(coarse_posteriors[:, i],
                                                                         temp_fine_posteriors.T
                                                                        ).T
        
        return posteriors
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
        
