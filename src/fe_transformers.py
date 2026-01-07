from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class ZeroImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_values = {
            'REF': '', 
            'ALT': '', 
            'MC': '', 
            'CHROM': '99',
            'dbnsfp.interpro.domain': '',
            'dbnsfp.revel.score': -1,
            'dbnsfp.phylop.100way_vertebrate.score': -1,
            'AF_EXAC': 0,
            'AF_TGP' : 0,
            'AF_ESP' : 0,
            'gnomad_exome.af.af' : 0,
            'GENE_SYMBOL': '',
            'GENEINFO': ''
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, val in self.fill_values.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(val)
        return X_copy
    



class OriginRareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.common_origin = None
        pass

    def fit(self, X, y=None):
        top_origins = X['ORIGIN'].value_counts()
        self.common_origin = set(top_origins[top_origins >= self.threshold].index)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['ORIGIN_GROUPED'] = X_copy['ORIGIN'].apply(lambda x: x if x in self.common_origin else 'other')
        X_copy['ORIGIN_GROUPED'] = X_copy['ORIGIN_GROUPED'].astype(str)
        X_copy = X_copy.drop(columns=['ORIGIN'], axis=1)
        return X_copy
    



class ImpactScoreEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, high_impact_keyword=None, medium_impact_keyword=None):
        if high_impact_keyword is None:
            self.high_impact_keyword = ['nonsense', 'frameshift', 'splice_acceptor', 'splice_donor', 'start_lost', 'stop_lost', 'initiator_codon']
        else:
            self.high_impact_keyword = high_impact_keyword
        if medium_impact_keyword is None:
            self.medium_impact_keyword = ['missense', 'inframe', 'protein_altering']
        else:
            self.medium_impact_keyword = medium_impact_keyword

    def fit(self, X, y=None):
        return self
    
    def _clean_mc_string(self, X):
        mc_series = X['MC'].astype(str)
        extracted = mc_series.str.extract(r'\|(\S*)', expand=False)
        extracted = extracted.fillna('')
        X['MC_CLEANED'] = extracted.str.replace(r',[^|]*', '', regex=True).str.lower()
        return X
    
    def _score_single_value(self, val):
        if any(keyword in val for keyword in self.high_impact_keyword):
            return 2
        if any(keyword in val for keyword in self.medium_impact_keyword):
            return 1
        return 0

    def transform(self, X):
        X_copy = X.copy()
        X_copy = self._clean_mc_string(X_copy)
        X_copy['MC_IMPACT_SCORE'] = X_copy['MC_CLEANED'].apply(self._score_single_value)
        X_copy = X_copy.drop(columns=['MC', 'MC_CLEANED'], axis=1, errors='ignore')
        return X_copy
    



class VariantsAtTributeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.chrom_map = {
            'X': 23,
            'Y': 24,
            'MT': 25, 
            'M': 25,
            'Mit': 25,
            'NT_187693.1': 99
        }
        self.af_cols = ['AF_EXAC', 'AF_TGP', 'AF_ESP', 'gnomad_exome.af.af']

    def fit(self, X, y=None):
        return self
    
    def _is_transition_checker(self, X) -> pd.DataFrame:
        cond_AG = (X['REF'] == 'A') & (X['ALT'] == 'G')
        cond_GA = (X['REF'] == 'G') & (X['ALT'] == 'A')
        cond_CT = (X['REF'] == 'C') & (X['ALT'] == 'T')
        cond_TC = (X['REF'] == 'T') & (X['ALT'] == 'C')
        X['is_transition'] = (cond_AG | cond_GA | cond_CT | cond_TC).astype(int)
        return X

    def transform(self, X) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.af_cols:
            X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
        X_copy['MAX_AF'] = X_copy[self.af_cols].max(axis=1)
        X_copy['DIFF_LEN'] = X_copy['ALT'].astype(str).str.len() - X_copy['REF'].astype(str).str.len()
        X_copy['is_frameshift'] = (X_copy['DIFF_LEN'] % 3 != 0).astype(int)
        X_copy = self._is_transition_checker(X_copy)
        X_copy['CHROM'] = X_copy['CHROM'].astype(str).replace(self.chrom_map)
        X_copy['CHROM'] = pd.to_numeric(X_copy['CHROM'], errors='coerce').fillna(99).astype(int)
        X_copy['is_in_critical_domain'] = X_copy['dbnsfp.interpro.domain'].notna().astype(int)
        X_copy = X_copy.drop(columns=['dbnsfp.interpro.domain', 'ALT', 'REF', 'AF_EXAC', 'AF_TGP', 'AF_ESP', 'gnomad_exome.af.af', 'GENEINFO', 'CLNVC', 'CLNREVSTAT', 'CLNDN'], axis=1)
        return X_copy
    


    
class AddAcmgRules(BaseEstimator, TransformerMixin):
    def __init__(self, af_threshold_benign=0.05, revel_threshold_pathogenic=0.75):
        self.af_threshold_benign = af_threshold_benign
        self.revel_threshold_pathogenic = revel_threshold_pathogenic
        

    def fit(self, X, y=None):
        return self
    
    def _apply_population_rules(self, df):
        df['ACMG_BA1'] = (df['MAX_AF'] > self.af_threshold_benign).astype(int)
        df['ACMG_BS1'] = ((df['MAX_AF'] > 0.01) & (df['MAX_AF'] <= self.af_threshold_benign)).astype(int)
        df['ACMG_PM2'] = (df['MAX_AF'] < 0.0001).astype(int)
        return df
    
    def _apply_structural_rules(self, df):
        df['ACMG_PSV1'] = (df['MC_IMPACT_SCORE'] == 2.0).astype(int)
        df['ACMG_PM4'] = ((df['DIFF_LEN'] != 0) & (df['is_frameshift'] == 0) & (df['MC_IMPACT_SCORE'] != 2.0)).astype(int)
        return df
    
    def _apply_computational_rules(self, df):
        col = 'dbnsfp.revel.score'
        df['ACMG_PP3'] = ((df[col] > self.revel_threshold_pathogenic) & (df[col] != -1)).astype(int)
        return df
    
    def _apply_domain_rules(self, df):
        df['ACMG_PM1'] = df['is_in_critical_domain']
        return df

    def _apply_evolution_rules(self, df):
        df['ACMG_PP2'] = ((df['dbnsfp.phylop.100way_vertebrate.score'] > 2.0) & (df['MC_IMPACT_SCORE'] == 1.0)).astype(int)
        return df 

    def _apply_origin_rules(self, df):
        df['ACMG_PS2'] = ((df['ORIGIN_GROUPED_32'] == 1) | (df['ORIGIN_GROUPED_33'] == 1)).astype(int)
        return df
    
    def _apply_synonumous_rules(self, df):
        is_synonymous = (df['MC_IMPACT_SCORE'] == 0)
        is_safe = True
        is_safe = ((df['dbnsfp.revel.score'] < 0.05) | (df['dbnsfp.revel.score'] == -1))
        df['ACMG_BP7'] = (is_safe & is_synonymous).astype(int)
        return df
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy = self._apply_population_rules(X_copy)
        X_copy = self._apply_structural_rules(X_copy)
        X_copy = self._apply_domain_rules(X_copy)
        X_copy = self._apply_computational_rules(X_copy)
        X_copy = self._apply_evolution_rules(X_copy)
        X_copy = self._apply_origin_rules(X_copy)
        X_copy = self._apply_synonumous_rules(X_copy)
        return X_copy
    



class GeneRiskEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, m=10):
        self.global_mean = None
        self.smooth_map = None
        self.m = m
        pass

    def fit(self, X, y):
        df= X.copy()
        df['TARGET'] = y.copy()
        gene_stats = df.groupby('GENE_SYMBOL')['TARGET'].agg(['count', 'mean'])
        self.global_mean = df['TARGET'].mean()
        self.smooth_map = (gene_stats['count'] * gene_stats['mean'] + self.m * self.global_mean) / (gene_stats['count'] + self.m)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['GENE_RISK'] = X_copy['GENE_SYMBOL'].map(self.smooth_map)
        X_copy['GENE_RISK'] = X_copy['GENE_RISK'].fillna(self.global_mean)
        X_copy = X_copy.drop(columns=['GENE_SYMBOL'], axis=1)
        return X_copy