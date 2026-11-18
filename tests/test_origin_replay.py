"""Temporal isolation and retention tests; no foundation-model downloads."""
import unittest

import numpy as np
import pandas as pd

from experiments.monthly_origin_replay import replay


class TrackingModel:
    def fit(self, df):
        assert df['date'].max() < pd.Timestamp('2026-02-01')
        assert df['y'].max() < 999
        return self

    def predict(self, df, horizon):
        assert len(df) == 14
        assert pd.isna(df['y'].iloc[-1])
        assert pd.isna(df['rain_mm'].iloc[-1])
        assert df['date'].max() == pd.Timestamp('2026-02-28')
        assert horizon == 12
        return pd.DataFrame({'step':range(1,13),'p10':np.arange(12),
                             'p50':np.arange(12)+1.,'p90':np.arange(12)+2.})


class ReplayTests(unittest.TestCase):
    def test_future_actuals_and_covariates_are_not_given_to_model(self):
        df=pd.DataFrame({'date':pd.date_range('2025-01-31',periods=15,freq='ME'),
                         'y':list(np.arange(13)+1.)+[999.,1000.],
                         'rain_mm':list(np.arange(13)+10.)+[9999.,10000.]})
        rows=replay(df,TrackingModel,'2026-02','2026-02')
        self.assertEqual(len(rows),12)
        self.assertEqual(rows[0]['origin'],'2026-02-01')
        self.assertEqual(rows[0]['target'],'2026-02')
        self.assertEqual(rows[-1]['target'],'2027-01')
        self.assertEqual(rows[0]['actual'],999.)
        self.assertIsNone(rows[2]['actual'])
        self.assertEqual(rows[-1]['p50'],12.)
        self.assertEqual(rows[0]['training_through'],'2026-01-31')

    def test_all_requested_origins_exist_even_without_target_rows(self):
        class Constant:
            def fit(self,df):return self
            def predict(self,df,horizon):
                assert pd.isna(df['y'].iloc[-1])
                return pd.DataFrame({'step':range(1,13),'p10':[0.]*12,'p50':[1.]*12,'p90':[2.]*12})
        df=pd.DataFrame({'date':pd.date_range('2025-01-31',periods=12,freq='ME'),'y':np.ones(12)})
        rows=replay(df,Constant,'2026-01','2026-11')
        self.assertEqual(len(rows),132)
        self.assertTrue(all(r['actual'] is None for r in rows))
        self.assertEqual(len({r['origin'] for r in rows}),11)


if __name__=='__main__':unittest.main()
