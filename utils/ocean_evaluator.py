"""
OCEAN 特征评估器
评估生成的 OCEAN 特征质量和预测能力
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import roc_auc_score
from scipy import stats

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


class OceanEvaluator:
    """
    评估 OCEAN 特征的质量
    """

    def __init__(self):
        self.reports = {}

    def evaluate_ground_truth_quality(self, df_truth: pd.DataFrame) -> Dict:
        """
        评估 Ground Truth 标注质量

        Args:
            df_truth: 包含 OCEAN_truth 列的 DataFrame

        Returns:
            质量报告
        """
        print(f"\n{'=' * 60}")
        print("Ground Truth 质量评估")
        print(f"{'=' * 60}\n")

        report = {}
        truth_cols = [f'{dim}_truth' for dim in OCEAN_DIMS]

        for col in truth_cols:
            if col not in df_truth.columns:
                continue

            dim = col.replace('_truth', '')
            values = df_truth[col].dropna()

            report[dim] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q50': float(values.quantile(0.50)),
                'q75': float(values.quantile(0.75))
            }

            # 评估
            std = report[dim]['std']
            status = "✅" if std > 0.1 else "⚠️"
            print(f"{dim:20s} {status}")
            print(f"  均值={report[dim]['mean']:.3f}, 标准差={report[dim]['std']:.3f}")
            print(f"  范围=[{report[dim]['min']:.3f}, {report[dim]['max']:.3f}]")

            if std < 0.1:
                print(f"  ⚠️ 警告: 区分度偏低（std < 0.1）")
            print()

        self.reports['ground_truth'] = report
        return report

    def evaluate_predictive_power(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        """
        评估 OCEAN 特征的预测能力

        Args:
            df: 包含 OCEAN 和 target 列的 DataFrame
            target_col: 目标变量列名

        Returns:
            预测能力报告
        """
        print(f"\n{'=' * 60}")
        print("OCEAN 预测能力评估")
        print(f"{'=' * 60}\n")

        if target_col not in df.columns:
            raise ValueError(f"目标列 {target_col} 不存在")

        report = {}
        y = df[target_col].values

        print(f"目标分布: {(y == 0).sum()} 正常 / {(y == 1).sum()} 违约\n")

        for dim in OCEAN_DIMS:
            if dim not in df.columns:
                continue

            x = df[dim].values

            # 1. 单变量 AUC
            try:
                auc = roc_auc_score(y, x)
            except:
                auc = 0.5

            # 2. 点二列相关（Point-biserial correlation）
            corr, p_value = stats.pointbiserialr(y, x)

            # 3. 分组均值差异（t-test）
            group_0 = x[y == 0]
            group_1 = x[y == 1]
            mean_diff = group_1.mean() - group_0.mean()
            t_stat, t_pval = stats.ttest_ind(group_1, group_0)

            # 4. IV (Information Value)
            iv = self._compute_iv(x, y)

            report[dim] = {
                'auc': float(auc),
                'correlation': float(corr),
                'corr_pvalue': float(p_value),
                'mean_default': float(group_1.mean()),
                'mean_paid': float(group_0.mean()),
                'mean_diff': float(mean_diff),
                't_statistic': float(t_stat),
                't_pvalue': float(t_pval),
                'iv': float(iv)
            }

            # 评估
            status = self._evaluate_feature(dim, report[dim])
            print(f"{dim:20s} {status}")
            print(f"  AUC={auc:.3f}, Corr={corr:+.3f} (p={p_value:.4f})")
            print(f"  均值差={mean_diff:+.3f} (违约-正常)")
            print(f"  IV={iv:.3f}")
            print()

        self.reports['predictive_power'] = report
        return report

    def _compute_iv(self, x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """计算 Information Value"""
        try:
            # 分箱
            bins = pd.qcut(x, q=n_bins, duplicates='drop')
            df = pd.DataFrame({'x': x, 'y': y, 'bin': bins})

            # 计算 IV
            iv = 0.0
            for bin_val in df['bin'].unique():
                bin_data = df[df['bin'] == bin_val]
                n_pos = (bin_data['y'] == 1).sum()
                n_neg = (bin_data['y'] == 0).sum()

                if n_pos == 0 or n_neg == 0:
                    continue

                total_pos = (y == 1).sum()
                total_neg = (y == 0).sum()

                pct_pos = n_pos / total_pos
                pct_neg = n_neg / total_neg

                if pct_pos > 0 and pct_neg > 0:
                    woe = np.log(pct_pos / pct_neg)
                    iv += (pct_pos - pct_neg) * woe

            return iv
        except:
            return 0.0

    def _evaluate_feature(self, dim: str, metrics: Dict) -> str:
        """
        评估单个特征

        返回状态标记：
        ✅ 有效
        ⚠️ 一般
        ❌ 无效
        """
        auc = metrics['auc']
        corr = abs(metrics['correlation'])
        corr_p = metrics['corr_pvalue']
        iv = metrics['iv']

        # 心理学期望
        expected_sign = {
            'conscientiousness': -1,  # 尽责性高 → 违约低
            'neuroticism': 1,          # 焦虑高 → 违约高
            'openness': 0,             # 不确定
            'extraversion': 0,         # 不确定
            'agreeableness': 0         # 不确定
        }

        actual_sign = np.sign(metrics['correlation'])
        exp_sign = expected_sign.get(dim, 0)

        # 判断
        if corr > 0.05 and corr_p < 0.05:
            if exp_sign != 0 and actual_sign == exp_sign:
                return "✅ 有效（符合预期）"
            else:
                return "✅ 有效"
        elif corr > 0.03:
            return "⚠️ 一般（弱相关）"
        else:
            return "❌ 无效（无相关）"

    def generate_summary_report(self) -> pd.DataFrame:
        """生成汇总报告"""
        if 'predictive_power' not in self.reports:
            raise ValueError("请先调用 evaluate_predictive_power()")

        rows = []
        for dim in OCEAN_DIMS:
            if dim in self.reports['predictive_power']:
                metrics = self.reports['predictive_power'][dim]
                status = self._evaluate_feature(dim, metrics)

                rows.append({
                    'dimension': dim,
                    'auc': metrics['auc'],
                    'correlation': metrics['correlation'],
                    'p_value': metrics['corr_pvalue'],
                    'iv': metrics['iv'],
                    'status': status
                })

        return pd.DataFrame(rows)

    def compare_models(self,
                       y_true: np.ndarray,
                       y_proba_baseline: np.ndarray,
                       y_proba_ocean: np.ndarray) -> Dict:
        """
        比较 Baseline vs Baseline+OCEAN 模型

        Args:
            y_true: 真实标签
            y_proba_baseline: Baseline 模型预测概率
            y_proba_ocean: Baseline+OCEAN 模型预测概率

        Returns:
            对比报告
        """
        from utils.metrics import compute_all_metrics, delong_test

        print(f"\n{'=' * 60}")
        print("模型对比: Baseline vs Baseline+OCEAN")
        print(f"{'=' * 60}\n")

        # 计算指标
        metrics_baseline = compute_all_metrics(y_true, y_proba_baseline)
        metrics_ocean = compute_all_metrics(y_true, y_proba_ocean)

        # DeLong 检验
        z_stat, p_val = delong_test(y_true, y_proba_baseline, y_proba_ocean)

        # 打印对比
        print(f"{'Metric':<15} {'Baseline':<15} {'+OCEAN':<15} {'Delta':<15}")
        print("-" * 60)

        for key in ['roc_auc', 'pr_auc', 'ks', 'brier']:
            val_base = metrics_baseline[key]
            val_ocean = metrics_ocean[key]
            delta = val_ocean - val_base

            print(f"{key:<15} {val_base:<15.4f} {val_ocean:<15.4f} {delta:+.4f}")

        print("\n" + "-" * 60)
        print(f"DeLong test: z={z_stat:.3f}, p={p_val:.4f}")
        significant = "✅ 显著" if p_val < 0.05 else "❌ 不显著"
        print(f"统计显著性 (α=0.05): {significant}")

        report = {
            'baseline': metrics_baseline,
            'ocean': metrics_ocean,
            'delta': {k: metrics_ocean[k] - metrics_baseline[k]
                     for k in ['roc_auc', 'pr_auc', 'ks', 'brier']},
            'delong_z': z_stat,
            'delong_p': p_val,
            'significant': p_val < 0.05
        }

        self.reports['model_comparison'] = report
        return report
