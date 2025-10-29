"""
OCEAN 权重学习器
使用监督学习（Ridge Regression）学习 categorical variables → OCEAN 的映射
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


class OceanWeightLearner:
    """
    从 Ground Truth 学习 categorical variables 到 OCEAN 的映射权重
    """

    def __init__(self, method: str = 'ridge', alpha: float = 0.1):
        """
        初始化学习器

        Args:
            method: 回归方法 ('ridge', 'lasso', 'elastic')
            alpha: 正则化强度
        """
        self.method = method
        self.alpha = alpha
        self.weights = {}
        self.encoder = None
        self.cv_scores = {}

    def fit(self,
            X_categorical: pd.DataFrame,
            y_ocean_truth: pd.DataFrame,
            cv: int = 5) -> Tuple[Dict, OneHotEncoder]:
        """
        学习权重

        Args:
            X_categorical: categorical features (500, N_features)
            y_ocean_truth: OCEAN ground truth (500, 5)
            cv: 交叉验证折数

        Returns:
            (learned_weights, encoder)
        """
        print(f"\n{'=' * 60}")
        print(f"OCEAN 权重学习器")
        print(f"方法: {self.method.upper()}, Alpha: {self.alpha}, CV: {cv}")
        print(f"{'=' * 60}\n")

        # 1. One-Hot 编码
        print(f"[Step 1] One-Hot 编码 categorical variables...")
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = self.encoder.fit_transform(X_categorical)
        feature_names = self.encoder.get_feature_names_out(X_categorical.columns)

        print(f"  原始: {X_categorical.shape[1]} 个 categorical variables")
        print(f"  编码后: {X_encoded.shape[1]} 个 binary features")
        print(f"  样本数: {X_encoded.shape[0]}\n")

        # 2. 为每个 OCEAN 维度训练独立模型
        print(f"[Step 2] 训练 {len(OCEAN_DIMS)} 个 OCEAN 维度的权重...\n")

        for dim in OCEAN_DIMS:
            print(f"  训练 {dim}...")

            # 准备目标变量
            y = y_ocean_truth[dim].values

            # 选择模型
            if self.method == 'ridge':
                model = Ridge(alpha=self.alpha, random_state=42)
            elif self.method == 'lasso':
                model = Lasso(alpha=self.alpha, random_state=42, max_iter=5000)
            elif self.method == 'elastic':
                model = ElasticNet(alpha=self.alpha, random_state=42, max_iter=5000)
            else:
                raise ValueError(f"未知方法: {self.method}")

            # 训练
            model.fit(X_encoded, y)

            # 交叉验证
            cv_r2 = cross_val_score(model, X_encoded, y, cv=cv, scoring='r2')
            self.cv_scores[dim] = {
                'mean': cv_r2.mean(),
                'std': cv_r2.std(),
                'scores': cv_r2.tolist()
            }

            # 保存权重
            self.weights[dim] = {
                'intercept': float(model.intercept_),
                'coef': {name: float(coef) for name, coef in zip(feature_names, model.coef_)}
            }

            # 显示结果
            print(f"    ✓ R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

            # 显示 Top 特征
            top_features = sorted(
                self.weights[dim]['coef'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            print(f"    Top 5 features:")
            for feat, weight in top_features:
                sign = "+" if weight >= 0 else ""
                print(f"      {feat[:40]:40s} {sign}{weight:.4f}")
            print()

        print("=" * 60)
        print("✅ 权重学习完成！\n")

        return self.weights, self.encoder

    def get_summary(self) -> pd.DataFrame:
        """获取学习结果摘要"""
        if not self.weights:
            raise ValueError("请先调用 fit() 方法")

        summary = []
        for dim in OCEAN_DIMS:
            summary.append({
                'dimension': dim,
                'r2_mean': self.cv_scores[dim]['mean'],
                'r2_std': self.cv_scores[dim]['std'],
                'intercept': self.weights[dim]['intercept'],
                'n_features': len(self.weights[dim]['coef'])
            })

        return pd.DataFrame(summary)

    def get_top_features(self, dim: str, top_n: int = 10) -> pd.DataFrame:
        """
        获取某个维度的 Top 特征

        Args:
            dim: OCEAN 维度名称
            top_n: 返回前 N 个特征

        Returns:
            DataFrame with columns: [feature, weight, abs_weight]
        """
        if dim not in self.weights:
            raise ValueError(f"维度 {dim} 不存在")

        coef_dict = self.weights[dim]['coef']
        features = []

        for feat, weight in coef_dict.items():
            features.append({
                'feature': feat,
                'weight': weight,
                'abs_weight': abs(weight)
            })

        df = pd.DataFrame(features).sort_values('abs_weight', ascending=False)
        return df.head(top_n)

    def save_weights(self, filepath: str):
        """保存权重到 JSON"""
        import json

        output = {
            'method': self.method,
            'alpha': self.alpha,
            'weights': self.weights,
            'cv_scores': self.cv_scores
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ 权重已保存到: {filepath}")

    @staticmethod
    def load_weights(filepath: str) -> Dict:
        """从 JSON 加载权重"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 权重已加载: {filepath}")
        print(f"   方法: {data['method']}, Alpha: {data['alpha']}")

        return data['weights']
