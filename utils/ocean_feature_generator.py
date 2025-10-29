"""
OCEAN 特征生成器
使用学到的权重为全量数据生成 OCEAN 特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import OneHotEncoder

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


class OceanFeatureGenerator:
    """
    使用学到的权重给新样本生成 OCEAN 特征
    """

    def __init__(self, learned_weights: Dict, encoder: OneHotEncoder):
        """
        初始化生成器

        Args:
            learned_weights: 从 OceanWeightLearner 学到的权重
            encoder: 拟合好的 OneHotEncoder
        """
        self.weights = learned_weights
        self.encoder = encoder

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        给全量数据生成 OCEAN 特征

        Args:
            df: 包含 categorical variables 的 DataFrame

        Returns:
            df + OCEAN 特征列
        """
        print(f"\n{'=' * 60}")
        print(f"OCEAN 特征生成器")
        print(f"{'=' * 60}\n")

        df = df.copy()

        # 1. 获取 categorical variables
        categorical_vars = list(self.encoder.feature_names_in_)
        print(f"[Step 1] 检测 categorical variables...")
        print(f"  需要的列: {categorical_vars}")

        # 检查缺失列
        missing_cols = [col for col in categorical_vars if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺失列: {missing_cols}")

        # 2. One-Hot 编码
        print(f"\n[Step 2] One-Hot 编码...")
        X_categorical = df[categorical_vars]
        X_encoded = self.encoder.transform(X_categorical)
        feature_names = self.encoder.get_feature_names_out(categorical_vars)

        print(f"  编码后特征数: {X_encoded.shape[1]}")
        print(f"  样本数: {X_encoded.shape[0]}")

        # 3. 计算 OCEAN 分数
        print(f"\n[Step 3] 计算 OCEAN 分数...")

        for dim in OCEAN_DIMS:
            intercept = self.weights[dim]['intercept']
            coef_dict = self.weights[dim]['coef']

            # 初始化分数（从 intercept 开始）
            scores = np.full(len(df), intercept)

            # 加权求和
            for feat_name, weight in coef_dict.items():
                # 找到特征在编码矩阵中的位置
                if feat_name in feature_names:
                    feat_idx = list(feature_names).index(feat_name)
                    scores += X_encoded[:, feat_idx] * weight

            # Clip 到更宽范围 [0.1, 0.9]（允许极端值）
            scores = np.clip(scores, 0.1, 0.9)

            # 添加到 DataFrame
            df[dim] = scores

            print(f"  ✓ {dim:20s} 均值={scores.mean():.3f}, 标准差={scores.std():.3f}")

        print("\n" + "=" * 60)
        print("✅ OCEAN 特征生成完成！\n")
        print("生成的特征列:", OCEAN_DIMS)
        print("\nOCEAN 特征统计:")
        print(df[OCEAN_DIMS].describe())

        return df

    def generate_single(self, borrower: Dict) -> Dict[str, float]:
        """
        给单个借款人生成 OCEAN 分数

        Args:
            borrower: 包含 categorical variables 的字典

        Returns:
            OCEAN 分数字典
        """
        # 转为 DataFrame
        df = pd.DataFrame([borrower])

        # 生成特征
        df_with_ocean = self.generate_features(df)

        # 提取 OCEAN 分数
        ocean_scores = {}
        for dim in OCEAN_DIMS:
            ocean_scores[dim] = float(df_with_ocean[dim].iloc[0])

        return ocean_scores

    def get_feature_contribution(self, borrower: Dict, dim: str) -> pd.DataFrame:
        """
        分析某个借款人在某个 OCEAN 维度上各特征的贡献

        Args:
            borrower: 借款人信息
            dim: OCEAN 维度

        Returns:
            DataFrame with columns: [feature, value, weight, contribution]
        """
        if dim not in self.weights:
            raise ValueError(f"维度 {dim} 不存在")

        # One-Hot 编码
        categorical_vars = list(self.encoder.feature_names_in_)
        df = pd.DataFrame([borrower])
        X_encoded = self.encoder.transform(df[categorical_vars])
        feature_names = self.encoder.get_feature_names_out(categorical_vars)

        # 计算贡献
        intercept = self.weights[dim]['intercept']
        coef_dict = self.weights[dim]['coef']

        contributions = []

        # Intercept
        contributions.append({
            'feature': 'intercept',
            'value': 1.0,
            'weight': intercept,
            'contribution': intercept
        })

        # 各特征
        for i, feat_name in enumerate(feature_names):
            value = X_encoded[0, i]
            weight = coef_dict.get(feat_name, 0.0)
            contribution = value * weight

            if value > 0:  # 只显示激活的特征
                contributions.append({
                    'feature': feat_name,
                    'value': value,
                    'weight': weight,
                    'contribution': contribution
                })

        df_contrib = pd.DataFrame(contributions).sort_values('contribution', ascending=False)

        return df_contrib

    def explain_score(self, borrower: Dict, dim: str, top_n: int = 5):
        """
        解释某个借款人的 OCEAN 分数

        Args:
            borrower: 借款人信息
            dim: OCEAN 维度
            top_n: 显示前 N 个贡献特征
        """
        contrib = self.get_feature_contribution(borrower, dim)

        total_score = contrib['contribution'].sum()

        print(f"\n{'=' * 60}")
        print(f"{dim.upper()} 分数解释")
        print(f"{'=' * 60}")
        print(f"最终分数: {total_score:.3f} (Clipped to [0.25, 0.75])")
        print(f"\nTop {top_n} 贡献特征:\n")

        for i, row in contrib.head(top_n).iterrows():
            sign = "+" if row['contribution'] >= 0 else ""
            print(f"  {row['feature']:40s} {sign}{row['contribution']:.4f}")
            print(f"    (value={row['value']:.1f} × weight={row['weight']:.4f})")
            print()
