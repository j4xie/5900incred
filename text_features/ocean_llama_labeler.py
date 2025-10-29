"""
OCEAN 人格特征标注器
使用 Llama 模型生成 Ground Truth 标签
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List
from utils.llama_client import LlamaClient

# OCEAN 五大人格维度
OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


class OceanLlamaLabeler:
    """
    使用 Llama 模型为借款人打 OCEAN 人格分数
    用于生成监督学习的 Ground Truth 标签
    """

    def __init__(self, hf_token: str = None):
        """
        初始化标注器

        Args:
            hf_token: Hugging Face Token (可选，会从环境变量读取)
        """
        self.client = LlamaClient(hf_token)
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "parse_errors": 0
        }

    def _build_prompt(self, row: pd.Series) -> str:
        """
        构建 OCEAN 评分的 Prompt

        Args:
            row: DataFrame 的一行数据（借款人信息）

        Returns:
            格式化的 Prompt
        """
        # 构建借款人画像
        profile_parts = []

        if 'purpose' in row and pd.notna(row['purpose']):
            profile_parts.append(f"- Loan Purpose: {row['purpose']}")

        if 'grade' in row and pd.notna(row['grade']):
            grade_str = f"{row['grade']}"
            if 'sub_grade' in row and pd.notna(row['sub_grade']):
                grade_str += f" ({row['sub_grade']})"
            profile_parts.append(f"- Credit Grade: {grade_str}")

        if 'term' in row and pd.notna(row['term']):
            profile_parts.append(f"- Loan Term: {row['term']}")

        if 'emp_length' in row and pd.notna(row['emp_length']):
            profile_parts.append(f"- Employment Length: {row['emp_length']}")

        if 'home_ownership' in row and pd.notna(row['home_ownership']):
            profile_parts.append(f"- Home Ownership: {row['home_ownership']}")

        if 'verification_status' in row and pd.notna(row['verification_status']):
            profile_parts.append(f"- Income Verification: {row['verification_status']}")

        if 'application_type' in row and pd.notna(row['application_type']):
            profile_parts.append(f"- Application Type: {row['application_type']}")

        profile = "\n".join(profile_parts) if profile_parts else "- Limited information"

        # 构建完整 Prompt（优化版：鼓励使用全范围）
        prompt = f"""You are an expert psychologist specializing in personality assessment for credit risk. Analyze this borrower's Big Five (OCEAN) traits based on their financial behavior patterns.

Borrower Profile:
{profile}

Rate each dimension on a 0.0-1.0 scale. BE DECISIVE and use the FULL scale:

- openness: Curiosity, innovation, risk-taking
  * Grade A/B + conservative purpose (car, home) → 0.2-0.4 (traditional)
  * Grade F/G + risky purpose (business, venture) → 0.7-0.9 (innovative)

- conscientiousness: Responsibility, discipline, planning
  * Grade A + short term + owns home → 0.7-0.9 (highly organized)
  * Grade G + long term + rents → 0.1-0.3 (spontaneous)

- extraversion: Sociability, assertiveness
  * Social purposes (wedding, vacation) → 0.6-0.8 (extroverted)
  * Private purposes (debt consolidation) → 0.2-0.4 (introverted)

- agreeableness: Cooperation, trust
  * Verified income + joint application → 0.6-0.8 (collaborative)
  * Not verified + individual → 0.3-0.5 (competitive)

- neuroticism: Anxiety, emotional instability
  * Grade G + high-risk choices → 0.7-0.9 (anxious)
  * Grade A + stable choices → 0.1-0.3 (calm)

CRITICAL INSTRUCTIONS:
1. Use the FULL 0.0-1.0 range - DO NOT cluster around 0.5
2. Strong signals → extreme scores (0.2 or 0.8, not 0.45)
3. Weak signals → moderate scores (0.4-0.6)
4. Output ONLY valid JSON, no explanation

Example (strong signals):
{{"openness": 0.75, "conscientiousness": 0.25, "extraversion": 0.40, "agreeableness": 0.65, "neuroticism": 0.80}}"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, float]:
        """
        解析 Llama 返回的 JSON

        Args:
            response_text: Llama 模型的原始返回

        Returns:
            OCEAN 分数字典
        """
        try:
            # 提取 JSON（处理可能的额外文本）
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[start:end]
            scores = json.loads(json_str)

            # 验证所有维度存在并标准化
            validated = {}
            for dim in OCEAN_DIMS:
                if dim in scores:
                    val = float(scores[dim])
                    # 如果值 >1，假设是百分制
                    if val > 1.0:
                        val = val / 100.0
                    # Clip 到更宽范围 [0.1, 0.9]（允许极端值）
                    validated[dim] = max(0.1, min(0.9, val))
                else:
                    # 缺失维度用默认值
                    validated[dim] = 0.5

            return validated

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.stats["parse_errors"] += 1
            print(f"⚠️ 解析失败: {e}")
            print(f"   Response: {response_text[:100]}...")
            # 返回默认中值
            return {dim: 0.5 for dim in OCEAN_DIMS}

    def label_sample(self, row: pd.Series, retries: int = 2) -> Dict[str, float]:
        """
        给单个样本打标签

        Args:
            row: 借款人数据
            retries: 失败重试次数

        Returns:
            OCEAN 分数字典
        """
        prompt = self._build_prompt(row)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(retries + 1):
            try:
                response = self.client.query(
                    messages,
                    max_tokens=200,
                    temperature=0  # 确定性输出
                )
                scores = self._parse_response(response)
                self.stats["success"] += 1
                return scores

            except Exception as e:
                if attempt < retries:
                    print(f"⚠️ 尝试 {attempt + 1}/{retries + 1} 失败，重试中...")
                    time.sleep(2)  # 等待后重试
                else:
                    print(f"❌ 样本标注失败: {e}")
                    self.stats["failed"] += 1
                    return {dim: 0.5 for dim in OCEAN_DIMS}

    def label_batch(self,
                    df: pd.DataFrame,
                    sample_size: int = 500,
                    stratified: bool = True,
                    rate_limit_delay: float = 0.5) -> pd.DataFrame:
        """
        批量打标签

        Args:
            df: 原始数据
            sample_size: 抽样数量
            stratified: 是否分层抽样（保证 default/non-default 平衡）
            rate_limit_delay: 每次调用间隔（秒）

        Returns:
            带 OCEAN ground truth 列的 DataFrame
        """
        # 分层抽样
        if stratified and 'target' in df.columns:
            n_positive = min(sample_size // 2, (df['target'] == 1).sum())
            n_negative = min(sample_size // 2, (df['target'] == 0).sum())

            df_pos = df[df['target'] == 1].sample(n=n_positive, random_state=42)
            df_neg = df[df['target'] == 0].sample(n=n_negative, random_state=42)

            sample_df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)
            print(f"[Llama Labeler] 分层抽样: {n_positive} 违约 + {n_negative} 正常")
        else:
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
            print(f"[Llama Labeler] 随机抽样: {len(sample_df)} 个样本")

        sample_df = sample_df.copy()

        print(f"\n开始使用 Llama 模型标注 OCEAN 特征...")
        print(f"模型: {self.client.model}")
        print("=" * 60)

        ocean_labels = []
        total = len(sample_df)

        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            self.stats["total"] += 1

            # 打标签
            scores = self.label_sample(row)
            ocean_labels.append(scores)

            # 进度显示
            if idx % 50 == 0 or idx == total:
                success_rate = self.stats["success"] / self.stats["total"] * 100
                print(f"进度: {idx}/{total} ({idx / total * 100:.1f}%) | "
                      f"成功率: {success_rate:.1f}%")

            # 限流
            if idx < total:
                time.sleep(rate_limit_delay)

        # 添加 ground truth 列
        for dim in OCEAN_DIMS:
            sample_df[f'{dim}_truth'] = [o[dim] for o in ocean_labels]

        print("\n" + "=" * 60)
        print("✅ 标注完成！")
        print(f"统计: {self.stats}")
        print(f"\nOCEAN 分布:")
        print(sample_df[[f'{d}_truth' for d in OCEAN_DIMS]].describe())

        return sample_df

    def get_stats(self) -> Dict:
        """获取标注统计信息"""
        return self.stats.copy()
