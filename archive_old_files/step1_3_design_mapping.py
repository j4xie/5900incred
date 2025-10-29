"""
Step 1.3: 基于数据发现，设计 Variable → OCEAN 的映射规则
目标: 把统计发现转换为心理学假设
"""
import pandas as pd
import json

print("=" * 70)
print("Step 1.3: Design Variable → OCEAN Mapping Rules")
print("=" * 70)

# 基于 Step 1.2 的发现设计映射
print("\n基于数据发现的心理学映射假设:\n")

mapping_rules = {
    "grade": {
        "description": "信用等级反映负责任程度和情绪稳定性",
        "OCEAN_mapping": {
            "A": {"conscientiousness": +0.30, "neuroticism": -0.25, "openness": +0.10},
            "B": {"conscientiousness": +0.20, "neuroticism": -0.15, "openness": +0.05},
            "C": {"conscientiousness": 0, "neuroticism": 0, "openness": 0},
            "D": {"conscientiousness": -0.10, "neuroticism": +0.10},
            "E": {"conscientiousness": -0.15, "neuroticism": +0.15},
            "F": {"conscientiousness": -0.20, "neuroticism": +0.20},
            "G": {"conscientiousness": -0.25, "neuroticism": +0.25}
        },
        "rationale": "高信用=更负责任、情绪稳定；低信用=可能焦虑、冲动"
    },

    "purpose": {
        "description": "借款目的反映生活方式和价值观",
        "OCEAN_mapping": {
            "small_business": {"openness": +0.25, "conscientiousness": +0.10, "extraversion": +0.15},
            "medical": {"neuroticism": +0.15, "agreeableness": +0.10},
            "moving": {"neuroticism": +0.10, "openness": +0.10},
            "debt_consolidation": {"conscientiousness": +0.15, "neuroticism": +0.05},
            "credit_card": {"conscientiousness": +0.10, "neuroticism": -0.05},
            "home_improvement": {"conscientiousness": +0.20, "openness": +0.10, "agreeableness": +0.10},
            "major_purchase": {"conscientiousness": +0.15, "neuroticism": -0.10},
            "wedding": {"agreeableness": +0.20, "extraversion": +0.15, "openness": +0.10},
            "vacation": {"openness": +0.15, "extraversion": +0.10},
            "car": {"conscientiousness": +0.05},
            "renewable_energy": {"openness": +0.20, "conscientiousness": +0.15},
            "house": {"conscientiousness": +0.20, "agreeableness": +0.10},
            "other": {"openness": -0.05}
        },
        "rationale": {
            "small_business": "创业=创新(O)+外向(E)+计划(C)",
            "home_improvement": "改善家居=负责任(C)+开放(O)+友好(A)",
            "wedding": "婚礼=社交(E)+友好(A)+开放(O)",
            "medical": "医疗=可能焦虑(N)+同理心(A)",
            "major_purchase": "大宗采购=计划周密(C)+情绪稳定(N低)"
        }
    },

    "term": {
        "description": "贷款期限反映风险偏好和规划能力",
        "OCEAN_mapping": {
            "36 months": {"conscientiousness": +0.12, "neuroticism": -0.08},
            "60 months": {"conscientiousness": -0.08, "neuroticism": +0.05, "openness": +0.05}
        },
        "rationale": "短期=谨慎规划(C)；长期=可能冲动/乐观(N低,O高)"
    },

    "home_ownership": {
        "description": "住房状况反映稳定性",
        "OCEAN_mapping": {
            "OWN": {"conscientiousness": +0.15, "neuroticism": -0.10, "agreeableness": +0.08},
            "MORTGAGE": {"conscientiousness": +0.10, "neuroticism": -0.05},
            "RENT": {"neuroticism": +0.08, "conscientiousness": -0.05, "openness": +0.05}
        },
        "rationale": "拥有房产=稳定(C,N低)；租房=不稳定(N高)"
    },

    "emp_length": {
        "description": "就业年限反映职业稳定性",
        "OCEAN_mapping": {
            "10+ years": {"conscientiousness": +0.15, "neuroticism": -0.10, "agreeableness": +0.08},
            "< 1 year": {"neuroticism": +0.12, "openness": +0.10, "conscientiousness": -0.08},
            # 其他年限线性插值
        },
        "rationale": "长期就业=稳定负责(C)；频繁跳槽=焦虑(N)或探索(O)"
    },

    "verification_status": {
        "description": "收入验证状态",
        "OCEAN_mapping": {
            "Verified": {"conscientiousness": +0.05, "agreeableness": +0.05},
            "Source Verified": {"conscientiousness": +0.03},
            "Not Verified": {"neuroticism": +0.03, "conscientiousness": -0.03}
        },
        "rationale": "主动验证=透明(C,A)；未验证=可能隐藏信息"
    }
}

# 保存为 JSON
with open('artifacts/results/ocean_mapping_rules.json', 'w', encoding='utf-8') as f:
    json.dump(mapping_rules, f, indent=2, ensure_ascii=False)

print("✅ 映射规则已保存: artifacts/results/ocean_mapping_rules.json\n")

# 打印规则摘要
for var, config in mapping_rules.items():
    print(f"【{var}】")
    print(f"  描述: {config['description']}")
    print(f"  理由: {config['rationale']}")

    if isinstance(config['rationale'], dict):
        for key, reason in config['rationale'].items():
            print(f"    - {key}: {reason}")
    print()

print("=" * 70)
print("映射规则设计完成！")
print("=" * 70)
print("\n权重系数说明:")
print("  +0.25 ~ +0.30  → 强正向影响")
print("  +0.10 ~ +0.20  → 中等正向影响")
print("  +0.03 ~ +0.08  → 弱正向影响")
print("  -0.03 ~ -0.08  → 弱负向影响")
print("  -0.10 ~ -0.20  → 中等负向影响")
print("  -0.25 ~ -0.30  → 强负向影响")
print("\n基准值: 所有 OCEAN 维度初始为 0.50")
print("范围限制: 最终值裁剪到 [0.25, 0.75]")

print("\n下一步: 实现 Rule-Based 打分算法")
print("运行: python3 step2_1_implement_rules.py")
