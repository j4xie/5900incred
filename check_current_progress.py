import numpy as np
import os

temp_file = 'bge_embeddings_2k.temp.npy'

if os.path.exists(temp_file):
    embeddings = np.load(temp_file)
    print(f"✓ 找到临时文件: {temp_file}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  文件大小: {os.path.getsize(temp_file) / 1024 / 1024:.2f} MB")

    # 检查有效数据
    vector_norms = np.linalg.norm(embeddings, axis=1)
    completed = np.sum(vector_norms > 0.01)

    print(f"\n进度:")
    print(f"  已完成: {completed} / {len(embeddings)} ({completed/len(embeddings)*100:.1f}%)")
    print(f"  剩余: {len(embeddings) - completed}")

    # 显示前20个和后20个
    print(f"\n前20个样本:")
    for i in range(min(20, len(embeddings))):
        norm = vector_norms[i]
        status = "✓" if norm > 0.01 else "✗"
        print(f"  {status} [{i:4d}] norm={norm:.4f}")

    print(f"\n样本 [20-40]:")
    for i in range(20, min(40, len(embeddings))):
        norm = vector_norms[i]
        status = "✓" if norm > 0.01 else "✗"
        print(f"  {status} [{i:4d}] norm={norm:.4f}")

else:
    print(f"✗ 文件不存在: {temp_file}")
