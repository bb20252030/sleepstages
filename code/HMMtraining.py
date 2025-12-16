"""
HMM-based NREM substate model selection script
===================================================
Strategy: Train Gaussian Hidden Markov Models (HMMs) on pre-extracted neural
features to discover latent NREM substates, and determine the optimal number
of states using mouse-wise cross-validation and information criteria.

Key advantages:
1. Organizes data into temporally continuous sequences per mouse
2. Performs mouse-level K-fold cross-validation to avoid data leakage
3. Supports optional feature preprocessing (standardization + PCA)
4. Selects optimal state number using CV log-likelihood, AIC, and BIC
5. Saves all core results immediately; visualization is optional and non-blocking

Author: Sun Shibin
Date: 2025-10-21
"""
import numpy as np
import pickle
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold
import umap
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class HMMModelSelection:
    """
    HMM模型选择器：用于确定NREM睡眠的最优子状态数量
    - 十折交叉验证
    - UMAP/t-SNE可视化
    """

    def __init__(self, random_state=42):
        """
        初始化HMM模型选择器
        
        参数:
        - random_state: 随机种子，保证结果可复现
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.umap_reducer = None
        self.models = {}
        self.selection_results = {}
        self.cv_results = {}
        self.mouse_data = {}  # 存储按小鼠组织的数据

    def load_features(self, features_file):
        """
        加载提取的CNN特征
        
        参数:
        - features_file: 特征文件路径
        """
        print(f"📂 Loading features from: {features_file}")
        with open(features_file, 'rb') as f:
            data = pickle.load(f)

        self.features = data['features']
        self.metadata = data['metadata']

        # 统计小鼠信息
        mouse_ids = list(set([m['mouse_id'] for m in self.metadata]))
        print(f"✓ Loaded {len(self.features)} epochs")
        print(f"✓ Feature dimension: {self.features.shape[1]}")
        print(f"✓ From {len(mouse_ids)} mice: {mouse_ids}")
        
        return self.features, self.metadata
    
    def preprocess_features(self, use_pca=True, pca_components=32):
        """
        预处理特征：标准化 + 可选的PCA降维
        
        参数:
        - use_pca: 是否使用PCA降维
        - pca_components: PCA主成分数量
        """
        print(f"🔧 Preprocessing features...")

        # 标准化
        X_scaled = self.scaler.fit_transform(self.features)
        print(f"✓ Standardized features")

        # 可选的PCA降维
        if use_pca and self.features.shape[1] > pca_components:
            self.pca = PCA(n_components=pca_components, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_scaled)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"✓ PCA: {self.features.shape[1]} → {pca_components} dims")
            print(f"✓ Explained variance: {explained_variance:.3f}")
        else:
            X_processed = X_scaled
            print(f"✓ No PCA applied")
        
        self.X_processed = X_processed
        return X_processed
    
    def organize_sequences_by_mouse(self, min_segment_length=5):
        """
        按照小鼠+时间连续性组织序列数据
        HMM需要序列数据,我们按小鼠分组
        检测时间不连续的gap，拆分成多个segment

        参数:
        - min_segment_length: 最短segment长度（默认5个epoch）
        """
        print(f"📋 Organizing sequences by mouse...")

        # 按小鼠分组
        mouse_sequences = {}
        for i, meta in enumerate(self.metadata):
            mouse_id = meta['mouse_id']
            epoch_idx = meta['epoch_index']
            if mouse_id not in mouse_sequences:
                mouse_sequences[mouse_id] = []
            mouse_sequences[mouse_id].append({
                'feature_idx': i,
                'epoch_idx': epoch_idx
            })
        
        # 为每个小鼠创建序列数据和长度
        self.mouse_data = {}
        all_sequences = []
        all_lengths = []
        segment_metadata = []  # 新增：记录每个segment的元信息

        total_segments = 0
        discarded_epochs = 0

        for mouse_id, items in mouse_sequences.items():
            # 按照epoch顺序排序
            items.sort(key=lambda x: x['epoch_idx'])

            # 检测连续性gap
            segments = []
            current_segment = [items[0]]

            for i in range(1, len(items)):
                prev_epoch = items[i-1]['epoch_idx']
                curr_epoch = items[i]['epoch_idx']

                # 如果epoch不连续（gap > 1），开始新segment
                if curr_epoch - prev_epoch > 1:
                    segments.append(current_segment)
                    current_segment = [items[i]]
                else:
                    current_segment.append(items[i])

            # 添加最后一个segment
            if current_segment:
                segments.append(current_segment)

            # 过滤太短的segment + 提取特征
            mouse_segments = []
            for seg_idx, segment in enumerate(segments):
                if len(segment) >= min_segment_length:
                    # 提取该segment的特征索引
                    feature_indices = [item['feature_idx'] for item in segment]
                    epoch_indices = [item['epoch_idx'] for item in segment]

                    # 获取特征
                    segment_features = self.X_processed[feature_indices]

                    # 记录segment信息
                    mouse_segments.append({
                        'features': segment_features,
                        'length': len(segment_features),
                        'start_epoch': epoch_indices[0],
                        'end_epoch': epoch_indices[-1],
                        'segment_id': f"{mouse_id}_seg{seg_idx}"
                    })
                    all_sequences.append(segment_features)
                    all_lengths.append(len(segment_features))
                    segment_metadata.append({
                        'mouse_id': mouse_id,
                        'segment_id': f"{mouse_id}_seg{seg_idx}",
                        'n_epochs': len(segment_features),
                        'start_epoch': epoch_indices[0],
                        'end_epoch': epoch_indices[-1]
                    })

                    total_segments += 1
                else:
                    discarded_epochs += len(segment)

            if mouse_segments:
                self.mouse_data[mouse_id] = {
                    'segments': mouse_segments,
                    'n_segments': len(mouse_segments),
                    'total_epochs': sum(s['length'] for s in mouse_segments)
                }
        
        # 合并所有序列（用于整体训练）
        self.X_sequences = np.vstack(all_sequences)
        self.sequence_lengths = all_lengths
        self.mouse_ids = list(self.mouse_data.keys())
        self.segment_metadata = segment_metadata  # 保存segment元信息
        
        print(f"✓ Organized {len(self.mouse_ids)} mice")
        print(f"✓ Sequence lengths: {min(all_lengths)} - {max(all_lengths)}")
        print(f"✓ Total epochs: {len(self.X_sequences)}")
        if discarded_epochs > 0:
            print(f"⚠️  Discarded {discarded_epochs} epochs in segments < {min_segment_length}")
        # 打印每只小鼠的segment统计
        print(f"\n📊 Per-mouse segment statistics:")
        for mouse_id in sorted(self.mouse_ids):
            n_seg = self.mouse_data[mouse_id]['n_segments']
            n_epochs = self.mouse_data[mouse_id]['total_epochs']
            print(f"  {mouse_id}: {n_seg} segments, {n_epochs} epochs")
        return self.X_sequences, self.sequence_lengths
    
    def cross_validation_split(self, n_folds=10):
        """
        创建基于小鼠的十折交叉验证分割
        """
        print(f"🔄 Creating {n_folds}-fold cross-validation splits...")
        
        # 确保有足够的小鼠进行交叉验证
        if len(self.mouse_ids) < n_folds:
            print(f"⚠️ Warning: Only {len(self.mouse_ids)} mice available, using {len(self.mouse_ids)}-fold CV")
            n_folds = len(self.mouse_ids)
        
        # 基于小鼠的K折分割
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        mouse_array = np.array(self.mouse_ids)
        
        cv_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(mouse_array)):
            train_mice = mouse_array[train_idx].tolist()
            val_mice = mouse_array[val_idx].tolist()
            
            # 收集训练和验证数据
            train_sequences = []
            train_lengths = []
            val_sequences = []
            val_lengths = []
            
            for mouse_id in train_mice:
                for segment in self.mouse_data[mouse_id]['segments']:
                    train_sequences.append(segment['features'])
                    train_lengths.append(segment['length'])
            
            for mouse_id in val_mice:
                for segment in self.mouse_data[mouse_id]['segments']:
                    val_sequences.append(segment['features'])
                    val_lengths.append(segment['length'])
            
            X_train = np.vstack(train_sequences) if train_sequences else np.empty((0, self.X_processed.shape[1]))
            X_val = np.vstack(val_sequences) if val_sequences else np.empty((0, self.X_processed.shape[1]))
            
            cv_splits.append({
                'fold': fold_idx,
                'train_mice': train_mice,
                'val_mice': val_mice,
                'X_train': X_train,
                'train_lengths': train_lengths,
                'X_val': X_val,
                'val_lengths': val_lengths
            })
            
            print(f"  Fold {fold_idx+1}: Train={len(train_mice)} mice ({len(X_train)} epochs), "
                  f"Val={len(val_mice)} mice ({len(X_val)} epochs)")
        
        self.cv_splits = cv_splits
        return cv_splits
    
    def train_hmm_with_cv(self, n_states, n_iter=100, tol=1e-4):
        """
        使用交叉验证训练HMM模型
        """
        print(f"🎯 Training HMM with {n_states} states using cross-validation...")
        
        cv_scores = []
        fold_models = []
        
        for split in self.cv_splits:
            fold_idx = split['fold']
            X_train = split['X_train']
            train_lengths = split['train_lengths']
            X_val = split['X_val']
            val_lengths = split['val_lengths']
            
            if len(X_train) == 0 or len(X_val) == 0:
                print(f"  Fold {fold_idx+1}: Skipped (empty data)")
                continue
            
            try:
                # 训练模型
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=n_iter,
                    tol=tol,
                    random_state=self.random_state + fold_idx,
                    verbose=False
                )
                
                model.fit(X_train, train_lengths)
                
                # 验证模型
                val_score = model.score(X_val, val_lengths)
                cv_scores.append(val_score)
                fold_models.append(model)
                
                print(f"  Fold {fold_idx+1}: Val score = {val_score:.2f}")
                
            except Exception as e:
                print(f"  Fold {fold_idx+1}: Failed - {e}")
                continue
        
        if cv_scores:
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            print(f"  ✓ CV Score: {cv_mean:.2f} ± {cv_std:.2f}")
            
            # 在全部数据上训练最终模型
            final_model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                tol=tol,
                random_state=self.random_state,
                verbose=False
            )
            
            final_model.fit(self.X_sequences, self.sequence_lengths)
            final_score = final_model.score(self.X_sequences, self.sequence_lengths)
            
            # 计算信息准则
            n_params = self._count_parameters(final_model)
            n_samples = len(self.X_sequences)
            
            aic = -2 * final_score + 2 * n_params
            bic = -2 * final_score + n_params * np.log(n_samples)
            
            return final_model, {
                'cv_score_mean': cv_mean,
                'cv_score_std': cv_std,
                'cv_scores': cv_scores,
                'log_likelihood': final_score,
                'aic': aic,
                'bic': bic,
                'n_params': n_params,
                'n_folds': len(cv_scores),
                'converged': final_model.monitor_.converged
            }
        else:
            return None, None
    
    def _count_parameters(self, model):
        """
        计算HMM模型的参数数量
        """

        n_states = model.n_components
        n_features = model.means_.shape[1]

        # 转移矩阵参数: (n_states-1) * n_states
        transition_params = n_states * (n_states - 1)

        # 初始状态概率: n_states - 1
        start_params = n_states - 1

        # 发射概率参数 (高斯分布):
        # 均值: n_states * n_features
        # 协方差矩阵: n_states * n_features * (n_features + 1) / 2
        emission_params = n_states * n_features  # 均值

        if model.covariance_type == "full":
            emission_params += n_states * n_features * (n_features + 1) // 2
        elif model.covariance_type == "diag":
            emission_params += n_states * n_features
        elif model.covariance_type == "spherical":
            emission_params += n_states

        return transition_params + start_params + emission_params
    
    def model_selection_with_cv(self, state_range=(2, 5), n_iter=100, n_folds=10):
        """
        使用交叉验证进行模型选择
        """
        print(f"🔍 Model selection with {n_folds}-fold CV for {state_range[0]} to {state_range[1]-1} states...")
        
        # 创建交叉验证分割
        self.cross_validation_split(n_folds=n_folds)
        
        results = {}
        
        for n_states in range(state_range[0], state_range[1]):
            print(f"\n--- K = {n_states} ---")
            
            model, metrics = self.train_hmm_with_cv(n_states, n_iter=n_iter)
            
            if model is not None:
                self.models[n_states] = model
                results[n_states] = metrics
                
                # 预测隐藏状态
                hidden_states = model.predict(self.X_sequences, self.sequence_lengths)
                results[n_states]['hidden_states'] = hidden_states

                # 后验概率 γ_t(k)
                _, posteriors = model.score_samples(self.X_sequences, self.sequence_lengths)
                results[n_states]['posteriors'] = posteriors
                
                # 计算状态分布
                state_counts = np.bincount(hidden_states, minlength=n_states)
                state_probs = state_counts / len(hidden_states)
                results[n_states]['state_distribution'] = state_probs
                
                print(f"  ✓ State distribution: {state_probs}")
            else:
                print(f"  ✗ Model training failed")
        
        self.cv_results = results
        return results
    
    def find_optimal_states_cv(self, criterion='bic'):
        """
        根据交叉验证结果和信息准则找到最优状态数
        """
        if not self.cv_results:
            raise ValueError("No CV results available. Run model_selection_with_cv() first.")
        
        print(f"\n📊 Model selection results with CV ({criterion.upper()}):")
        
        states = []
        cv_scores = []
        criteria_values = []
        
        for n_states, metrics in self.cv_results.items():
            states.append(n_states)
            cv_scores.append(metrics['cv_score_mean'])
            criteria_values.append(metrics[criterion])
            
            print(f"  K={n_states}: {criterion.upper()}={metrics[criterion]:.2f}, "
                  f"CV={metrics['cv_score_mean']:.2f}±{metrics['cv_score_std']:.2f}, "
                  f"LL={metrics['log_likelihood']:.2f}")
        
        # 根据准则找到最优值
        best_idx = np.argmin(criteria_values)
        optimal_states = states[best_idx]
        
        # 也检查CV分数最高的模型
        best_cv_idx = np.argmax(cv_scores)
        best_cv_states = states[best_cv_idx]
        
        print(f"\n🎯 Optimal by {criterion.upper()}: K = {optimal_states}")
        print(f"🎯 Optimal by CV score: K = {best_cv_states}")
        
        # 优先选择信息准则，但也报告CV最优
        self.optimal_states = optimal_states
        self.optimal_model = self.models[optimal_states]
        self.best_cv_states = best_cv_states
        
        return optimal_states, self.optimal_model
    
    def prepare_visualization_data(self, use_umap=True, n_neighbors=15, min_dist=0.1):
        """
        准备可视化数据：UMAP或t-SNE降维
        ⚠️  使用X_sequences而不是X_processed，确保维度与hidden_states匹配
        """
        print(f"🎨 Preparing visualization data...")
        
        if use_umap:
            print(f"  Using UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
            self.umap_reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=self.random_state,
                verbose=False
            )
            
            embedding = self.umap_reducer.fit_transform(self.X_sequences)
            method = "UMAP"
        else:
            print(f"  Using t-SNE")
            from sklearn.manifold import TSNE
            tsne = TSNE(
                n_components=2,
                random_state=self.random_state,
                perplexity=min(30, len(self.X_sequences)//4)
            )
            
            embedding = tsne.fit_transform(self.X_sequences)
            method = "t-SNE"
        
        self.embedding = embedding
        self.embedding_method = method
        
        print(f"  ✓ {method} embedding computed: {embedding.shape}")
        print(f"  ✓ Matches hidden_states dimension: {len(self.cv_results[list(self.cv_results.keys())[0]]['hidden_states'])}")
        return embedding
    
    def plot_enhanced_results(self, save_dir=None):
        """
        绘制增强版结果：模型选择 + 两类可视化
        ⚠️  这个函数现在是可选的，即使失败也不影响核心结果
        """
        if not self.cv_results:
            raise ValueError("No CV results available.")
        
        print(f"\n{'='*60}")
        print("📊 Starting visualization (optional step)")
        print(f"{'='*60}\n")
        
        try:
            # 准备可视化数据
            if not hasattr(self, 'embedding'):
                print("  Preparing embedding for visualization...")
                self.prepare_visualization_data()
            
            # 创建第一个大图：模型选择结果
            print("  Creating model selection metrics plot...")
            self._plot_model_selection_metrics(save_dir)
            
            # 创建第二个大图：状态为主的可视化
            print("  Creating state-centered visualization...")
            self._plot_state_centered_visualization(save_dir)
            
            print(f"\n✅ All visualizations completed successfully!")
            
        except Exception as e:
            print(f"\n⚠️  Warning: Visualization failed, but core results are already saved!")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_model_selection_metrics(self, save_dir=None):
        """
        绘制模型选择指标
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('HMM Model Selection Metrics', fontsize=16)
        
        states = list(self.cv_results.keys())
        
        # 1. AIC/BIC
        aic_values = [self.cv_results[k]['aic'] for k in states]
        bic_values = [self.cv_results[k]['bic'] for k in states]
        
        axes[0, 0].plot(states, aic_values, 'bo-', linewidth=2, markersize=8, label='AIC')
        axes[0, 0].plot(states, bic_values, 'ro-', linewidth=2, markersize=8, label='BIC')
        axes[0, 0].set_xlabel('Number of States (K)')
        axes[0, 0].set_ylabel('Information Criterion')
        axes[0, 0].set_title('AIC/BIC vs Number of States')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cross-validation scores
        cv_means = [self.cv_results[k]['cv_score_mean'] for k in states]
        cv_stds = [self.cv_results[k]['cv_score_std'] for k in states]
        
        axes[0, 1].errorbar(states, cv_means, yerr=cv_stds, fmt='go-', linewidth=2, 
                           markersize=8, capsize=5, capthick=2)
        axes[0, 1].set_xlabel('Number of States (K)')
        axes[0, 1].set_ylabel('CV Log-likelihood')
        axes[0, 1].set_title('Cross-Validation Scores')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. State distribution comparison
        for k in states:
            state_dist = self.cv_results[k]['state_distribution']
            x_pos = np.arange(len(state_dist)) + (k-min(states))*0.1
            axes[1, 0].bar(x_pos, state_dist, width=0.08, alpha=0.7, label=f'K={k}')
        
        axes[1, 0].set_xlabel('State Index')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].set_title('State Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model complexity
        n_params = [self.cv_results[k]['n_params'] for k in states]
        log_likelihoods = [self.cv_results[k]['log_likelihood'] for k in states]
        
        axes[1, 1].plot(n_params, log_likelihoods, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Number of Parameters')
        axes[1, 1].set_ylabel('Log-likelihood')
        axes[1, 1].set_title('Model Complexity vs Fit')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_file = os.path.join(save_dir, 'model_selection_metrics.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"    ✓ Model selection metrics saved to: {plot_file}")
        
        plt.close()
    
    def _plot_state_centered_visualization(self, save_dir=None):
        """
        绘制以状态为中心的可视化（主要用于评估聚类质量）
        """
        states = list(self.cv_results.keys())
        n_states = len(states)
        
        # 动态显示所有K值
        fig, axes = plt.subplots(1, n_states, figsize=(6*n_states, 6))

        # 确保axes总是数组，便于统一处理
        if n_states == 1:
            axes = [axes]
        
        fig.suptitle(f'State-Centered Visualization ({self.embedding_method})\n'
                    f'Each color represents a different HMM state', fontsize=16)
        
        # 为状态定义颜色
        state_colors = plt.cm.Set1(np.arange(max(states)))  # 使用区分度高的颜色映射
        
        for idx, k in enumerate(states):
            ax = axes[idx]
            
            if k in self.cv_results:
                hidden_states = self.cv_results[k]['hidden_states']
                
                # 验证维度匹配
                assert len(hidden_states) == len(self.embedding), \
                    f"Dimension mismatch: hidden_states={len(hidden_states)}, embedding={len(self.embedding)}"
                
                # 为每个状态绘制不同颜色的点
                for state in range(k):
                    state_mask = hidden_states == state
                    if np.any(state_mask):
                        ax.scatter(
                            self.embedding[state_mask, 0],
                            self.embedding[state_mask, 1],
                            c=[state_colors[state]],
                            s=20, alpha=0.7,
                            label=f'State {state}',
                            edgecolors='black', linewidth=0.1
                        )
                
                # 计算每个状态的样本数
                state_counts =  np.bincount(hidden_states, minlength=k)
                state_percentages = state_counts / len(hidden_states) * 100
                
                # 创建标题，包含详细信息
                title_lines = [f'K={k} States']
                for s in range(k):
                    title_lines.append(f'S{s}: {state_counts[s]} ({state_percentages[s]:.1f}%)')
                ax.set_title('\n'.join(title_lines), fontsize=11)
                ax.set_xlabel(f'{self.embedding_method} 1')
                ax.set_ylabel(f'{self.embedding_method} 2')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_file = os.path.join(save_dir, 'state_centered_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"    ✓ State-centered visualization saved to: {plot_file}")
        
        plt.close()
    
    def save_enhanced_results(self, output_dir):
        """
        保存增强版结果
        ⚠️  这是最关键的函数，必须成功！
        """
        print(f"\n{'='*60}")
        print("💾 Saving HMM core results (CRITICAL STEP)")
        print(f"{'='*60}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存主要结果
        results_file = os.path.join(output_dir, 'enhanced_hmm_results.pkl')
        
        try:
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'cv_results': self.cv_results,
                    'optimal_states': getattr(self, 'optimal_states', None),
                    'best_cv_states': getattr(self, 'best_cv_states', None),
                    'optimal_model': getattr(self, 'optimal_model', None),
                    'models': self.models,
                    'metadata': self.metadata,
                    'mouse_data': self.mouse_data,
                    'embedding': getattr(self, 'embedding', None),
                    'embedding_method': getattr(self, 'embedding_method', None),
                    'cv_splits': getattr(self, 'cv_splits', None),
                    'preprocessing_params': {
                        'scaler': self.scaler,
                        'pca': self.pca,
                        'umap_reducer': getattr(self, 'umap_reducer', None)
                    },
                    'X_sequences': self.X_sequences,
                    'sequence_lengths': self.sequence_lengths
                }, f)
            
            print(f"✅ Core HMM results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to save core results!")
            print(f"   Error: {e}")
            raise  # 重新抛出异常，因为这是致命错误
        
        # 保存CSV格式的摘要
        try:
            summary_data = []
            for k, metrics in self.cv_results.items():
                summary_data.append({
                    'n_states': k,
                    'cv_score_mean': metrics['cv_score_mean'],
                    'cv_score_std': metrics['cv_score_std'],
                    'aic': metrics['aic'],
                    'bic': metrics['bic'],
                    'log_likelihood': metrics['log_likelihood'],
                    'n_params': metrics['n_params'],
                    'converged': metrics['converged'],
                    'n_folds': metrics['n_folds']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(output_dir, 'model_selection_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary CSV saved to: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save summary CSV (non-critical)")
            print(f"   Error: {e}")
        
        return results_file


# 使用示例
def main():
    """
    主函数：增强版HMM模型选择流程
    优化后的执行顺序：
        1. 数据加载和预处理
        2. HMM训练
        3. 立即保存核心结果
        4. 可视化（可选）
    """
    # 设置路径
    features_file = "../data/NR03extracted-LSTM-FullRecord/all_nrem_lstm_features.pkl"
    output_dir = "../data/hmm_results/lstm"
    
    try:
        print(f"\n{'='*60}")
        print("🚀 Starting HMM Model Selection Pipeline")
        print(f"{'='*60}\n")
        
        # ==================== 阶段1: 数据准备 ====================
        print(f"{'='*60}")
        print("📊 STAGE 1: Data Preparation")
        print(f"{'='*60}\n")
        
        hmm_selector = HMMModelSelection(random_state=42)
        features, metadata = hmm_selector.load_features(features_file)
        X_processed = hmm_selector.preprocess_features(use_pca=True, pca_components=32)
        X_sequences, sequence_lengths = hmm_selector.organize_sequences_by_mouse(min_segment_length=5)
        
        # ==================== 阶段2: HMM训练 ====================
        print(f"\n{'='*60}")
        print("🎯 STAGE 2: HMM Training (This may take a while...)")
        print(f"{'='*60}\n")
        
        cv_results = hmm_selector.model_selection_with_cv(
            state_range=(2, 5), 
            n_iter=100, 
            n_folds=10
        )
        
        optimal_k, optimal_model = hmm_selector.find_optimal_states_cv(criterion='bic')
        
        # ==================== 阶段3: 立即保存核心结果！ ====================
        print(f"\n{'='*60}")
        print("💾 STAGE 3: Saving Core Results (PRIORITY)")
        print(f"{'='*60}\n")
        
        #  核心结果保存
        results_file = hmm_selector.save_enhanced_results(output_dir)
        
        print(f"\n{'='*60}")
        print("✅ CORE RESULTS SAVED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📁 Location: {results_file}")
        print(f"🎯 Optimal K (BIC): {optimal_k}")
        if hasattr(hmm_selector, 'best_cv_states'):
            print(f"🎯 Optimal K (CV): {hmm_selector.best_cv_states}")
        print(f"{'='*60}\n")
        
        # ==================== 阶段4: 可视化（可选） ====================
        print(f"\n{'='*60}")
        print("🎨 STAGE 4: Visualization (Optional)")
        print(f"{'='*60}\n")       
        try:
            # 准备可视化数据
            embedding = hmm_selector.prepare_visualization_data(use_umap=True)
            
            # 绘制结果
            hmm_selector.plot_enhanced_results(save_dir=output_dir)
            
        except Exception as viz_error:
            print(f"\n⚠️  Visualization failed, but don't worry!")
            print(f"   Your HMM results are already saved at: {results_file}")
            print(f"   You can retry visualization later using the saved results.")
            print(f"   Error details: {viz_error}")
        
        # ==================== 完成总结 ====================
        print(f"\n{'='*60}")
        print("🎉 HMM Model Selection Pipeline Complete!")
        print(f"{'='*60}")
        print(f"📁 Results directory: {output_dir}")
        print(f"✅ Core HMM models: SAVED")
        print(f"✅ CV results: SAVED")
        print(f"✅ Metadata: SAVED")
        print(f"{'='*60}\n")
        
        return hmm_selector
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Pipeline Error: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return None
    
if __name__ == "__main__":
    hmm_selector = main()