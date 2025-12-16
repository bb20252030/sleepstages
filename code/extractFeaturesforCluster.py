"""
Full-record LSTM feature extraction script
===================================================
Strategy: Reuse the mature pipeline from offline.py to extract LSTM features
over the entire continuous EEG recording, and then select NREM epochs
based on the original sleep-stage labels.

Key advantages:
1. Preserves true temporal context (preceded by Wake and followed by REM epochs)
2. Uses exactly the same sliding-window mechanism as training and inference
3. Leverages the established ClassifierClient + EEGFileReaderServer pipeline
4. Automatically handles LSTM warm-up (the first lstm_length - 1 epochs)

Author: Sun Shibin
Date: 2025-10-21
"""
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import codecs

from classifierClient import ClassifierClient
from eegFileReaderServer import EEGFileReaderServer
from parameterSetup import ParameterSetup


class FullRecordFeatureExtractor:
    """
    全记录特征提取器
    利用offline.py的成熟管道提取LSTM隐藏层特征
    """
    def __init__(self, classifier_id='KHFT5F', sampling_freq=128, epoch_time=4, step_size=4):
        """
        初始化特征提取器
        参数:
            classifier_id: 分类器ID
            sampling_freq: 采样频率(Hz)
            epoch_time: epoch时长(秒)
            step_size: 滑窗步长(秒)
        """
        # 加载参数
        self.params = ParameterSetup()
        self.classifier_id = classifier_id
        self.sampling_freq = sampling_freq
        self.epoch_time = epoch_time
        self.step_size = step_size

        # 获取LSTM序列长度
        self.lstm_length = self.params.torch_lstm_length

        # 特征存储 - 每个bag独立存储
        self.extracted_features = []  # 存储所有提取的特征
        self.feature_metadata = []    # 存储元数据
        self.epoch_counter = 0  # 全局epoch计数器
        
        # Hook是否已注册
        self.hook_registered = False
        self.client = None

        print(f"✓ Full Record Extractor (Simplified) initialized")
        print(f"  - Classifier ID: {classifier_id}")
        print(f"  - Sampling freq: {sampling_freq} Hz")
        print(f"  - Epoch time: {epoch_time} s")
        print(f"  - Step size: {step_size} s")
        print(f"  - Number of bags: 1 (step_size == epoch_time)")

    def setup_feature_extraction_hook(self, client):
        """
        设置特征提取hook
        注意: 必须在client创建后调用
        """
        if self.hook_registered:
            return
        
        self.client = client

        def hook_fn(module, input, output):
            """捕获LSTM后的特征"""
            features = input[0].detach().cpu().numpy().flatten()

            # 存储特征和元数据
            self.extracted_features.append(features)
            self.feature_metadata.append({
                'epoch_index': self.epoch_counter,
                'feature_dim': len(features)})

            self.epoch_counter += 1
        
        # 注册hook到final_fc_lstm的输入
        self.hook = client.stagePredictor.classifier.model.final_fc_lstm.register_forward_hook(hook_fn)
        self.hook_registered = True
        print(f"  ✓ Feature extraction hook registered")
    
    def extract_from_eeg_file(self, eeg_file_path, mouse_id):
        """
        从单个EEG文件提取特征(全记录)
        """
        print(f"\n🐭 Processing {mouse_id} from {eeg_file_path}")
        
        # 重置状态
        self.extracted_features = []
        self.feature_metadata = []
        self.epoch_counter = 0
        self.hook_registered = False
        
        try:
            # 创建ClassifierClient
            # 注意: stepSizeInSec=4 (与epoch_time相同)
            self.client = ClassifierClient(
                recordWaves=False,
                extractorType=self.params.extractorType,
                classifierType=self.params.classifierType,
                classifierID=self.classifier_id,
                inputFileID=mouse_id,
                samplingFreq=self.sampling_freq,
                epochTime=self.epoch_time,
                stepSizeInSec=self.step_size  # 4秒
            )
            
            # 开启预测模式
            self.client.predictionStateOn()
            self.client.hasGUI = False
            
            # 注册特征提取hook
            self.setup_feature_extraction_hook(self.client)
            
            # 处理整个文件
            print(f"  📊 Starting full-record processing...")
            server = EEGFileReaderServer(
                self.client,
                eeg_file_path,
                model_samplingFreq=self.sampling_freq,
                model_epochTime=self.epoch_time,
                observed_samplingFreq=self.sampling_freq,
                observed_epochTime=self.epoch_time
            )
            
            print(f"  ✓ Processing complete")
            print(f"  📦 Extracted {len(self.extracted_features)} features")
            
            # 前lstm_length-1个epoch会被跳过(warmup)
            if len(self.extracted_features) > self.lstm_length - 1:
                print(f"     (First {self.lstm_length-1} epochs skipped for LSTM warmup)")
            
            return self.extracted_features
            
        except Exception as e:
            print(f"  ✗ Error processing {mouse_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        finally:
            # 清理hook
            if hasattr(self, 'hook') and self.hook:
                self.hook.remove()
            self.hook_registered = False
    
    def read_stage_labels(self, label_file_path):
        """
        读取标签文件
        """
        for encoding in ['shift_jis', 'utf-8', None]:
            try:
                if encoding:
                    fp = codecs.open(label_file_path, 'r', encoding)
                else:
                    fp = open(label_file_path, 'r')
                all_lines = fp.readlines()
                fp.close()
                break
            except:
                continue
        
        print(f'  📄 Reading labels from {os.path.basename(label_file_path)}')
        
        # 找到数据开始位置
        data_start_idx = 0
        for i, line in enumerate(all_lines):
            if line.startswith('Time'):
                data_start_idx = i + 1
                break
            elif i >= 20:
                data_start_idx = 0
                break
        
        labels_list = []
        
        for line in all_lines[data_start_idx:]:
            line = line.rstrip()
            if not line or line.replace(',', '').strip() == '':
                continue
            
            elems = [elem.strip() for elem in line.split(',')]
            
            try:
                if len(elems) >= 3:
                    stage_label = elems[2].replace('*', '').upper()
                    
                    # 统一标签格式
                    if stage_label in ['NR', '2']:
                        stage_label = 'S'
                    elif stage_label in ['1','h','L','H']:
                        stage_label = 'W'
                    elif stage_label in ['3']:
                        stage_label = 'R'
                    
                    labels_list.append(stage_label)
            except:
                continue
        
        # 统计
        label_counts = {}
        for label in labels_list:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"     Loaded {len(labels_list)} epochs")
        print(f"     Distribution: {label_counts}")
        
        return labels_list
    
    def process_mouse(self, eeg_file, label_file, mouse_id, output_dir):
        """
        处理单个小鼠的完整流程
        """
        try:
            # 1. 提取全记录特征
            all_features = self.extract_from_eeg_file(eeg_file, mouse_id)
            
            if not all_features:
                print(f"  ✗ No features extracted")
                return False
            
            # 2. 读取标签
            labels_list = self.read_stage_labels(label_file)
            
            # 3. 对齐特征和标签
            # 注意: 前lstm_length-1个epoch没有特征输出
            feature_offset = self.lstm_length - 1
            
            # 创建对齐的数据
            aligned_data = []
            nrem_features = []
            nrem_metadata = []
            
            for i, features in enumerate(all_features):
                # 对应的标签索引
                label_idx = i + feature_offset
                
                if label_idx < len(labels_list):
                    label = labels_list[label_idx]
                    
                    aligned_data.append({
                        'epoch_index': label_idx,
                        'features': features,
                        'label': label
                    })
                    
                    # 筛选NREM
                    if label in ['S', 'N', 'NR']:
                        nrem_features.append(features)
                        nrem_metadata.append({
                            'mouse_id': mouse_id,
                            'epoch_index': label_idx,
                            'label': label
                        })
            
            print(f"  🎯 Aligned {len(aligned_data)} epochs")
            print(f"     NREM epochs: {len(nrem_features)}")
            
            if not nrem_features:
                print(f"  ⚠️ No NREM features after filtering")
                return False
            
            # 4. 保存结果
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存NREM特征
            nrem_file = os.path.join(output_dir, f"{mouse_id}_nrem_lstm_features.pkl")
            with open(nrem_file, 'wb') as f:
                pickle.dump({
                    'mouse_id': mouse_id,
                    'features': np.array(nrem_features),
                    'metadata': nrem_metadata,
                    'n_nrem_epochs': len(nrem_features),
                    'total_epochs': len(all_features),
                    'feature_offset': feature_offset,
                    'lstm_length': self.lstm_length,
                    'extraction_time': datetime.now(),
                    'classifier_id': self.classifier_id
                }, f)
            
            print(f"  💾 Saved NREM features to {nrem_file}")
            
            # 保存完整对齐数据(用于验证)
            aligned_file = os.path.join(output_dir, f"{mouse_id}_aligned_data.pkl")
            with open(aligned_file, 'wb') as f:
                pickle.dump({
                    'mouse_id': mouse_id,
                    'aligned_data': aligned_data,
                    'all_labels': labels_list,
                    'feature_offset': feature_offset,
                    'extraction_time': datetime.now()
                }, f)
            
            print(f"  💾 Saved aligned data to {aligned_file}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to process {mouse_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_multiple_mice(self, file_pairs, output_dir):
        """
        批量处理多个小鼠
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        all_nrem_features = []
        all_nrem_metadata = []
        
        n_mice = len(file_pairs)
        print(f"\n{'='*60}")
        print(f"Processing {n_mice} mice...")
        print(f"{'='*60}\n")
        
        for i, (eeg_path, label_path) in enumerate(file_pairs):
            # 提取mouse_id
            eeg_filename = os.path.basename(eeg_path)
            if '_Trend' in eeg_filename:
                mouse_id = eeg_filename.split('_Trend')[0]
            else:
                mouse_id = eeg_filename.split('.')[0]
            
            print(f"\n[{i+1}/{n_mice}] Processing {mouse_id}...")
            
            success = self.process_mouse(eeg_path, label_path, mouse_id, output_dir)
            
            if success:
                # 读取NREM特征
                nrem_file = os.path.join(output_dir, f"{mouse_id}_nrem_lstm_features.pkl")
                with open(nrem_file, 'rb') as f:
                    mouse_data = pickle.load(f)
                
                results.append({
                    'mouse_id': mouse_id,
                    'n_nrem_epochs': mouse_data['n_nrem_epochs'],
                    'total_epochs': mouse_data['total_epochs'],
                    'nrem_file': nrem_file
                })
                
                # 合并到总集合
                all_nrem_features.extend(mouse_data['features'])
                all_nrem_metadata.extend(mouse_data['metadata'])
        
        # 保存汇总文件(用于HMM)
        if all_nrem_features:
            summary_file = os.path.join(output_dir, "all_nrem_lstm_features.pkl")
            with open(summary_file, 'wb') as f:
                pickle.dump({
                    'features': np.array(all_nrem_features),
                    'metadata': all_nrem_metadata,
                    'n_mice': len(results),
                    'total_nrem_epochs': len(all_nrem_features),
                    'lstm_length': self.lstm_length,
                    'extraction_time': datetime.now(),
                    'mice_info': results
                }, f)
            
            print(f"\n{'='*60}")
            print(f"🎉 Feature extraction complete!")
            print(f"{'='*60}")
            print(f"✓ Processed: {len(results)}/{n_mice} mice")
            print(f"📊 Total NREM epochs: {len(all_nrem_features)}")
            print(f"📁 Summary file: {summary_file}")
            print(f"{'='*60}\n")

def auto_discover_files(eeg_dir, label_dir):
    """
    自动发现匹配的EEG和标签文件对
    
    返回:
        file_pairs: [(eeg_path, label_path), ...]
    """
    file_pairs = []
    
    # 获取所有标签文件
    label_files = {}
    for f in os.listdir(label_dir):
        if f.endswith('_Trend.csv') or f.endswith('.csv'):
            base = f.replace('_Trend.csv', '').replace('.csv', '')
            label_files[base] = os.path.join(label_dir, f)
    
    # 匹配EEG文件
    for f in os.listdir(eeg_dir):
        if f.endswith('.csv') and not f.endswith('_Trend.csv'):
            base = f.replace('.csv', '')
            if base in label_files:
                eeg_path = os.path.join(eeg_dir, f)
                label_path = label_files[base]
                file_pairs.append((eeg_path, label_path))
    
    print(f"Found {len(file_pairs)} matching file pairs")
    return file_pairs   


def main():
    """主函数"""
    # 设置路径
    eeg_dir = "../data/aipost"
    label_dir = "../data/labels"
    output_dir = "../data/NR03extracted-LSTM-FullRecord"
    
    # 自动发现文件对
    file_pairs = auto_discover_files(eeg_dir, label_dir)
    
    if not file_pairs:
        print("❌ No matching file pairs found!")
        return
    
    # 创建提取器
    extractor = FullRecordFeatureExtractor(
        classifier_id='KHFT5F',
        sampling_freq=128,
        epoch_time=4,
        step_size=4
    )
    
    # 批量处理
    extractor.process_multiple_mice(file_pairs, output_dir)
    
    print("\n✨ All done! Features ready for HMM clustering.")


if __name__ == "__main__":
    main()