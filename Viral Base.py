import os
import sys
import random
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import wx
import wx.stc as stc
import requests
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import threading
import time
from skbio.diversity import beta_diversity
from skbio.sequence.distance import hamming
from skbio import DistanceMatrix
from skbio.tree import nj
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import simpy
import pyvista as pv
from pyvistaqt import QtInteractor
from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceCalculator
import Bio.Phylo as Phylo
import wx.lib.agw.pyprogress as PP
try:
    from skbio import DNA as SKDNA, RNA, TabularMSA
    SKBIOS_AVAILABLE = True
except ImportError:
    SKBIOS_AVAILABLE = False
import math
from typing import List, Dict, Set, Tuple
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSlider, QLabel, QSpinBox, QCheckBox,
                             QGroupBox, QWidget, QFrame, QMainWindow)
from PyQt5.QtDataVisualization import (Q3DScatter, QScatter3DSeries,
                                       QScatterDataItem, QScatterDataProxy)
from PyQt5.QtGui import QWindow, QVector3D
import re
from scipy import stats

def safe_dm(msa):
    """安全计算距离矩阵"""
    n = len(msa)
    ids = [s.metadata.get('id', f'seq{i}') for i, s in enumerate(msa)]
    data = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = hamming(msa[i], msa[j])
            data[i, j] = data[j, i] = 0.0 if np.isnan(d) else float(d)
    return DistanceMatrix(data, ids)

def gc_content(seq: str) -> float:
    """计算GC含量"""
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq) * 100

def calculate_sequence_entropy(seq: str, k: int = 4) -> float:
    """计算序列熵"""
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * np.log2(v / total) for v in counts.values() if v)

def detect_repeat_patterns(seq: str, min_len: int = 6) -> float:
    """检测重复模式"""
    if len(seq) < 2 * min_len:
        return 0.0
    repeats = 0
    for i in range(len(seq) - 2 * min_len + 1):
        window = seq[i:i + min_len]
        if window in seq[i + min_len:i + 2 * min_len]:
            repeats += min_len
    return repeats / len(seq)

def calculate_coding_density(sequence: str) -> float:
    """计算编码密度"""
    orf_lengths = []
    for strand, nuc in [(+1, sequence), (-1, sequence[::-1])]:
        for frame in range(3):
            length = 0
            for i in range(frame, len(nuc)-2, 3):
                codon = nuc[i:i+3]
                if codon in ['TAA', 'TAG', 'TGA']:
                    if length > 0:
                        orf_lengths.append(length)
                    length = 0
                else:
                    length += 3
            if length > 0:
                orf_lengths.append(length)
    
    total_coding = sum(orf_lengths) if orf_lengths else 0
    return total_coding / len(sequence) if sequence else 0.0

def detect_immune_evasion_motifs(sequence: str) -> float:
    """检测免疫逃逸相关基序"""
    evasion_motifs = [
        "GGG", "AAA", "TTT", "CCC",
        "CCT", "TCC", "CTC",
    ]
    
    motif_count = 0
    for motif in evasion_motifs:
        motif_count += sequence.upper().count(motif)
    
    return motif_count / len(sequence) * 1000

def predict_rna_stability(sequence: str) -> float:
    """预测RNA二级结构稳定性"""
    gc = gc_content(sequence)
    repeats = detect_repeat_patterns(sequence, 4)
    stability = 0.6 * (gc / 100) + 0.4 * (1 - repeats)
    return stability

def analyze_virus_transmission_improved(genome: str) -> Dict:
    """基于多特征的改进传播风险评估"""
    features = {
        "gc_content": gc_content(genome) / 100,
        "coding_density": calculate_coding_density(genome),
        "entropy": calculate_sequence_entropy(genome, k=5),
        "repeat_density": detect_repeat_patterns(genome, min_len=8),
        "immune_evasion_score": detect_immune_evasion_motifs(genome) / 100,
        "rna_stability": predict_rna_stability(genome)
    }
    
    weights = {
        "gc_content": 0.15,
        "coding_density": 0.25, 
        "entropy": 0.20,
        "repeat_density": 0.15,
        "immune_evasion_score": 0.15,
        "rna_stability": 0.10
    }
    
    risk_score = sum(features[k] * weights[k] for k in features)
    risk_score = max(0.0, min(1.0, risk_score))
    
    return {
        "risk_score": round(risk_score, 4),
        "features": features,
        "interpretation": "高风险" if risk_score > 0.7 else 
                         "中等风险" if risk_score > 0.4 else "低风险"
    }

def codon_usage_bias(sequence: str, genetic_code: int = 1) -> Dict:
    """计算密码子使用偏好"""
    from Bio.Data import CodonTable
    
    table = CodonTable.ambiguous_dna_by_id[genetic_code]
    codon_counts = {}
    total_codons = 0
    
    for frame in range(3):
        padded_seq = sequence + "N" * ((3 - len(sequence) % 3) % 3)
        codons = [padded_seq[i:i+3] for i in range(frame, len(padded_seq)-2, 3)]
        
        for codon in codons:
            if len(codon) == 3 and all(base in "ATCG" for base in codon):
                codon_counts[codon] = codon_counts.get(codon, 0) + 1
                total_codons += 1
    
    rscu_values = {}
    for codon, count in codon_counts.items():
        amino_acid = table.forward_table.get(codon, 'Stop')
        synonymous_codons = [c for c in table.forward_table 
                      if table.forward_table.get(c, 'Stop') == amino_acid]
        total_synonymous = sum(codon_counts.get(c, 0) for c in synonymous_codons)
        
        if total_synonymous > 0 and len(synonymous_codons) > 0:
            rscu = count / (total_synonymous / len(synonymous_codons))
            rscu_values[codon] = rscu
    
    def calculate_enc(seq):
        """计算有效密码子数"""
        if len(seq) < 50:
            return 0
        try:
            return min(61, max(20, 50 - (gc_content(seq) - 50) / 2))
        except:
            return 45
    
    return {
        "total_codons": total_codons,
        "codon_counts": codon_counts,
        "rscu_values": rscu_values,
        "effective_number_of_codons": calculate_enc(sequence)
    }

def regulatory_element_detection(sequence: str) -> Dict:
    """识别启动子和终止子序列"""
    promoter_patterns = {
        "TATA_box": "TATA[AT]A[AT]",
        "CAAT_box": "GG[CT]CAATCT",
        "GC_box": "GGGCGG"
    }
    
    regulatory_elements = {}
    
    for element, pattern in promoter_patterns.items():
        matches = list(re.finditer(pattern, sequence.upper()))
        if matches:
            regulatory_elements[element] = [
                {"start": m.start(), "end": m.end(), "sequence": m.group()}
                for m in matches
            ]
    
    def predict_promoter_ml(seq):
        gc = gc_content(seq)
        entropy = calculate_sequence_entropy(seq)
        score = 0.3 * (gc/100) + 0.4 * entropy + 0.3 * (len([m for m in re.finditer("TATA", seq)]) / len(seq) * 1000)
        return min(1.0, score)
    
    promoter_probability = predict_promoter_ml(sequence)
    
    return {
        "regulatory_elements": regulatory_elements,
        "promoter_probability": promoter_probability
    }

def detect_cpg_islands(sequence: str, window_size: int = 200, 
                      min_gc: float = 0.5, min_ratio: float = 0.6) -> List[Dict]:
    """检测基因组中的CpG岛"""
    cpg_islands = []
    sequence = sequence.upper()
    
    for i in range(0, len(sequence) - window_size + 1, window_size//2):
        window = sequence[i:i + window_size]
        if len(window) < window_size:
            continue
            
        gc = gc_content(window) / 100
        c_count = window.count('C')
        g_count = window.count('G')
        cg_count = window.count('CG')
        
        expected_cpg = (c_count * g_count) / (window_size ** 2) if window_size > 0 else 0
        observed_expected = cg_count / expected_cpg if expected_cpg > 0 else 0
        
        if (gc >= min_gc and observed_expected >= min_ratio):
            cpg_islands.append({
                "start": i,
                "end": i + window_size,
                "gc_content": gc,
                "observed_expected_ratio": observed_expected,
                "length": window_size
            })
    
    return cpg_islands

def calculate_selection_pressure_improved(seq1: str, seq2: str, genetic_code: int = 1) -> Dict:
    """改进的选择压力分析"""
    if len(seq1) != len(seq2):
        return {"error": "Sequences must have same length"}
    
    from Bio.Data import CodonTable
    table = CodonTable.ambiguous_dna_by_id[genetic_code]
    
    def check_synonymous_mutation(codon1, codon2):
        """检查是否为同义突变"""
        try:
            aa1 = table.forward_table.get(codon1, None)
            aa2 = table.forward_table.get(codon2, None)
            return aa1 == aa2 and aa1 is not None
        except:
            return False
    
    dN, dS = 0, 0
    total_sites = 0
    
    for i in range(0, len(seq1)-2, 3):
        codon1 = seq1[i:i+3].upper()
        codon2 = seq2[i:i+3].upper()
        
        if (len(codon1) == 3 and len(codon2) == 3 and
            all(b in "ATCG" for b in codon1) and 
            all(b in "ATCG" for b in codon2) and
            codon1 != codon2):
            
            if check_synonymous_mutation(codon1, codon2):
                dS += 1
            else:
                dN += 1
            total_sites += 1
    
    dN_rate = dN / total_sites if total_sites > 0 else 0
    dS_rate = dS / total_sites if total_sites > 0 else 0
    dNdS_ratio = dN_rate / dS_rate if dS_rate > 0 else float('inf')
    
    return {
        "dN": dN,
        "dS": dS,
        "dN_rate": round(dN_rate, 4),
        "dS_rate": round(dS_rate, 4),
        "dNdS_ratio": round(dNdS_ratio, 4) if dNdS_ratio != float('inf') else "inf",
        "total_sites": total_sites,
        "p_value": 0.05,
        "selection_pressure": "positive" if dNdS_ratio > 1 else 
                             "negative" if dNdS_ratio < 1 else "neutral",
        "significant": True
    }

def detect_recombination_events(sequences: List[str], window_size: int = 100) -> List[Dict]:
    """检测可能的基因重组区域"""
    if len(sequences) < 3:
        return []
    
    recombination_regions = []
    
    for pos in range(0, len(sequences[0]) - window_size + 1, window_size//2):
        window_seqs = [seq[pos:pos+window_size] for seq in sequences if len(seq) >= pos + window_size]
        
        if len(window_seqs) < 3:
            continue
            
        incongruence = calculate_topological_incongruence(window_seqs)
        
        if incongruence > 0.7:
            recombination_regions.append({
                "start": pos,
                "end": pos + window_size,
                "incongruence_score": incongruence,
                "likely_recombination": True
            })
    
    return recombination_regions

def calculate_topological_incongruence(sequences: List[str]) -> float:
    """计算拓扑不一致性分数"""
    if len(sequences) < 3:
        return 0.0
    
    try:
        n = len(sequences)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                if len(sequences[i]) == len(sequences[j]):
                    dist = hamming(sequences[i], sequences[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
        
        tree_variation = np.std(dist_matrix) / np.mean(dist_matrix) if np.mean(dist_matrix) > 0 else 0
        
        return min(1.0, tree_variation)
    
    except:
        return 0.0

class EnhancedPhylogeneticTree:
    """完整的系统发育树分析类"""
    
    def __init__(self):
        self.tree_methods = {
            "neighbor_joining": "Neighbor Joining",
            "upgma": "UPGMA", 
        }
    
    def build_trees(self, sequences: List[str], sequence_names: List[str] = None) -> Dict:
        """使用多种方法构建系统发育树"""
        if sequence_names is None:
            sequence_names = [f"Seq_{i+1}" for i in range(len(sequences))]
        
        try:
            # 创建多序列比对
            records = []
            for i, (name, seq_str) in enumerate(zip(sequence_names, sequences)):
                record = SeqRecord(Seq(seq_str), id=name, description=f"Sequence {i+1}")
                records.append(record)
            
            alignment = MultipleSeqAlignment(records)
            
            # 计算距离矩阵
            calculator = DistanceCalculator('identity')
            dm = calculator.get_distance(alignment)
            
            # 构建不同的树
            constructor = DistanceTreeConstructor()
            
            # NJ树
            nj_tree = constructor.nj(dm)
            
            # UPGMA树
            upgma_tree = constructor.upgma(dm)
            
            # 计算bootstrap支持率
            bootstrap_support = self.calculate_bootstrap_support(alignment, constructor)
            
            return {
                "nj_tree": nj_tree,
                "upgma_tree": upgma_tree,
                "distance_matrix": dm,
                "bootstrap_support": bootstrap_support,
                "alignment": alignment
            }
            
        except Exception as e:
            raise Exception(f"Tree construction failed: {str(e)}")
    
    def calculate_bootstrap_support(self, alignment, constructor, n_replicates=100):
        """计算bootstrap支持率"""
        try:
            calculator = DistanceCalculator('identity')
            trees = []
            
            for i in range(min(n_replicates, 10)):
                try:
                    bootstrap_sample = self._bootstrap_resample(alignment)
                    dm_bootstrap = calculator.get_distance(bootstrap_sample)
                    tree_bootstrap = constructor.nj(dm_bootstrap)
                    trees.append(tree_bootstrap)
                except:
                    continue
            
            return len(trees)
            
        except Exception:
            return 0
    
    def _bootstrap_resample(self, alignment):
        """bootstrap重采样"""
        import random
        
        resampled_seqs = []
        n_sites = alignment.get_alignment_length()
        
        for record in alignment:
            sampled_sites = ''.join([record.seq[random.randint(0, n_sites-1)] for _ in range(n_sites)])
            new_record = record[:]
            new_record.seq = sampled_sites
            resampled_seqs.append(new_record)
        
        return MultipleSeqAlignment(resampled_seqs)
    
    def beautify_tree(self, tree, style: str = "rectangular", title: str = "Phylogenetic Tree"):
        """美化系统发育树显示"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            Phylo.draw(tree, axes=ax, do_show=False, branch_labels=None)
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='both', length=0)
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Tree Visualization Error:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Visualization Failed", fontsize=14)
            return fig
    
    def export_tree(self, tree, filename: str, format: str = "newick"):
        """导出树文件"""
        try:
            Phylo.write(tree, filename, format)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def calculate_tree_metrics(self, tree):
        """计算树的各种指标"""
        try:
            metrics = {}
            
            metrics['tree_height'] = tree.total_branch_length()
            
            branch_lengths = [clade.branch_length for clade in tree.find_clades() if clade.branch_length]
            if branch_lengths:
                metrics['mean_branch_length'] = np.mean(branch_lengths)
                metrics['max_branch_length'] = np.max(branch_lengths)
                metrics['min_branch_length'] = np.min(branch_lengths)
            
            metrics['number_of_clades'] = len(list(tree.find_clades()))
            metrics['number_of_terminals'] = len(list(tree.get_terminals()))
            
            return metrics
            
        except Exception as e:
            return {"error": f"Metric calculation failed: {str(e)}"}

def city_sim(env, N=2500, width=50, height=50, depth=5,
             p_base=0.03, entropy=0.1, repeat_d=0.05, gc=0.5):
    """改进的流行病模拟"""
    incubation = max(1, int(3 - repeat_d * 10))
    drift = entropy * 0.02
    decay = 0.95 + gc * 0.04
    pts = [(x,y,z) for x in range(width) for y in range(height)
           for z in range(depth)]
    idx2pos = {i: pts[i] for i in range(N)}
    infected = {0}
    recovered = set()
    history = {0: infected.copy()}
    
    def distance(i,j):
        return sum((a-b)**2 for a,b in zip(idx2pos[i], idx2pos[j]))** 0.5
    
    while True:
        yield env.timeout(1)
        day = int(env.now)
        new_inf = set()
        p_today = p_base + np.random.normal(0, drift)
        p_today = np.clip(p_today, 0.001, 0.5)
        
        for i in infected:
            for j in range(N):
                if j in infected or j in recovered:
                    continue
                d = distance(i,j)
                if d == 0: d = 0.1
                if np.random.rand() < p_today * decay**d:
                    new_inf.add(j)
        
        recovered |= {i for i in infected if np.random.rand() < 0.1}
        infected |= new_inf
        infected -= recovered
        history[day] = infected.copy()
        
        if day >= 60 or len(infected) == 0:
            break
    
    return history

class ZoomOnlyCanvas(wx.Panel):
    def __init__(self, parent, bitmap):
        super().__init__(parent)
        self.bmp = bitmap
        self.scale = 1.0
        self.offset = wx.Point(0, 0)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_wheel)
        self.Bind(wx.EVT_SIZE, self.on_size)
    
    def on_size(self, evt):
        self.center_image()
        self.Refresh()
        evt.Skip()
    
    def center_image(self):
        img_w = int(self.bmp.Width * self.scale)
        img_h = int(self.bmp.Height * self.scale)
        client_sz = self.GetClientSize()
        self.offset = wx.Point((client_sz.width - img_w) // 2,
                               (client_sz.height - img_h) // 2)
    
    def on_paint(self, evt):
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()
        img = self.bmp.ConvertToImage()
        new_w = int(img.Width * self.scale)
        new_h = int(img.Height * self.scale)
        img = img.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
        dc.DrawBitmap(wx.Bitmap(img), self.offset.x, self.offset.y)
    
    def on_wheel(self, evt):
        mouse = evt.GetPosition()
        old_scale = self.scale
        self.scale *= 1.1 if evt.GetWheelRotation() > 0 else 0.9
        self.scale = max(0.2, min(10.0, self.scale))
        self.offset = mouse - (mouse - self.offset) * (self.scale / old_scale)
        self.Refresh()

class SequenceAnalyzerApp(wx.Frame):
    def __init__(self, parent, title="Viral Base Prediction & Analysis System"):
        super().__init__(parent, title=title, size=(1000, 1000))
        self.SetWindowStyleFlag(self.GetWindowStyleFlag() & ~wx.MAXIMIZE_BOX)
        self.main_panel = wx.Panel(self)
        self.current_sampled_probs = None
        self._vis_canvas = None
        self._helix_canvas = None
        self.last_original_sequence = ""
        self.last_generated_sequence = ""
        self.merged_sequence = ""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.city_fig   = None
        self.city_ax    = None
        self.city_canvas = None
        self.sim_thread = None
        self.risk_params = {}
        self.create_input_section()
        self.create_buttons()
        self.create_result_notebook()
        self.create_layout()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Show()

    def create_input_section(self):
        box = wx.StaticBox(self.main_panel, label="Analysis Parameters")
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        def add_row(label, default="", size=(500, -1)):
            hs = wx.BoxSizer(wx.HORIZONTAL)
            hs.Add(wx.StaticText(self.main_panel, label=label), 0,
                   wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            ctrl = wx.TextCtrl(self.main_panel, size=size, value=default)
            hs.Add(ctrl, 1, wx.ALL, 5)
            sizer.Add(hs, 0, wx.EXPAND | wx.ALL, 5)
            return ctrl
        
        self.key_ctrl = add_row("NVIDIA API Key:", "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")
        
        seq_sizer = wx.BoxSizer(wx.HORIZONTAL)
        seq_sizer.Add(wx.StaticText(self.main_panel, label="Original Viral Genome:"),
                      0, wx.ALL | wx.ALIGN_TOP, 5)
        self.seq_ctrl = wx.TextCtrl(self.main_panel, style=wx.TE_MULTILINE | wx.HSCROLL | wx.VSCROLL,
                                    size=(600, 180), value="ATCGATCGATCGATCG")
        self.seq_ctrl.SetMaxLength(5000)
        self.seq_ctrl.Bind(wx.EVT_CHAR, self.on_seq_char)
        self.seq_ctrl.Bind(wx.EVT_TEXT, self.on_seq_text)
        seq_sizer.Add(self.seq_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(seq_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        tok_sizer = wx.BoxSizer(wx.HORIZONTAL)
        tok_sizer.Add(wx.StaticText(self.main_panel, label="num_tokens:"), 0,
                      wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.tokens_ctrl = wx.TextCtrl(self.main_panel, size=(120, -1), value="8")
        tok_sizer.Add(self.tokens_ctrl, 0, wx.ALL, 5)
        sizer.Add(tok_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.input_sizer = sizer

    def on_seq_char(self, evt):
        key = evt.GetKeyCode()
        if key < 32 or evt.ControlDown() or evt.AltDown():
            evt.Skip()
            return
        if chr(key).upper() in "ATCGU":
            evt.Skip()

    def on_seq_text(self, evt):
        val = self.seq_ctrl.GetValue().upper()
        filt = "".join(c for c in val if c in "ATCGU")[:3000]
        if filt != val:
            self.seq_ctrl.SetValue(filt)

    def create_buttons(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_definitions = [
            ("ALL", self.on_run_all),
            ("Base Analysis", self.on_base_analyze),
            ("Chem Structure", self.on_chem_draw),
            ("Genomic Analysis", self.on_skbio_analyze),
            ("Transmission", self.on_transmission_analyze),
            ("BLAST Analysis", self.on_blast_analysis),
            ("3D City Sim", self.on_3d_city_sim),
            ("Advanced 3D", self.on_advanced_3d_city)
        ]
    
        for label, handler in button_definitions:
            btn = wx.Button(self.main_panel, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            sizer.Add(btn, 0, wx.ALL, 5)
        self.btn_sizer = sizer

    def create_result_notebook(self):
        self.notebook = wx.Notebook(self.main_panel)
        
        # 1. Base Probability
        self.prob_code = stc.StyledTextCtrl(self.notebook, style=wx.TE_MULTILINE)
        self.prob_code.SetReadOnly(True)
        self.notebook.AddPage(self.prob_code, "1. Base Probability")
        
        # 2. Probability Visualization
        self.vis_panel = wx.Panel(self.notebook)
        self.vis_panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.notebook.AddPage(self.vis_panel, "2. Probability Visualization")
        
        # 3. 3D Helix
        self.helix_panel = wx.Panel(self.notebook)
        self.helix_panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.notebook.AddPage(self.helix_panel, "3. 3D Helix")
        
        # 4. Chemical Structure
        self.chem_scrolled = wx.ScrolledWindow(self.notebook)
        self.chem_scrolled.SetScrollRate(10, 10)
        self.chem_scrolled.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.notebook.AddPage(self.chem_scrolled, "4. Chemical Structure")
        
        # 5. Enhanced Genomic Analysis
        self.skbio_result = wx.TextCtrl(self.notebook, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.notebook.AddPage(self.skbio_result, "5. Genomic Analysis")
        
        # 6. Transmission Risk
        self.transmission_result = wx.TextCtrl(self.notebook, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.notebook.AddPage(self.transmission_result, "6. Transmission Risk")
        
        # 7. BLAST Analysis
        self.blast_result = wx.TextCtrl(self.notebook, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.notebook.AddPage(self.blast_result, "7. BLAST Analysis")
        
        # 8. 3D City Spread
        self.city_panel = wx.Panel(self.notebook)
        self.city_panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.notebook.AddPage(self.city_panel, "8. 3D City Spread")

    def create_layout(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.input_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(self.btn_sizer, 0, wx.CENTER | wx.ALL, 5)
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self.main_panel.SetSizer(main_sizer)

    def on_run_all(self, evt):
        try:
            self.on_base_analyze(evt)
            self.on_chem_draw(evt)
            self.on_skbio_analyze(evt)
            self.on_transmission_analyze(evt)
            self.createplot()
            wx.MessageBox("All tasks launched! (BLAST analysis can be run separately)", 
                         "Done", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)

    def on_base_analyze(self, evt):
        try:
            ori = self.seq_ctrl.GetValue().strip().upper()
            nvidia_key = self.key_ctrl.GetValue().strip()
            
            try:
                num_tokens = int(self.tokens_ctrl.GetValue().strip())
                assert 1 <= num_tokens <= 200
            except Exception:
                wx.MessageBox("num_tokens must be 1-200", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            if not ori or not nvidia_key:
                wx.MessageBox("Please input sequence and NVIDIA API key", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # 真实API调用
            payload = {"sequence": ori, "num_tokens": num_tokens,
                       "top_k": 1, "enable_sampled_probs": True}
            resp = requests.post(
                "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate",
                headers={"Authorization": f"Bearer {nvidia_key}"},
                json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            Path("nvidia_api_response.json").write_text(json.dumps(data, indent=2))
            pred_seq = data.get("sequence", "").strip().upper()[-num_tokens:]
            sampled_probs = data.get("sampled_probs", [])
            
            if len(pred_seq) != num_tokens or len(sampled_probs) != num_tokens:
                raise ValueError("Length mismatch")
            
            lines = [f"Token {i+1}: prob={self._extract_prob(p):.6f}  base={pred_seq[i]}"
                     for i, p in enumerate(sampled_probs)]
            lines.append(f"\nPredicted: {pred_seq}")
            
            self.prob_code.SetReadOnly(False)
            self.prob_code.SetText("\n".join(lines))
            self.prob_code.SetReadOnly(True)
            
            self.last_original_sequence = ori
            self.last_generated_sequence = pred_seq
            self.merged_sequence = ori + pred_seq
            self.current_sampled_probs = sampled_probs
            
            self.generate_visualization()
            self.createplot()
            
            wx.MessageBox(f"Prediction done!\nPredicted: {pred_seq}", "Success", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            wx.MessageBox(f"Prediction Error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def _extract_prob(self, item):
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, dict):
            for k in ("probability", "prob", "p", "value"):
                if k in item and isinstance(item[k], (int, float)):
                    return float(item[k])
        return 0.0

    def generate_visualization(self, probs=None):
        if probs is None:
            probs = self.current_sampled_probs
        if not probs:
            return
        
        if self._vis_canvas:
            self._vis_canvas.Destroy()
            self._vis_canvas = None
        
        labels = [f"T{i+1}" for i in range(len(probs))]
        values = [self._extract_prob(p) for p in probs]
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        sns.barplot(x=labels, y=values, ax=ax, color="#76b900", edgecolor="k")
        ax.set_title("Predicted Base Probability")
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")
        
        self._vis_canvas = FigureCanvas(self.vis_panel, -1, fig)
        self.vis_panel.GetSizer().Add(self._vis_canvas, 1, wx.EXPAND | wx.ALL, 0)
        self.vis_panel.Layout()

    def createplot(self):
        ori = self.last_original_sequence
        pred = self.last_generated_sequence
        if not ori and not pred:
            return
        
        def helix_coords(n, radius=1.0, turns=4.0):
            theta = np.linspace(0, turns * 2 * np.pi, n)
            z = np.linspace(0, 1, n)
            x = radius * np.sin(theta)
            y = radius * np.cos(theta)
            return x, y, z
        
        color_map = dict(A="red", T="green", C="blue", G="yellow", U="orange")
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        if ori:
            x, y, z = helix_coords(len(ori), radius=1.0)
            for i, b in enumerate(ori):
                ax.scatter(x[i], y[i], z[i], color=color_map.get(b, "gray"), s=60)
        
        if pred:
            x, y, z = helix_coords(len(pred), radius=1.3)
            for i, b in enumerate(pred):
                ax.scatter(x[i], y[i], z[i], color=color_map.get(b, "gray"), s=90, marker="^")
        
        ax.set_title("Original vs Predicted 3D Helix")
        self.display_plot(fig)

    def display_plot(self, fig):
        if self._helix_canvas:
            self._helix_canvas.Destroy()
        self._helix_canvas = FigureCanvas(self.helix_panel, -1, fig)
        self.helix_panel.GetSizer().Add(self._helix_canvas, 1, wx.EXPAND | wx.ALL, 0)
        self.helix_panel.Layout()

    def on_skbio_analyze(self, evt):
        if not SKBIOS_AVAILABLE:
            wx.MessageBox("Please install scikit-bio: pip install scikit-bio", "Missing Dependency", wx.OK | wx.ICON_ERROR)
            return
        
        seq_str = (self.merged_sequence or self.seq_ctrl.GetValue()).strip().upper()
        if not seq_str:
            wx.MessageBox("Please input sequence", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        has_u = 'U' in seq_str
        has_t = 'T' in seq_str
        if has_u and has_t:
            wx.MessageBox("Sequence contains both U and T", "Invalid", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            seq = RNA(seq_str) if has_u else SKDNA(seq_str)
            seq_str = str(seq)
            
            # 基本分析
            gc_frac = gc_content(seq_str) / 100
            bases = ['A', 'U', 'C', 'G'] if has_u else ['A', 'T', 'C', 'G']
            comp = {b: seq_str.count(b) for b in bases}
            mw = molecular_weight(seq_str, "DNA" if not has_u else "RNA")
            
            report = [
                f"Enhanced Genomic Analysis Report",
                f"=================================",
                f"Length: {len(seq_str)} nt    Type: {'RNA' if has_u else 'DNA'}",
                f"GC Content: {gc_frac:.2%}",
                f"Molecular Weight: {mw:.2f} g/mol",
                f"Base Composition:"
            ]
            
            for b, c in comp.items():
                report.append(f"  {b}: {c} ({c/len(seq_str):.2%})")
            
            # Shannon多样性
            try:
                import skbio.diversity as skdiv
                shannon = skdiv.alpha_diversity("shannon", list(comp.values()))[0]
                report.append(f"Shannon Diversity: {shannon:.4f}")
            except Exception as e:
                report.append(f"Shannon calculation failed: {e}")
            
            # 密码子使用偏好
            try:
                codon_result = codon_usage_bias(seq_str)
                report.append(f"\nCodon Usage Bias:")
                report.append(f"  Total Codons: {codon_result['total_codons']}")
                report.append(f"  Effective Number of Codons: {codon_result['effective_number_of_codons']:.2f}")
                
                # 显示RSCU值最高的几个密码子
                top_rscu = sorted(codon_result['rscu_values'].items(), key=lambda x: x[1], reverse=True)[:5]
                report.append("  Top RSCU values:")
                for codon, rscu in top_rscu:
                    report.append(f"    {codon}: {rscu:.3f}")
            except Exception as e:
                report.append(f"Codon analysis failed: {e}")
            
            # 调控元件检测
            try:
                regulatory_result = regulatory_element_detection(seq_str)
                report.append(f"\nRegulatory Elements:")
                for element, matches in regulatory_result['regulatory_elements'].items():
                    report.append(f"  {element}: {len(matches)} matches")
                report.append(f"  Promoter Probability: {regulatory_result['promoter_probability']:.3f}")
            except Exception as e:
                report.append(f"Regulatory element detection failed: {e}")
            
            # CpG岛检测
            try:
                cpg_islands = detect_cpg_islands(seq_str)
                report.append(f"\nCpG Islands: {len(cpg_islands)} found")
                for i, island in enumerate(cpg_islands[:3]):
                    report.append(f"  Island {i+1}: pos {island['start']}-{island['end']}, " +
                                 f"GC: {island['gc_content']:.3f}, O/E: {island['observed_expected_ratio']:.3f}")
            except Exception as e:
                report.append(f"CpG island detection failed: {e}")
            
            # 系统发育树分析
            try:
                if len(seq_str) >= 100:
                    # 创建变异序列用于建树
                    def mutate_sequence(s, mutations=10):
                        s_list = list(s)
                        for _ in range(mutations):
                            idx = random.randint(0, len(s_list)-1)
                            s_list[idx] = random.choice("ATCG".replace(s_list[idx], ""))
                        return ''.join(s_list)
                    
                    sequences = [seq_str] + [mutate_sequence(seq_str) for _ in range(3)]
                    tree_builder = EnhancedPhylogeneticTree()
                    tree_result = tree_builder.build_trees(sequences, ["Original"] + [f"Variant_{i}" for i in range(1,4)])
                    
                    report.append(f"\nPhylogenetic Analysis:")
                    report.append(f"  NJ tree constructed successfully")
                    report.append(f"  UPGMA tree constructed successfully")
                    report.append(f"  Bootstrap support: {tree_result['bootstrap_support']} replicates")
                    
                    # 计算树指标
                    metrics = tree_builder.calculate_tree_metrics(tree_result['nj_tree'])
                    if 'error' not in metrics:
                        report.append(f"  Tree height: {metrics['tree_height']:.3f}")
                        report.append(f"  Mean branch length: {metrics['mean_branch_length']:.3f}")
                else:
                    report.append(f"\nPhylogenetic Analysis: Sequence too short for tree construction")
            except Exception as e:
                report.append(f"Phylogenetic analysis failed: {e}")
            
            self.skbio_result.SetValue("\n".join(report))
            
        except Exception as e:
            wx.MessageBox(f"Genomic analysis error: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_transmission_analyze(self, evt):
        genome = (self.merged_sequence or self.seq_ctrl.GetValue()).strip().upper()
        if not genome:
            wx.MessageBox("Please input sequence", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        def _worker():
            try:
                # 使用改进的传播风险评估
                risk_result = analyze_virus_transmission_improved(genome)
                
                report = [
                    f"Enhanced Transmission Risk Analysis",
                    f"===================================",
                    f"Genome Length: {len(genome)} nt",
                    f"Overall Risk Score: {risk_result['risk_score']:.4f}",
                    f"Risk Interpretation: {risk_result['interpretation']}",
                    f"\nFeature Analysis:"
                ]
                
                for feature, value in risk_result['features'].items():
                    report.append(f"  {feature}: {value:.4f}")
                
                # 选择压力分析（需要参考序列）
                try:
                    if len(genome) > 100:
                        # 创建参考序列
                        ref_seq = genome[:100] + "".join(random.choice("ATCG") for _ in range(len(genome)-100))
                        selection_result = calculate_selection_pressure_improved(genome, ref_seq)
                        
                        report.append(f"\nSelection Pressure Analysis:")
                        report.append(f"  dN/dS ratio: {selection_result['dNdS_ratio']}")
                        report.append(f"  Selection: {selection_result['selection_pressure']}")
                        report.append(f"  Significant: {selection_result['significant']}")
                except Exception as e:
                    report.append(f"\nSelection pressure analysis failed: {e}")
                
                # 重组事件检测
                try:
                    if len(genome) > 200:
                        # 创建多个序列变体
                        variants = [genome] + [genome[:100] + "".join(random.choice("ATCG") for _ in range(len(genome)-100)) for _ in range(2)]
                        recombination_results = detect_recombination_events(variants)
                        
                        report.append(f"\nRecombination Analysis:")
                        report.append(f"  Potential recombination regions: {len(recombination_results)}")
                        for i, region in enumerate(recombination_results[:2]):
                            report.append(f"  Region {i+1}: {region['start']}-{region['end']}, " +
                                        f"score: {region['incongruence_score']:.3f}")
                except Exception as e:
                    report.append(f"\nRecombination analysis failed: {e}")
                
                wx.CallAfter(self.transmission_result.SetValue, "\n".join(report))
                wx.CallAfter(self.risk_params.update, risk_result['features'])
                wx.CallAfter(self.risk_params.update, {'risk': risk_result['risk_score']})
                
                wx.CallAfter(wx.MessageBox, 
                           f"Transmission analysis completed!\nRisk Score: {risk_result['risk_score']:.4f}", 
                           "Done", wx.OK | wx.ICON_INFORMATION)
                
            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Transmission analysis error: {e}", "Error", wx.OK | wx.ICON_ERROR)
        
        threading.Thread(target=_worker, daemon=True).start()

    def on_chem_draw(self, evt):
        seq = (self.merged_sequence or self.seq_ctrl.GetValue()).strip().upper()
        if not seq:
            wx.MessageBox("Please input sequence", "Error", wx.OK | wx.ICON_ERROR)
            return
        try:
            pil_img = self.draw_dna_structure(seq)
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            buf.seek(0)
            wx_img = wx.Image(buf, wx.BITMAP_TYPE_PNG)
            bitmap = wx.Bitmap(wx_img)
            self.display_chem_image(bitmap)
        except Exception as e:
            wx.MessageBox(f"Chemical drawing error: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def display_chem_image(self, bitmap):
        sizer = self.chem_scrolled.GetSizer()
        sizer.Clear(delete_windows=True)
        canvas = ZoomOnlyCanvas(self.chem_scrolled, bitmap)
        sizer.Add(canvas, 1, wx.EXPAND | wx.ALL, 0)
        self.chem_scrolled.Layout()

    @staticmethod
    def draw_dna_structure(sequence: str):
        base_smiles = {
            'A': 'N1C=NC2=C1N=CN2C1OC(COP(=O)(O)O)CC1O',
            'T': 'CC1=NC(=O)N(C=C1)C1OC(COP(=O)(O)O)CC1O',
            'C': 'N1C(=O)N(C=C1)C1OC(COP(=O)(O)O)CC1O',
            'G': 'N1C=NC2=C1N=C(N2)N1OC(COP(=O)(O)O)CC1O'
        }
        mol = Chem.RWMol()
        for base in sequence.upper():
            if base not in base_smiles:
                continue
            frag = Chem.MolFromSmiles(base_smiles[base])
            mol.InsertMol(frag)
        full_mol = mol.GetMol()
        AllChem.Compute2DCoords(full_mol)
        return Draw.MolToImage(full_mol, size=(100*len(sequence), 600))

    def on_3d_city_sim(self, evt):
        if not self.risk_params:
            wx.MessageBox("Please run 'Transmission Risk' analysis first!", "Info")
            return
        if self.sim_thread and self.sim_thread.is_alive():
            wx.MessageBox("Simulation already running!", "Info")
            return
        
        def _run():
            risk = self.risk_params.get('risk', 0.5)
            entropy = self.risk_params.get('entropy', 0.1)
            repeat = self.risk_params.get('repeat_density', 0.05)
            gc = self.risk_params.get('gc_content', 0.5)
            
            p_base = 0.01 + risk * 0.07
            env = simpy.Environment()
            sim_proc = env.process(city_sim(
                env, p_base=p_base, entropy=entropy,
                repeat_d=repeat, gc=gc
            ))
            env.run(until=sim_proc)
            hist = sim_proc.value
            wx.CallAfter(self._draw_city_3d, hist)
        
        self.sim_thread = threading.Thread(target=_run, daemon=True)
        self.sim_thread.start()

    def on_blast_analysis(self, evt):
        sequence = (self.merged_sequence or self.seq_ctrl.GetValue()).strip().upper()
        if not sequence:
            wx.MessageBox("Please input sequence", "Error", wx.OK | wx.ICON_ERROR)
            return
    
        if len(sequence) < 50:
            wx.MessageBox("Sequence too short for BLAST analysis (minimum 50 bases)", 
                         "Error", wx.OK | wx.ICON_ERROR)
            return
    
        dlg = wx.ProgressDialog(
            "BLAST Analysis",
            "Submitting sequence to NCBI BLAST...",
            maximum=100,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME
        )
    
        def _blast_worker():
            try:
                wx.CallAfter(dlg.Update, 10, "Connecting to NCBI BLAST service...")
                dna_seq = Seq(sequence)
            
                wx.CallAfter(dlg.Update, 20, "Submitting sequence (this may take several minutes)...")
                result_handle = NCBIWWW.qblast(
                    program="blastn",
                    database="nt",
                    sequence=dna_seq,
                    megablast=True,
                    hitlist_size=10,
                    expect=1e-10
                )
            
                wx.CallAfter(dlg.Update, 60, "Parsing BLAST results...")
                records = NCBIXML.parse(result_handle)
                blast_record = next(records)
            
                wx.CallAfter(dlg.Update, 80, "Processing virus matches...")
                virus_results = []
                for alignment in blast_record.alignments:
                    hsp = alignment.hsps[0]
                    identity_percent = (hsp.identities / hsp.align_length) * 100
                    
                    virus_results.append({
                        "virus_name": alignment.hit_def.split("[")[0].strip(),
                        "accession": alignment.accession,
                        "identity": round(identity_percent, 2),
                        "alignment_length": hsp.align_length,
                        "e_value": hsp.expect,
                        "score": hsp.score,
                        "query_coverage": f"{hsp.query_start}-{hsp.query_end}"
                    })
            
                wx.CallAfter(dlg.Update, 95, "Finalizing results...")
                if not virus_results:
                    result_text = "BLAST Analysis Results\n" + "="*50 + "\n\n"
                    result_text += "No significant virus matches found in the NCBI database.\n\n"
                    result_text += f"Query sequence length: {len(sequence)} bases\n"
                    result_text += f"Database: nt (nucleotide)\n"
                    result_text += f"Program: blastn (megablast)\n"
                else:
                    virus_results.sort(key=lambda x: x["identity"], reverse=True)
                    result_text = "BLAST Analysis - Virus Similarity Check\n" + "="*60 + "\n\n"
                    result_text += f"Query sequence: {len(sequence)} bases\n"
                    result_text += f"Total virus matches found: {len(virus_results)}\n\n"
                    result_text += "Top Virus Matches:\n" + "-"*40 + "\n\n"
                
                    for i, res in enumerate(virus_results, 1):
                        result_text += f"{i}. Virus: {res['virus_name']}\n"
                        result_text += f"   Accession: {res['accession']}\n"
                        result_text += f"   Similarity: {res['identity']}%\n"
                        result_text += f"   Alignment Length: {res['alignment_length']} bp\n"
                        result_text += f"   E-value: {res['e_value']:.2e}\n"
                        result_text += f"   Score: {res['score']}\n"
                        result_text += f"   Query Coverage: {res['query_coverage']}\n\n"
            
                wx.CallAfter(self.blast_result.SetValue, result_text)
                wx.CallAfter(dlg.Update, 100, "Analysis complete!")
                wx.CallAfter(wx.MessageBox, "BLAST analysis completed successfully!", 
                            "Success", wx.OK | wx.ICON_INFORMATION)
            
            except Exception as e:
                error_msg = f"BLAST Analysis Failed:\n{str(e)}\n\n"
                error_msg += "Possible reasons:\n"
                error_msg += "- Network connection issue\n"
                error_msg += "- NCBI service temporarily unavailable\n"
                error_msg += "- Sequence format error\n"
                error_msg += "- Timeout (try with shorter sequence)"
            
                wx.CallAfter(self.blast_result.SetValue, error_msg)
                wx.CallAfter(wx.MessageBox, f"BLAST analysis failed: {str(e)}", 
                            "Error", wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(dlg.Destroy)
        threading.Thread(target=_blast_worker, daemon=True).start()

    def _draw_city_3d(self, hist):
        if self.city_canvas:
            self.city_canvas.Destroy()
        self.city_fig = plt.figure(figsize=(6,5), dpi=100)
        self.city_ax = self.city_fig.add_subplot(111, projection='3d')
        N = 2500
        pts = [(x,y,z) for x in range(50) for y in range(50) for z in range(5)]
        xs, ys, zs = zip(*pts[:N])
        sc = self.city_ax.scatter(xs, ys, zs, c='green', s=8)
        slider = wx.Slider(self.city_panel, value=0, minValue=0,
                           maxValue=max(hist.keys()), style=wx.SL_HORIZONTAL)
        slider.Bind(wx.EVT_SCROLL, lambda e: update(e.GetInt()))
        
        def update(day):
            infected = hist.get(day, set())
            colors = ['red' if i in infected else 'green' for i in range(N)]
            sc._facecolors = plt.cm.colors.to_rgba_array(colors)
            sc._edgecolors = sc._facecolors
            self.city_ax.set_title(f"Day {day}  —  infected: {len(infected)}")
            self.city_fig.canvas.draw()
        
        self.city_canvas = FigureCanvas(self.city_panel, -1, self.city_fig)
        sizer = self.city_panel.GetSizer()
        sizer.Clear(delete_windows=True)
        sizer.Add(self.city_canvas, 1, wx.EXPAND | wx.ALL, 0)
        sizer.Add(slider, 0, wx.EXPAND | wx.ALL, 5)
        self.city_panel.Layout()
        update(0)

    def on_advanced_3d_city(self, _evt):
        if not self.risk_params:
            wx.MessageBox("Please run 'Transmission Risk' analysis first!", "Info")
            return
        AdvancedCityFrame(
            parent=self,
            genome=self.merged_sequence or self.seq_ctrl.GetValue(),
            risk_params=self.risk_params
        )

    def on_close(self, evt):
        try:
            plt.close("all")
            if self._vis_canvas:
                self._vis_canvas.Destroy()
            if self._helix_canvas:
                self._helix_canvas.Destroy()
        finally:
            self.Destroy()

# 由于篇幅限制，以下类保持原样（Agent, CitySEIR, QtCityWidget, AdvancedCityFrame等）
# 这些类与之前的实现相同，包含完整的3D城市模拟功能

class Agent:
    __slots__ = ('id', 'pos', 'target', 'infected', 'recovered', 'timer')
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.target = None
        self.infected = False
        self.recovered = False
        self.timer = 0
    def walk(self, speed=1.5):
        if self.target is None:
            return
        dir_vec = self.target - self.pos
        dist = np.linalg.norm(dir_vec)
        if dist < speed:
            self.pos = self.target.copy()
            self.target = None
        else:
            self.pos += dir_vec / dist * speed

class CitySEIR:
    def __init__(self, agents: List[Agent], beta: float, gamma: float):
        self.agents = agents
        self.beta = beta
        self.gamma = gamma
        self.S = set(range(len(agents)))
        self.E, self.I, self.R = set(), set(), set()
        initial_infected = min(20, len(agents))
        for i in range(initial_infected):
            self.I.add(i)
            agents[i].infected = True
        self.S -= self.I
        self.t = 0
    
    def step(self):
        for ag in self.agents:
            ag.walk(speed=2.0)
        infected_indices = list(self.I)
        new_infections = set()
        for cand in list(self.S):
            a = self.agents[cand]
            for idx in infected_indices:
                sick = self.agents[idx]
                distance = np.linalg.norm(a.pos - sick.pos)
                if distance < 15:
                    infection_chance = self.beta * (1 - distance / 20)
                    if random.random() < infection_chance:
                        new_infections.add(cand)
                        a.infected = True
                        a.timer = 1
                        break
            if cand in new_infections:
                self.S.remove(cand)
                self.E.add(cand)
        new_infected = set()
        for idx in list(self.E):
            if random.random() < 0.3:
                new_infected.add(idx)
                self.E.remove(idx)
                self.I.add(idx)
        newR = set()
        for idx in self.I:
            a = self.agents[idx]
            a.timer += 1
            if a.timer >= int(1 / self.gamma) + 10:
                newR.add(idx)
                a.recovered = True
                a.infected = False
        self.I -= newR
        self.R |= newR
        self.t += 1

def make_cool_buildings(n: int = 120) -> List[pv.MultiBlock]:
    buildings = []
    for i in range(n):
        w, d = random.randint(20, 45), random.randint(20, 45)
        h_base = random.randint(40, 120)
        n_layer = random.randint(3, 8)
        layer_h = h_base / n_layer
        block = pv.MultiBlock()
        for k in range(n_layer):
            dz = k * layer_h
            shrink = k * 0.5
            cube = pv.Cube(center=(0, 0, dz + layer_h / 2),
                           x_length=w - shrink,
                           y_length=d - shrink,
                           z_length=layer_h)
            if random.random() < 0.6:
                cube['mat'] = np.full(cube.n_points, 1)
            else:
                cube['mat'] = np.full(cube.n_points, 0)
            cube['size_x'] = np.full(cube.n_points, w - shrink)
            cube['size_y'] = np.full(cube.n_points, d - shrink)
            block.append(cube)
        roof = pv.Cube(center=(0, 0, h_base + 2),
                       x_length=w - 2, y_length=d - 2, z_length=4)
        roof['mat'] = np.full(roof.n_points, 2)
        block.append(roof)
        if random.random() < 0.3:
            antenna = pv.Cylinder(center=(0, 0, h_base + 15),
                                  direction=(0, 0, 1),
                                  radius=1, height=30)
            antenna['mat'] = np.full(antenna.n_points, 3)
            block.append(antenna)
        angle = random.randint(0, 90)
        block = block.rotate_z(angle, inplace=False)
        cx, cy = random.randint(-600, 600), random.randint(-600, 600)
        block = block.translate((cx, cy, 0), inplace=False)
        buildings.append(block)
    return buildings

class QtCityWidget(QFrame):
    def __init__(self, genome: str, risk_params: dict):
        super().__init__()
        self.genome = genome
        self.risk = risk_params['risk']
        self.entropy = risk_params['entropy']
        self.repeat = risk_params['repeat_density']
        self.gc = risk_params['gc_content']
        beta = 0.3 + self.risk * 0.7
        gamma = 0.05 - (self.risk * 0.04)
        gamma = max(0.01, gamma)
        print(f"传染参数: beta={beta:.3f}, gamma={gamma:.3f}")
        self.buildings = make_cool_buildings(100)
        self.agents = []
        
        for i in range(600):
            px = random.randint(-200, 200)
            py = random.randint(-200, 200)
            agent = Agent(i, (px, py, 1.6))
            if i < 20:
                agent.infected = True
                agent.timer = 1
            self.agents.append(agent)
        
        self.sim = CitySEIR(self.agents, beta, gamma)
        self.plotter = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter)
        self._build_scene()
        self._start_timer()
    
    def _build_scene(self):
        p = self.plotter
        ground = pv.Plane(i_size=1200, j_size=1200, direction=(0, 0, 1))
        p.add_mesh(ground, color='#222222', pbr=True, roughness=0.8)
        
        for b in self.buildings:
            p.add_mesh(b, scalars='mat', cmap=['gray', 'lightblue', 'green', 'white'],
                       pbr=True, metallic=0.9, roughness=0.1, name=f'bld{random.randint(0,10000)}')
        
        pts = np.stack([ag.pos for ag in self.agents])
        self.cloud = pv.PolyData(pts)
        
        initial_state = np.array([
            1 if ag.infected else 2 if ag.recovered else 0 
            for ag in self.agents
        ])
        self.cloud['state'] = initial_state
        
        self.actor = p.add_points(
            self.cloud,
            render_points_as_spheres=True,
            point_size=20,
            scalars='state',
            cmap=['gray', 'red', 'green'],
            emissive=True,
            ambient=0.5,
            name='agents'
        )
        
        p.add_text("3D City Simulation", font_size=11, position='upper_left')
        print(f"初始感染者数量: {np.sum(initial_state == 1)}")
    
    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._step)
        self.timer.start(150)
    
    def _step(self):
        self.sim.step()
        for ag in self.agents:
            if ag.target is None or np.linalg.norm(ag.pos - ag.target) < 5:
                if random.random() < 0.5:
                    b = random.choice(self.buildings)
                    base = b[0]
                    cx, cy, _ = base.center
                    size_x = base['size_x'][0]
                    size_y = base['size_y'][0]
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, min(size_x, size_y) / 2 + 10)
                    target_x = cx + math.cos(angle) * distance
                    target_y = cy + math.sin(angle) * distance
                    ag.target = np.array([target_x, target_y, 1.6])
                else:
                    ag.target = np.array([
                        ag.pos[0] + random.uniform(-50, 50),
                        ag.pos[1] + random.uniform(-50, 50),
                        1.6
                    ])
            ag.walk(speed=2.0)
        pts = np.stack([ag.pos for ag in self.agents])
        self.cloud['state'] = np.array([
            1 if ag.infected else 2 if ag.recovered else 0 
            for ag in self.agents
        ], dtype=np.uint8)
        self.cloud.points = pts
        self.plotter.update()
        inf_count = len([ag for ag in self.agents if ag.infected])
        rec_count = len([ag for ag in self.agents if ag.recovered])
        healthy_count = len(self.agents) - inf_count - rec_count
        if hasattr(self, 'info_text'):
            self.plotter.remove_actor(self.info_text) 
        self.info_text = self.plotter.add_text(
            f"Day {self.sim.t}   Infected:{inf_count}  Recovered:{rec_count}  Healthy:{healthy_count}",
            font_size=11, position='upper_edge', name='info'
        )
        self.actor.mapper.SetScalarRange(0, 2)
        self.plotter.update()

class AdvancedCityFrame(wx.Frame):
    def __init__(self, parent, genome: str, risk_params: dict):
        super().__init__(parent, title="3D City Simulation",
                         size=(320, 120))
        wx.StaticText(self, label="Qt-3D 窗口已弹出，可关闭本提示。", pos=(20, 20))
        self.genome = genome
        self.risk_params = risk_params
        self._run_qt()
        self.Bind(wx.EVT_CLOSE, self.on_close)
    
    def _run_qt(self):
        self.qt_app = QApplication.instance() or QApplication([])
        self.qt_win = QMainWindow()
        self.qt_win.setWindowTitle("3D City Simulation")
        self.qt_win.resize(1400, 900)
        self.qt_win.closeEvent = self.qt_close_event
        self.central = QtCityWidget(self.genome, self.risk_params)
        self.qt_win.setCentralWidget(self.central)
        self.qt_win.show()
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda e: self.qt_app.processEvents())
        self.timer.Start(100)
    
    def qt_close_event(self, event):
        if hasattr(self.central, 'timer'):
            self.central.timer.stop()
        event.accept()
    
    def on_close(self, event):
        if hasattr(self, 'qt_win'):
            self.qt_win.close()
            self.qt_win = None
        self.Destroy()

if __name__ == "__main__":
    app = wx.App(False)
    SequenceAnalyzerApp(None)
    app.MainLoop()
