import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import pandas as pd
import scipy.io as sci
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import DeformConv2d
from scipy import signal
import random
from scipy.signal import find_peaks


dir='./signal_frags'  # Directory for the input signal fragments after signal preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp=os.listdir(dir)
ecg_sig=pd.DataFrame(columns=["y_hr","wt","X_ecg_mat", "X_radar_mat","pat_id"])
for i in temp:
    if not i.endswith("_Tkeep.mat"):
        data=sci.loadmat(os.path.join(dir, i),struct_as_record=False, squeeze_me=True)

        df = pd.DataFrame({
            "y_hr": data["y_hr"],
            "wt": data["wt"],
            "X_ecg_mat": list(data["X_ecg_mat"]),
            "X_radar_mat": list(data["X_radar_mat"])
        })

        
        s = pd.Series([i] * df.shape[0])
        df["pat_id"]=s 
        if ecg_sig.empty:
            ecg_sig = df.copy()  # initialize directly
        else:
            ecg_sig = pd.concat([ecg_sig, df], axis=0, ignore_index=True)
        
        #ecg_sig.append([i,data['y_hr'],data['X_ecg_mat'],data['X_radar_mat'],data['wt']])

ecg_sig = ecg_sig[[ecg_sig.columns[-1]] + ecg_sig.columns[:-1].tolist()]
### Split data####
train_parts = []
test_parts = []

# loop over each patient
for pid, group in ecg_sig.groupby('pat_id'):
    n_rows = len(group)
    n_train = int(np.ceil(0.9 * n_rows))  # 90%
    
    # shuffle within this patient's group
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # split into train/test
    train_parts.append(group.iloc[:n_train])
    test_parts.append(group.iloc[n_train:])

# combine everything back
train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)



def extract_rr_intervals(
    x,
    fs: float,
    hr_min: float = 35.0,
    hr_max: float = 180.0,
    prominence: float = 0.6,
):
    """
    Extract RR intervals (seconds) from a 1D signal fragment.
    Returns:
      rr_sec: np.ndarray of RR intervals in seconds (len = n_peaks-1)
      peaks:  np.ndarray of peak indices (samples)
    """
    x = np.asarray(x, dtype=np.float32)
    x0 = x - np.mean(x)
    xz = x0 / (np.std(x0) + 1e-8)

    # min distance between peaks based on maximum HR
    min_dist = int(fs * 60.0 / hr_max)
    min_dist = max(1, min_dist)

    peaks, props = find_peaks(xz, distance=min_dist, prominence=prominence)

    if peaks.size < 2:
        return np.array([], dtype=np.float32), peaks

    rr_sec = np.diff(peaks).astype(np.float32) / float(fs)

    # filter RR intervals to physiological range
    rr_lo = 60.0 / hr_max
    rr_hi = 60.0 / hr_min
    rr_sec = rr_sec[(rr_sec >= rr_lo) & (rr_sec <= rr_hi)]

    return rr_sec, peaks


def rr_mae_ms(rr_ref_sec, rr_est_sec):
    """
    RR-interval MAE (ms) after simple truncation to common length.
    Returns np.nan if not enough intervals.
    """
    rr_ref_sec = np.asarray(rr_ref_sec, dtype=np.float32)
    rr_est_sec = np.asarray(rr_est_sec, dtype=np.float32)

    K = min(len(rr_ref_sec), len(rr_est_sec))
    if K < 1:
        return np.nan
    return float(np.mean(np.abs(rr_ref_sec[:K] - rr_est_sec[:K])) * 1000.0)


class MultiECGDataset(Dataset):
    def __init__(self, ecg_signals, fs=2000, hr_max=180, hr_min=35):
        """
        Args:
            ecg_signals: list of tuples (I_signal, Q_signal, ecg, timestamps, pat_id)
            original_fs: sampling rate (Hz) used for BOTH streams
            window_size: seconds per fragment (default 12 s)
            overlap: seconds overlap (default 6 s)
            hr_min, hr_max: HR normalization bounds (bpm)
            return_meta: if True, __getitem__ returns dict with diagnostics
        """
        self.ecg_signals = ecg_signals
        self.hr_max=hr_max
        self.hr_min=hr_min
        self.fs=fs

    def mk_sos(self, fs):
        sos_hp_radar = signal.butter(2, 0.1, btype='highpass', fs=fs, output='sos')      # 0.1 Hz
        sos_hr = signal.butter(4, [0.8, 5.0], btype='bandpass', fs=fs, output='sos')     # HR view
        sos_ecg = signal.butter(4, [0.5, 40], btype='bandpass', fs=fs, output='sos')     # ECG base
        return sos_hp_radar, sos_hr, sos_ecg

    def filtfilt_safe(self, sos, x):
        # shorter pad to avoid edge weirdness on 10 s windows
        pad = min(3000, max(0, len(x) - 1))
        return signal.sosfiltfilt(sos, x, padlen=pad)

    def preprocess_radar(self, x, sos_hp_radar, sos_hr):
        # x is already unwrapped + mean-subtracted
        x = signal.detrend(x, type='linear')                      # same length
        x = self.filtfilt_safe(sos_hp_radar, x)                        # high-pass baseline
        x_hr = self.filtfilt_safe(sos_hr, x)                           # HR-band view (for lag / input channel)
    
        return x, x_hr

    def preprocess_ecg(self, ecg, sos_ecg, sos_hr, notch=None):
        ecg = self.filtfilt_safe(sos_ecg, ecg)
        if notch is not None:
            b, a = notch
            ecg = signal.filtfilt(b, a, ecg)
        ecg_hr = self.filtfilt_safe(sos_hr, ecg)
        return ecg, ecg_hr
    
    def __len__(self):
        return len(self.ecg_signals)

    def __getitem__(self, idx):
        sos_hp_radar, sos_hr, sos_ecg = self.mk_sos(self.fs)

        row = self.ecg_signals.iloc[idx]
        pat_id = row["pat_id"]
        y_hr = (row["y_hr"])
        wt = (row["wt"])

        

        # X_ecg_mat / X_radar_mat are arrays stored in the row (object dtype)
        fragment_ecg = np.asarray(row["X_ecg_mat"], dtype=np.float32)
        _, fragment_ecg=self.preprocess_ecg(fragment_ecg, sos_ecg, sos_hr)

        fragment_radar = np.asarray(row["X_radar_mat"], dtype=np.float32)
        _, fragment_radar=self.preprocess_radar(fragment_radar, sos_hp_radar, sos_hr)

        ecg_mean, ecg_std = float(fragment_ecg.mean()), float(fragment_ecg.std() + 1e-6)
        rad_mean, rad_std = float(fragment_radar.mean()), float(fragment_radar.std() + 1e-6)

        ecg_z = (fragment_ecg - ecg_mean) / ecg_std
        rad_z = (fragment_radar - rad_mean) / rad_std

        x_ecg = torch.tensor(ecg_z.copy(), dtype=torch.float32)

        x_radar = torch.tensor(rad_z.copy(), dtype=torch.float32)

        #y_norm = (y_hr - np.mean(y_hr)) / (np.std(y_hr)+1e-6)#
        return pat_id, x_ecg, x_radar, y_hr, wt




    # --------------------------
    # Utility for inverse transform
    # --------------------------
    def denormalize_hr(self, hr_norm, hr_std, hr_mean):
        """Convert normalized HR [0,1] back to bpm."""
        return hr_norm * hr_std + hr_mean
        #return hr_norm*180


train_ds = MultiECGDataset(train_df)
train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)

test_ds = MultiECGDataset(test_df)
test_dataloader = DataLoader(test_ds, batch_size=8, shuffle=True)



class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

#SE Block
class SEBlock1D(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels//r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels//r), channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b,c,_ = x.shape
        s = self.pool(x).view(b,c)
        s = self.fc(s).view(b,c,1)
        return x * s

def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

# Upsampling block
class UpBlock1D(nn.Module):
    """Upsample x by 2, concat with skip, then two convs."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act   = Swish()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.apply(_init_weights)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        # guard off-by-one
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


# 1-D Deformable Convolutions
class DeformableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DeformableConv1D, self).__init__()
        # Offset generator for 1D convolution
        self.offsets = nn.Conv1d(in_channels, 2 * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Deformable 2D convolution treated as 1D (height = 1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding))

    def forward(self, x):
        # Reshape the input to simulate a 2D convolution (batch, channels, 1, width)
        batch_size, channels, width = x.size()
        x_2d = x.unsqueeze(2)  # Add height dimension (batch, channels, 1, width)
        
        # Generate offsets and reshape for 2D deformable convolution
        offsets = self.offsets(x).unsqueeze(2)  # Offset shape: (batch, 2*kernel_size, 1, width)
        
        # Apply deformable convolution
        out = self.deform_conv(x_2d, offsets)
        
        # Squeeze the height dimension back to restore 1D output
        return out.squeeze(2)

# Encoder block
class EncoderHR(nn.Module):
    def __init__(self, in_channels=1, feature_channels=8):
        super().__init__()
        self.act = Swish()
        self.drop = nn.Dropout(0.2)

        # your blocks (kept as in your Encoder)
        self.enc1 = DeformableConv1D(in_channels, feature_channels, kernel_size=3, stride=2)
        self.bn1  = nn.BatchNorm1d(feature_channels)
        self.se1  = SEBlock1D(feature_channels)

        self.enc2 = DeformableConv1D(feature_channels, feature_channels*2, kernel_size=3, stride=2)
        self.conv2= nn.Conv1d(feature_channels, feature_channels*2, kernel_size=1, stride=2)
        self.bn2  = nn.BatchNorm1d(feature_channels*2)
        self.se2  = SEBlock1D(feature_channels*2)

        self.enc3 = DeformableConv1D(feature_channels*2, feature_channels*4, kernel_size=3, stride=2)
        self.conv3= nn.Conv1d(feature_channels*2, feature_channels*4, kernel_size=1, stride=2)
        self.bn3  = nn.BatchNorm1d(feature_channels*4)
        self.se3  = SEBlock1D(feature_channels*4)

        self.enc4 = DeformableConv1D(feature_channels*4, feature_channels*8, kernel_size=3, stride=2)
        self.conv4= nn.Conv1d(feature_channels*4, feature_channels*8, kernel_size=1, stride=2)
        self.bn4  = nn.BatchNorm1d(feature_channels*8)
        self.se4  = SEBlock1D(feature_channels*8)

        self.enc5 = DeformableConv1D(feature_channels*8, feature_channels*16, kernel_size=3, stride=2)
        self.conv5= nn.Conv1d(feature_channels*8, feature_channels*16, kernel_size=1, stride=2)
        self.bn5  = nn.BatchNorm1d(feature_channels*16)

        # HR head (GAP → MLP)
        C = feature_channels*16
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(C, 128)
        self.fc2 = nn.Linear(128, 1)  # final HR regressor (bpm or z-units)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # block 1
        x = self.bn1(self.enc1(x)); x = self.act(x); x = self.drop(x); x = self.se1(x)
        # block 2
        id2 = self.bn2(self.conv2(x)); x = self.bn2(self.enc2(x)); x = self.act(x + id2); x = self.drop(x); x = self.se2(x)
        # block 3
        id3 = self.bn3(self.conv3(x)); x = self.bn3(self.enc3(x)); x = self.act(x + id3); x = self.drop(x); x = self.se3(x)
        # block 4
        id4 = self.bn4(self.conv4(x)); x = self.bn4(self.enc4(x)); x = self.act(x + id4); x = self.drop(x); x = self.se4(x)
        # block 5
        id5 = self.bn5(self.conv5(x)); x = self.bn5(self.enc5(x)); x = self.act(x + id5); x = self.drop(x)

        feat = x                               # [B, C, T_reduced] features for distillation if needed
        v = self.gap(x).squeeze(-1)            # [B, C]
        v = self.act(self.fc1(v))              # [B, 128]
        y = self.fc2(v).squeeze(-1)            # [B]  (HR reg)

        return y, feat




#CAR-Net
class EncoderUNetHR(nn.Module):
    """
    Radar -> encoder (deformable) with 5 strides (x/32) + SE
          -> HR head (z-score output)
          -> decoder (simple conv) with skips -> phase_hat [B,T]
    Returns: y_hr_z, phase_hat, latent, [s1..s5]
    """
    def __init__(self, in_channels=1, feature_channels=8):
        super().__init__()
        C1 = feature_channels
        C2 = feature_channels*2
        C3 = feature_channels*4
        C4 = feature_channels*8
        C5 = feature_channels*16

        self.act  = Swish()
        self.drop = nn.Dropout(0.2)

        # ----- encoder -----
        self.enc1 = DeformableConv1D(in_channels, C1, kernel_size=3, stride=2)
        self.bn1  = nn.BatchNorm1d(C1)
        self.se1  = SEBlock1D(C1)

        self.enc2 = DeformableConv1D(C1, C2, kernel_size=3, stride=2)
        self.conv2= nn.Conv1d(C1, C2, kernel_size=1, stride=2)
        self.bn2  = nn.BatchNorm1d(C2)
        self.se2  = SEBlock1D(C2)

        self.enc3 = DeformableConv1D(C2, C3, kernel_size=3, stride=2)
        self.conv3= nn.Conv1d(C2, C3, kernel_size=1, stride=2)
        self.bn3  = nn.BatchNorm1d(C3)
        self.se3  = SEBlock1D(C3)

        self.enc4 = DeformableConv1D(C3, C4, kernel_size=3, stride=2)
        self.conv4= nn.Conv1d(C3, C4, kernel_size=1, stride=2)
        self.bn4  = nn.BatchNorm1d(C4)
        self.se4  = SEBlock1D(C4)

        self.enc5 = DeformableConv1D(C4, C5, kernel_size=3, stride=2)
        self.conv5= nn.Conv1d(C4, C5, kernel_size=1, stride=2)
        self.bn5  = nn.BatchNorm1d(C5)

        # ----- HR head (z-space) -----
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.fc1  = nn.Linear(C5, 128)
        self.fc2  = nn.Linear(128, 1)

        # ----- decoder (simple conv U-Net) -----
        self.up4 = UpBlock1D(in_ch=C5, skip_ch=C4, out_ch=C4)  # T/32 -> T/16
        self.up3 = UpBlock1D(in_ch=C4, skip_ch=C3, out_ch=C3)  # T/16 -> T/8
        self.up2 = UpBlock1D(in_ch=C3, skip_ch=C2, out_ch=C2)  # T/8  -> T/4
        self.up1 = UpBlock1D(in_ch=C2, skip_ch=C1, out_ch=C1)  # T/4  -> T/2

        self.dec_top = nn.Sequential(                          # T/2 -> T
            nn.Conv1d(C1, C1, kernel_size=3, padding=1),
            nn.BatchNorm1d(C1),
            Swish(),
        )
        self.recon_head = nn.Conv1d(C1, 1, kernel_size=1)      # phasê

        # init
        self.apply(_init_weights)

    def forward(self, x):
        """
        x: [B,1,T]
        Returns:
          y_hr_z:   [B]     (z-scored HR prediction)
          phase_hat:[B,T]   (reconstructed phase/displacement-like target)
          latent:   [B,C5,T/32]
          skips:    list of [s1..s5] encoder features for KD if needed
        """
        T_in = x.size(-1)

        # ----- encoder -----
        x1 = self.bn1(self.enc1(x));          x1 = self.act(x1); x1 = self.drop(x1); x1 = self.se1(x1)       # [B,C1,T/2]
        id2= self.bn2(self.conv2(x1));        x2 = self.bn2(self.enc2(x1));  x2 = self.act(x2 + id2); x2 = self.drop(x2); x2 = self.se2(x2)  # [B,C2,T/4]
        id3= self.bn3(self.conv3(x2));        x3 = self.bn3(self.enc3(x2));  x3 = self.act(x3 + id3); x3 = self.drop(x3); x3 = self.se3(x3)  # [B,C3,T/8]
        id4= self.bn4(self.conv4(x3));        x4 = self.bn4(self.enc4(x3));  x4 = self.act(x4 + id4); x4 = self.drop(x4); x4 = self.se4(x4)  # [B,C4,T/16]
        id5= self.bn5(self.conv5(x4));        x5 = self.bn5(self.enc5(x4));  x5 = self.act(x5 + id5); x5 = self.drop(x5)                     # [B,C5,T/32]
        latent = x5
        skips  = [x1, x2, x3, x4, x5]

        # ----- HR head (z-space) -----
        v = self.gap(x5).squeeze(-1)          # [B,C5]
        v = self.act(self.fc1(v))             # [B,128]
        y_hr_z = self.fc2(v).squeeze(-1)      # [B]

        # ----- decoder (U-Net) -----
        y = x5
        y = self.up4(y, x4)                   # [B,C4,T/16]
        y = self.up3(y, x3)                   # [B,C3,T/8]
        y = self.up2(y, x2)                   # [B,C2,T/4]
        y = self.up1(y, x1)                   # [B,C1,T/2]
        y = F.interpolate(y, size=T_in, mode="linear", align_corners=False)  # [B,C1,T]
        y = self.dec_top(y)                   # [B,C1,T]
        phase_hat = self.recon_head(y).squeeze(1)  # [B,T]

        return y_hr_z, phase_hat, latent, skips




# Projection head for embeddings
class Proj1D(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, f):          # f: [B,C,Tred]
        return self.proj(f).mean(dim=-1)  # [B,out_ch]

# Loading Teacher model (SED-Net)
teacher = EncoderHR(in_channels=1, feature_channels=8).to(device)
checkpoint=torch.load("./trained_models/teacher_best.pth", weights_only=True)
teacher.load_state_dict(checkpoint["model_state"])
# To avoid training teacher weights
for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# Loading Student model (CAR-Net)
student=EncoderUNetHR(in_channels=1, feature_channels=8).to(device)
student_check=torch.load("./trained_models/student_best.pt", map_location=torch.device('cuda'), weights_only=True)
student.load_state_dict(student_check["student"])

# Projection Heads for Teacher and Student
C5 = 8*16  # if feature_channels changes, compute dynamically instead
proj_T = Proj1D(in_ch=C5, out_ch=128).to(device);  [p.requires_grad_(False) for p in proj_T.parameters()]
proj_S = Proj1D(in_ch=C5, out_ch=128).to(device)
proj_S.load_state_dict(student_check["proj_S"])
proj_T.eval()
proj_S.eval()

#Data Loaders
train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_dataloader  = DataLoader(test_ds,  batch_size=8, shuffle=False)

fs=2000

# Mean and Standard Deviation calculation from training set for value denormalisation
train_hr_mean=train_df['y_hr'].mean()
train_hr_std = float(train_df['y_hr'].std().clip(1e-6))
HR_MU  = float(train_hr_mean)
HR_SIG = float(train_hr_std if train_hr_std > 1e-6 else 1e-6)




student.eval()
teacher.eval()
fig_dir="./figs"
os.makedirs(fig_dir, exist_ok=True)


# For removing edge artifacts
def crop_edges(x, fs, crop_sec=0.3):
    n = len(x)
    k = int(crop_sec * fs)

    if n <= 2*k + 10:
        t = np.arange(n) / fs
        return x, t

    x_crop = x[k:n-k]
    t_crop = np.arange(len(x_crop)) / fs   # starts at 0
    return x_crop, t_crop


# R-R intervals calculation
def extract_rr_intervals(
    x,
    fs: float,
    hr_min: float = 35.0,
    hr_max: float = 180.0,
    prominence: float = 0.9,
):
    """
    Extract RR intervals (seconds) from a 1D signal fragment.
    Returns:
      rr_sec: np.ndarray of RR intervals in seconds (len = n_peaks-1)
      peaks:  np.ndarray of peak indices (samples)
    """
    x = np.asarray(x, dtype=np.float32)
    x0 = x - np.mean(x)
    xz = x0 / (np.std(x0) + 1e-8)

    # min distance between peaks based on maximum HR
    min_dist = int(fs * 60.0 / hr_max)
    min_dist = max(1, min_dist)

    peaks, props = find_peaks(xz, distance=min_dist, prominence=prominence)

    if peaks.size < 2:
        return np.array([], dtype=np.float32), peaks

    rr_sec = np.diff(peaks).astype(np.float32) / float(fs)

    # filter RR intervals to physiological range
    rr_lo = 60.0 / hr_max
    rr_hi = 60.0 / hr_min
    rr_sec = rr_sec[(rr_sec >= rr_lo) & (rr_sec <= rr_hi)]

    return rr_sec, peaks

# R-R intervals MSE
def rr_mae_ms(rr_ref_sec, rr_est_sec):
    """
    RR-interval MAE (ms) after simple truncation to common length.
    Returns np.nan if not enough intervals.
    """
    rr_ref_sec = np.asarray(rr_ref_sec, dtype=np.float32)
    rr_est_sec = np.asarray(rr_est_sec, dtype=np.float32)

    K = min(len(rr_ref_sec), len(rr_est_sec))
    if K < 1:
        return np.nan
    return float(np.mean(np.abs(rr_ref_sec[:K] - rr_est_sec[:K])) * 1000.0)

records = []  # to store per-fragment metrics
# Test loop
with torch.no_grad():
    for b_idx, batch in enumerate(test_dataloader):
        # unpack
        pat_id, x_ecg, x_radar, y_hr_bpm, w = batch      # x_*: [B, T]
        x_radar  = x_radar.to("cuda").float().unsqueeze(1)  # [B,1,T]
        x_ecg    = x_ecg.to("cuda").float().unsqueeze(1)    # [B,1,T]
        y_hr_bpm = y_hr_bpm.to("cuda").float().view(-1)     # [B]

        # forward student + teacher
        yS_z, phase_hat, feat_S, _ = student(x_radar)       # yS_z: [B], phase_hat: [B,T]
        hr_norm, _ = teacher(x_ecg)                         # teacher HR (z-space if trained that way)

        # BPM versions (using training-set HR_MU, HR_SIG)
        yS_bpm = (yS_z.view(-1) * HR_SIG + HR_MU).detach().cpu().numpy()  # [B]
        hr_bpm = (hr_norm.view(-1) * HR_SIG + HR_MU).detach().cpu().numpy()  # [B]

        # move signals to CPU & numpy
        phat_batch = phase_hat.detach().cpu().numpy()  # [B, T]
        ptgt_batch = x_ecg[:, 0].detach().cpu().numpy()  # [B, T]
        B, T = phat_batch.shape
        t = np.arange(T) / float(fs)                   # [T]

        for i in range(B):
            pat_id1=pat_id[i]
            phat_i = phat_batch[i]  # [T]
            ptgt_i = ptgt_batch[i]  # [T]

            # zero-mean per fragment
            phat0 = phat_i - phat_i.mean()
            ptgt0 = ptgt_i - ptgt_i.mean()

            # per-fragment z-MAE & z-corr
            mae_sample  = float(np.mean(np.abs(phat0 - ptgt0)))
            corr_sample = float(np.corrcoef(phat0, ptgt0)[0, 1])


            # --- RR intervals ---
            ptgt0_cropped,t_ptgt0=crop_edges(ptgt0, fs=fs, crop_sec=0.3)
            rr_ecg_sec, peaks_ecg = extract_rr_intervals(
                ptgt0_cropped, fs=fs, hr_min=35, hr_max=180, prominence=0.6
            )
            phat0_cropped,t_phat0=crop_edges(phat0, fs=fs, crop_sec=0.3)
            rr_rec_sec, peaks_rec = extract_rr_intervals(
                phat0_cropped, fs=fs, hr_min=35, hr_max=180, prominence=0.6
            )

            rr_err_ms = rr_mae_ms(rr_ecg_sec, rr_rec_sec)
            t = np.arange(T) / float(fs)                   # [T]
            # plot
            plt.figure(figsize=(10, 4))
            plt.plot(
                t_ptgt0, ptgt0_cropped / (ptgt0_cropped.std() + 1e-8),
                label="ECG z (target)", linewidth=1.2
            )
            plt.plot(
                t_phat0, phat0_cropped / (phat0_cropped.std() + 1e-8),
                label="Student recon (z)", linewidth=1.0, alpha=0.85
            )

            plt.grid(True, alpha=0.3)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalised Amplitude (z-score normalisation)")
            plt.title(
                f"HR_pred={yS_bpm[i]:.1f} bpm | "
                f"HR_gt={hr_bpm[i]:.1f} bpm | "
                f"z-MAE={mae_sample:.3f} | z-Corr={corr_sample:.2f}"
            )
            plt.legend()

            fig_name = f"b{pat_id1}_frag{i:02d}_recon.png"
            fig_path = os.path.join(fig_dir, fig_name)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=600)
            plt.close()

            print(f"Saved reconstruction demo for batch {b_idx}, fragment {i} -> {fig_path}")

            # store metrics for this fragment
            records.append({
                "batch_idx": pat_id1,
                "frag_idx": i,
                "hr_pred_bpm": yS_bpm[i],
                "hr_gt_bpm": hr_bpm[i],
                "z_mae": mae_sample,
                "z_corr": corr_sample,
                "rr_mae":rr_err_ms
            })

# optional: save all metrics to CSV
# Storing results
if records:
    df_metrics = pd.DataFrame(records)
    csv_path = os.path.join(fig_dir, "recon_metrics_test.csv")
    df_metrics.to_csv(csv_path, index=False)
    print("Saved per-fragment metrics to:", csv_path)

#print("Best checkpoint:", best_paths[0] if best_paths else "—")
