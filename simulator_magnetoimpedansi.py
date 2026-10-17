"""Port Python dari simulator-magnetoimpedansi-teoretis.html (JS vanilla).

Satu file, hanya stdlib (+ matplotlib opsional untuk plot).
Rumus mengikuti JS baris-per-baris: LLG/Smit-Beljers -> chiLayer -> zOne (Maxwell
trilayer) -> zSensor(Phi histeresis) -> zSensorAvg(GH-Hk) -> totalZ (+partikel,
Maxwell-Garnett, GH dipolar) -> modelCurve/measure/peakAt/slopeAt/sigmaMIR/S/LOD.
Utilitas analisis juga diport: clamp_P/clamp_mat (tiruan getP/getMat), interpModel,
modelBand/modelMIR, parseExpCsv, nelderMead, expCost, fitSensor/fitCd, dan
measureCsv (tiruan unduhan CSV halaman HTML).

Konvensi: kompleks = Python complex, waktu exp(+j w t) seperti JS [re, im].
Satuan: H dalam mT, B dalam T, f_MHz di P['f'], massa gram, tp/tebal cm.
Phi memakai math.erf (lebih akurat dari aproksimasi erf JS ~1.5e-7); rumus lain
identik baris-per-baris.

Divergensi yang disengaja dari JS: operator `or` Python menganggap NaN truthy
sedangkan `||` JS tidak (NaN dari dict langsung tidak jatuh ke default); input
degenerate (f=0 pada z_one, plateau datar pada peak_info) mengembalikan NaN
seperti JS alih-alih raise; CLI menolak --step <= 0 (JS tidak punya CLI).

Contoh:
    conda run -n nr python simulator_magnetoimpedansi.py
    conda run -n nr python simulator_magnetoimpedansi.py --plot --csv out.csv
    conda run -n nr python simulator_magnetoimpedansi.py --meas-csv ukur.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import re
import sys
import warnings

# ---------------- Konstanta fisika (sama dgn JS) ----------------
MU0 = 4e-7 * math.pi
KB = 1.380649e-23
GAM = 1.76085963e11  # rad/s/T

# Gauss-Hermite 7 titik ternormalisasi: E_N(mu,sig)[f] ~= sum W f(mu+sig X)
GH_X = [-3.750440, -2.366759, -1.154405, 0.0, 1.154405, 2.366759, 3.750440]
GH_W = [0.000548269, 0.0307571, 0.240123, 0.457143, 0.240123, 0.0307571, 0.000548269]

# ---------------- Preset (disalin dari JS MATS/SENS/CFG defaults) ----------------
MATS = {
    "none": dict(name="Tanpa partikel", type="none", Msp=0, d=10, K=0, tau0=1, alphaP=0.05, chiV=0, rho=5),
    "fe3o4": dict(name="Fe3O4 superparamagnetik 10 nm", type="sp", Msp=400, d=10, K=11, tau0=1, alphaP=0.05, chiV=0, rho=5.18),
    "fe2o3": dict(name="gamma Fe2O3 superparamagnetik 10 nm", type="sp", Msp=350, d=10, K=4.7, tau0=1, alphaP=0.05, chiV=0, rho=4.9),
    "cofe2o4": dict(name="CoFe2O4 terblokir 20 nm", type="fm", Msp=350, d=20, K=200, tau0=1, alphaP=0.05, chiV=0, rho=5.3),
    "ni": dict(name="Ni terblokir 30 nm", type="fm", Msp=485, d=30, K=50, tau0=1, alphaP=0.05, chiV=0, rho=8.9),
    "co": dict(name="Co terblokir 30 nm", type="fm", Msp=1400, d=30, K=450, tau0=1, alphaP=0.05, chiV=0, rho=8.9),
    "fe": dict(name="Fe terblokir 30 nm", type="fm", Msp=1700, d=30, K=48, tau0=1, alphaP=0.05, chiV=0, rho=7.87),
    "permalloy": dict(name="Permalloy NiFe serbuk lunak", type="fm", Msp=800, d=30, K=1, tau0=1, alphaP=0.02, chiV=0, rho=8.7),
    "ndfeb": dict(name="NdFeB serbuk magnet keras", type="fm", Msp=1280, d=1000, K=4300, tau0=1, alphaP=0.05, chiV=0, rho=7.5),
    "gd2o3": dict(name="Gd2O3 paramagnetik", type="para", Msp=0, d=10, K=0, tau0=1, alphaP=0.05, chiV=0.0136, rho=7.4),
    "au": dict(name="Au diamagnetik", type="dia", Msp=0, d=10, K=0, tau0=1, alphaP=0.05, chiV=-3.4e-5, rho=19.3),
    "zno": dict(name="ZnO diamagnetik lemah", type="dia", Msp=0, d=10, K=0, tau0=1, alphaP=0.05, chiV=-2.3e-5, rho=5.6),
}
SENS = {
    "nife": dict(name="NiFe permalloy", Ms=1.0, alpha=0.008, dH0=0.3, sigF=3.3),
    "cofeb": dict(name="CoFeB amorf", Ms=1.5, alpha=0.006, dH0=0.5, sigF=0.6),
    "cofesib": dict(name="CoFeSiB amorf", Ms=0.7, alpha=0.01, dH0=0.5, sigF=0.8),
    "finemet": dict(name="FeCuNbSiB nanokristalin", Ms=1.2, alpha=0.01, dH0=0.3, sigF=0.85),
}


def default_P() -> dict:
    """Default = resetAll() JS: sens nife, mat fe3o4, sapuan 70 mT, f=10 MHz, sigH=sigZ=0 (bawaan tanpa noise)."""
    s = SENS["nife"]
    P = dict(
        Ms=s["Ms"], alpha=s["alpha"], dH0=s["dH0"], sigF=s["sigF"], sigC=50.0, axis="long",
        Hk=3.0, sigHk=0.0, Hc=0.2, sigHc=0.0, edgeC=0.0, dwAmp=0.0,
        tf=1.0, tcu=2.0, len=5.0, wid=0.5, f=10.0, Rc=0.0, RcSkin=0.0, Lp=0.0,
        Hmax=70.0, Hfine=10.0, d1=1.0, d2=3.0, sigH=0.0, sigZ=0.0,
        pf=0.2, Ac=20.0, Cd=0.1, T=300.0,
        eta=1.0, dblSide=False, showSet=False,
        kcoilTheory=0.0, kcoil=0.0, kcoilTrue=0.0, Boff=0.0, sigHr=0.0,
        N=1000.0, Rcoil=6.0,
    )
    P["kcoilTheory"] = 0.7155 * MU0 * P["N"] / (P["Rcoil"] / 100) * 1000  # mT/A
    P["kcoil"] = P["kcoilTheory"]
    P["kcoilTrue"] = P["kcoil"]
    return P


def default_mat(key: str = "fe3o4") -> dict:
    m = copy.deepcopy(MATS[key])
    m.update(sigD=0.0, sigP=0.0)
    return m


# ---------------- Util kompleks (tiruan JS [re,im] + clamp) ----------------
def _clamp(x, a, b):
    return min(b, max(a, x))


def _clamp_result(r: dict, fn, bounds: list) -> dict:
    """Jepit x hasil Nelder-Mead ke bound fisik dan hitung ulang fun di titik terjepit.

    cost() menjepit parameter secara internal, jadi x mentah bisa keluar bound
    (mis. sigHk negatif) sementara fun dihitung di titik terjepit — keduanya
    lalu tidak konsisten. Fungsi ini mengembalikan keduanya dari titik yang sama.
    """
    xc = [_clamp(v, lo, hi) for v, (lo, hi) in zip(r["x"], bounds)]
    if xc == list(r["x"]):
        return dict(x=xc, fun=r["fun"])
    return dict(x=xc, fun=fn(xc))


def _round4(x: float) -> float:
    """Tiruan (+x.toFixed(4)) JS: pembulatan setengah ke atas, bukan bankir."""
    return math.floor(x * 1e4 + 0.5) / 1e4 if x >= 0 else -math.floor(-x * 1e4 + 0.5) / 1e4


def clamp_P(P: dict) -> dict:
    """Tiruan clamp getP() JS; dipakai di titik masuk measure/modelCurve/peakAt dkk."""
    P = dict(P)
    P["sigHk"] = max(P.get("sigHk") or 0, 0)
    P["sigHc"] = max(P.get("sigHc") or 0, 0)
    P["edgeC"] = _clamp(P.get("edgeC") or 0, 0, 1)
    P["dwAmp"] = max(P.get("dwAmp") or 0, 0)
    P["Hfine"] = min(P.get("Hfine") or 0, P.get("Hmax") or 0)
    P["RcSkin"] = max(P.get("RcSkin") or 0, 0)
    P["pf"] = _clamp(P.get("pf") or 0.2, 0.01, 0.64)
    P["Ac"] = max(P.get("Ac") or 20, 0.01)
    P["Cd"] = max(P.get("Cd") or 0, 0)
    P["T"] = max(P.get("T") or 300, 1)
    eta = P.get("eta")
    P["eta"] = 1.0 if eta is None or not math.isfinite(eta) else _clamp(eta, 0, 1)
    P["sigHr"] = max(P.get("sigHr") or 0, 0)
    return P


def clamp_mat(mat: dict) -> dict:
    """Tiruan clamp getMat() JS: d>=0.5, sigD/sigP>=0, K/tau0/alphaP/rho di atas nol."""
    m = dict(mat)
    m["d"] = max(m.get("d") or 0, 0.5)
    m["sigD"] = max(m.get("sigD") or 0, 0)
    m["sigP"] = max(m.get("sigP") or 0, 0)
    m["K"] = max(m.get("K") or 0, 1e-6)
    m["tau0"] = max(m.get("tau0") or 0, 1e-6)
    m["alphaP"] = max(m.get("alphaP") or 0, 1e-4)
    m["rho"] = max(m.get("rho") or 0, 0.1)
    return m


def csqrt(z: complex) -> complex:
    r = abs(z)
    re = math.sqrt((r + z.real) / 2.0)
    im = math.sqrt(max((r - z.real) / 2.0, 0.0))
    if z.imag < 0:
        im = -im
    return complex(re, im)


def ccosh(z: complex) -> complex:
    a = _clamp(z.real, -350.0, 350.0)
    b = z.imag
    return complex(math.cosh(a) * math.cos(b), math.sinh(a) * math.sin(b))


def csinh(z: complex) -> complex:
    a = _clamp(z.real, -350.0, 350.0)
    b = z.imag
    return complex(math.sinh(a) * math.cos(b), math.cosh(a) * math.sin(b))


def phi_norm(x: float) -> float:
    """Phi memakai math.erf (lebih akurat ~1e-16 vs aproksimasi JS ~1.5e-7)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def langevin(x: float) -> float:
    if abs(x) < 1e-3:
        return x / 3.0 - x ** 3 / 45.0
    return 1.0 / math.tanh(x) - 1.0 / x


# ---------------- RNG deterministik mulberry32 + Box-Muller (tiruan JS) ----------------
def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _s32(x: int) -> int:
    x = _u32(x)
    return x - 0x100000000 if x & 0x80000000 else x


def _imul(a: int, b: int) -> int:
    return _s32(_u32(a) * _u32(b))


class Mulberry32:
    def __init__(self, seed: int):
        self.a = _s32(seed)

    def random(self) -> float:
        self.a = _s32(self.a + 0x6D2B79F5)
        t = _imul(self.a ^ (_u32(self.a) >> 15), 1 | self.a)
        t = _s32(t + _imul(t ^ (_u32(t) >> 7), 61 | t) ^ t)
        return (_u32(t ^ (_u32(t) >> 14))) / 4294967296.0


class Randn:
    def __init__(self, seed: int):
        self.r = Mulberry32(seed)

    def __call__(self) -> float:
        u = 0.0
        while u == 0.0:
            u = self.r.random()
        v = self.r.random()
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


# ---------------- 1. LLG / Smit-Beljers ----------------
def layer_stiff(Hm: float, S: dict, s_ov=None) -> dict:
    Ms = S["Ms"]
    Hk0 = (S.get("Hk") or 0) * 1e-3
    Nw = S["tf"] / (S["tf"] + max(S["wid"] * 1000.0, 1e-6))
    Ny = (S.get("edgeC") or 0) * Nw
    Nz = 1.0 - Ny
    if S.get("axis", "long") == "long":
        qy = max(Hk0 + Ny * Ms, 1e-9)
        qz = Hk0 + Nz * Ms
    else:
        qy = Ny * Ms - Hk0
        qz = Nz * Ms
    h = Hm * 1e-3
    if qy > 0:
        s = s_ov if s_ov is not None else (1 if h >= 0 else -1)
        Hp = max(s * h, -0.95 * qy)
        return dict(A=qy + Hp, B=qz + Hp, c2=1.0, Hke=qy)
    Hk = max(-qy, 1e-9)
    H = abs(h)
    if H >= Hk:
        return dict(A=H - Hk, B=H + qz, c2=1.0, Hke=Hk)
    return dict(A=(Hk * Hk - H * H) / Hk, B=Hk + qz, c2=(H / Hk) ** 2, Hke=Hk)


def chi_layer(Hm: float, direction: str, f: float, S: dict, s_ov=None) -> complex:
    w = 2 * math.pi * f
    wg = w / GAM
    Ms = S["Ms"]
    st = layer_stiff(Hm, S, s_ov)
    A, B, c2, Hk = st["A"], st["B"], st["c2"], st["Hke"]
    G = max(S["alpha"] * wg + S["dH0"] * 1e-3 / 2.0, 1e-9)
    den = complex(A * B - G * G - wg * wg, G * (A + B))
    c = complex(Ms * B, Ms * G) / den
    cr, ci = c2 * c.real, c2 * c.imag
    dw = S.get("dwAmp") or 0
    if dw > 1e-9:
        Ast = max(A, 0.05 * Hk, 1e-9)
        u = f / 5e6
        chi_dw = (dw / 100.0) * (Ms / Ast) / (1 + u * u)
        cr += chi_dw
        ci -= chi_dw * u
    return complex(cr, ci)


# ---------------- 2. Maxwell trilayer ----------------
def z_one(Hm: float, direction: str, S: dict, f: float, s_ov=None) -> complex:
    if f == 0:  # JS: kf=0 dan jwm=0 -> NaN; hindari ZeroDivisionError
        return complex(float("nan"), float("nan"))
    w = 2 * math.pi * f
    chi = chi_layer(Hm, direction, f, S, s_ov)
    mu = 1.0 + chi
    sc = S["sigC"] * 1e6
    sf = S["sigF"] * 1e6
    kc = csqrt(complex(0, w * MU0 * sc))
    kf = csqrt(complex(0, w * MU0 * sf) * mu)
    dc = S["tcu"] * 1e-6 / 2.0
    tm = S["tf"] * 1e-6
    kcd = kc * dc
    kft = kf * tm
    A = ccosh(kcd)
    B = mu * kc * csinh(kcd) / kf
    Es = A * ccosh(kft) + B * csinh(kft)
    Hs0 = A * csinh(kft) + B * ccosh(kft)
    jwm = complex(-w * MU0 * mu.imag, w * MU0 * mu.real)
    Hs = (kf / jwm) * Hs0
    Zs = Es / Hs
    g = (S["len"] * 1e-3) / (2 * S["wid"] * 1e-3)
    return Zs * g


def z_sensor(Hm: float, direction: str, S: dict, f: float) -> complex:
    if S.get("axis") == "long" and (S.get("Hc") or 0) > 0:
        Hc = min(S["Hc"], 0.95 * S["Hk"])
        sg = max(0.3 * Hc, 0.03) + (S.get("sigHc") or 0)
        p = phi_norm((Hm - Hc) / sg) if direction == "up" else phi_norm((Hm + Hc) / sg)
        if p > 0.9995:
            return z_one(Hm, direction, S, f, 1)
        if p < 0.0005:
            return z_one(Hm, direction, S, f, -1)
        a = z_one(Hm, direction, S, f, 1)
        b = z_one(Hm, direction, S, f, -1)
        return p * a + (1 - p) * b
    return z_one(Hm, direction, S, f, None)


def z_sensor_avg(Hm: float, direction: str, S: dict, f: float) -> complex:
    sHK = S.get("sigHk") or 0
    if not (sHK > 1e-9):
        return z_sensor(Hm, direction, S, f)
    re = im = 0.0
    for i in range(7):
        Hki = max(S["Hk"] + sHK * GH_X[i], 0.05)
        Si = dict(S, Hk=Hki)
        z = z_sensor(Hm, direction, Si, f)
        re += GH_W[i] * z.real
        im += GH_W[i] * z.imag
    return complex(re, im)


def rdc(S: dict) -> float:
    return (S["len"] * 1e-3) / (S["wid"] * 1e-3 * (S["sigC"] * 1e6 * S["tcu"] * 1e-6 + 2 * S["sigF"] * 1e6 * S["tf"] * 1e-6))


# ---------------- 3. Nanopartikel ----------------
def Nin(tp: float, Ac: float) -> float:  # tp cm, Ac cm^2
    a = math.sqrt(Ac / math.pi)
    e = (tp / 2.0) / a
    if e >= 1:
        return 1.0 / 3.0
    if e <= 0:
        return 0.0
    Nz = (1.0 / (1 - e * e)) * (1 - e * math.acos(e) / math.sqrt(1 - e * e))
    return (1 - Nz) / 2.0


def particle_resp(Bapp: float, direction: str, mat: dict, f: float, T: float) -> dict:
    w = 2 * math.pi * f
    t = mat["type"]
    Ms = mat["Msp"] * 1e3
    if t == "none":
        return dict(Mp=0.0, s=0.0, chi=complex(0, 0))
    if t in ("dia", "para"):
        return dict(Mp=mat["chiV"] * Bapp, s=0.0, chi=complex(mat["chiV"], 0))
    V = math.pi * (mat["d"] * 1e-9) ** 3 / 6.0
    K = mat["K"] * 1e3
    sP = mat.get("sigP") or 0
    eddy_im = 0.0
    if sP > 0:
        dd = max(mat["d"], 0.5) * 1e-9
        eddy_im = -MU0 * (sP * 1e6) * w * dd * dd / 40.0
    if t == "sp":
        m = Ms * V
        xi = m * abs(Bapp) / (KB * T)
        L = langevin(xi) * (1 if (Bapp or 1) >= 0 else -1)
        chi0 = MU0 * Ms * m / (3 * KB * T) if xi < 1e-6 else MU0 * Ms * abs(L) / abs(Bapp)
        tau = mat["tau0"] * 1e-9 * math.exp(min(K * V / (KB * T), 60.0))
        wt = w * tau
        den = 1 + wt * wt
        return dict(Mp=MU0 * Ms * L, s=abs(L), chi=complex(chi0 / den, -chi0 * wt / den + eddy_im))
    chi_i = MU0 * Ms * Ms / (3 * K)
    BK = 2 * K / Ms
    Bc = 0.479 * BK
    sh = Bapp - (1 if direction == "up" else -1) * Bc
    mbar = math.tanh(chi_i * sh / (MU0 * Ms))
    chiH = chi_i / (1 + abs(Bapp) / (1.5 * BK))
    wr = GAM * (BK + abs(Bapp))
    x = w / wr
    c = complex(chiH, 0) / complex(1 - x * x, 2 * mat["alphaP"] * x)
    return dict(Mp=MU0 * Ms * mbar, s=math.sqrt(mbar ** 2 + (1 - mbar ** 2) / 3.0),
                chi=complex(c.real, c.imag + eddy_im))


def particle_avg(Bapp: float, direction: str, mat: dict, f: float, T: float) -> dict:
    sD = mat.get("sigD") or 0
    if not (sD > 1e-9):
        return particle_resp(Bapp, direction, mat, f, T)
    dd, ww = [], []
    wsum = 0.0
    for i in range(7):
        di = max(mat["d"] * math.exp(sD * GH_X[i] - 0.5 * sD * sD), 0.5)
        dd.append(di)
        wi = GH_W[i] * di ** 3
        ww.append(wi)
        wsum += wi
    Mp = s = cr = ci = 0.0
    for i in range(7):
        wv = ww[i] / wsum
        pr = particle_resp(Bapp, direction, dict(mat, d=dd[i]), f, T)
        Mp += wv * pr["Mp"]
        s += wv * pr["s"]
        cr += wv * pr["chi"].real
        ci += wv * pr["chi"].imag
    return dict(Mp=Mp, s=s, chi=complex(cr, ci))


def layer_geom(P: dict, mat: dict, mass: float) -> dict:
    if mass <= 0 or mat["type"] == "none":
        return dict(tp=0.0, teff=0.0, N=0.0, lam=0.0)
    tp = mass / (mat["rho"] * P["pf"] * P["Ac"])  # cm
    lam = (P["wid"] * 1e-3 / math.pi) * 100  # cm
    teff = lam * (1 - math.exp(-tp / lam))
    return dict(tp=tp, teff=teff, N=Nin(tp, P["Ac"]), lam=lam)


# ---------------- 4. Impedansi total ----------------
def total_z(Hm: float, direction: str, P: dict, mat: dict, mass: float) -> dict:
    f = P["f"] * 1e6
    w = 2 * math.pi * f
    Bs = sig = 0.0
    dZ = complex(0, 0)
    if mass > 0 and mat["type"] != "none":
        g = layer_geom(P, mat, mass)
        pr = particle_avg(Hm * 1e-3, direction, mat, f, P["T"])
        eta = 1.0 if P.get("eta") is None else P["eta"]
        nS = 2 if P.get("dblSide") else 1
        Bs = -nS * eta * g["N"] * P["pf"] * pr["Mp"]
        sig = math.sqrt(nS) * eta * P["Cd"] * P["pf"] * MU0 * (mat.get("Msp") or 0) * 1e3 * pr["s"]
        mup = 1.0 + pr["chi"]
        q = (mup - 1) / (mup + 2)
        fq = P["pf"] * q
        muL = (1 + 2 * fq) / (1 - fq)
        k = w * MU0 * g["teff"] * 1e-2 * (P["len"] * 1e-3) / (2 * P["wid"] * 1e-3)
        dZ = complex(-k * muL.imag, k * (muL.real - 1)) * eta * nS
    Hc = Hm + Bs * 1e3
    if sig * 1e3 < 0.01:
        z = z_sensor_avg(Hc, direction, P, f)
    else:
        re = im = 0.0
        for i in range(7):
            z = z_sensor_avg(Hc + sig * 1e3 * GH_X[i], direction, P, f)
            re += GH_W[i] * z.real
            im += GH_W[i] * z.imag
        z = complex(re, im)
    RcEff = (P.get("Rc") or 0) + (P.get("RcSkin") or 0) * math.sqrt(max(P["f"], 0))
    re = z.real + dZ.real + RcEff
    im = z.imag + dZ.imag + w * P.get("Lp", 0) * 1e-9
    return dict(re=re, im=im, abs=math.hypot(re, im))


def k_ratio_of(P: dict) -> float:
    return (P["kcoilTrue"] / P["kcoil"]) if (P["kcoil"] > 0 and P["kcoilTrue"] > 0) else 1.0


def hend_act(P: dict) -> float:
    return P["Hmax"] * k_ratio_of(P) + (P.get("Boff") or 0)


def ref_z(P: dict, mat: dict, mass: float) -> float:
    return total_z(hend_act(P), "up", P, mat, mass)["abs"]


# ---------------- 5. Sapuan / ukur / analisis ----------------
def setpoints(P: dict) -> list:
    pos = {0.0}
    h = P["d1"]
    while h <= P["Hfine"] + 1e-9:
        pos.add(_round4(h))
        h += P["d1"]
    start = math.ceil((P["Hfine"] + 1e-9) / P["d1"]) * P["d1"]
    h = max(start, P["Hfine"]) + P["d2"]
    while h < P["Hmax"] - 1e-9:
        pos.add(_round4(h))
        h += P["d2"]
    pos.add(P["Hmax"])
    p = sorted(pos)
    neg = [-v for v in p if v > 0]
    return sorted(neg + p)


def measure(P: dict, masses: list, mat: dict, seed: int) -> list:
    P = clamp_P(P)
    mat = clamp_mat(mat)
    sp = setpoints(P)
    kRatio = P["kcoilTrue"] / P["kcoil"] if P["kcoil"] > 0 else 1.0
    out = []
    for ri, mass in enumerate(masses):
        rnd = Randn(seed + ri * 7919)
        zref = ref_z(P, mat, mass) * (1 + P["sigZ"] / 100 * rnd())
        pts = []
        for direction in ("up", "down"):
            arr = sp if direction == "up" else list(reversed(sp))
            for h in arr:
                Icmd = h / P["kcoil"] if P["kcoil"] > 0 else float("nan")
                Hact = h * kRatio + P.get("Boff", 0) + P["sigH"] * rnd()
                Iact = (Hact - P.get("Boff", 0)) / P["kcoilTrue"] if P["kcoilTrue"] > 0 else float("nan")
                z = total_z(Hact, direction, P, mat, mass)
                zm = z["abs"] * (1 + P["sigZ"] / 100 * rnd())
                Hread = Hact + P["sigHr"] * rnd()
                pts.append(dict(H=Hread, Hset=h, Hact=Hact, dir=direction, I=Icmd, Iact=Iact,
                                z=zm, mir=100 * (zm - zref) / zref))
        out.append(dict(mass=mass, pts=pts, zref=zref))
    return out


def model_curve(P: dict, mat: dict, mass: float, direction: str = "up", step: float = 0.25) -> list:
    P = clamp_P(P)
    mat = clamp_mat(mat)
    zref = ref_z(P, mat, mass)
    out = []
    h = -P["Hmax"]
    while h <= P["Hmax"] + 1e-9:
        z = total_z(h, direction, P, mat, mass)
        out.append(dict(x=h, y=100 * (z["abs"] - zref) / zref, re=z["re"], im=z["im"], abs=z["abs"]))
        h += step
    return out


def peak_info(curve: list) -> dict:
    ip = max(range(len(curve)), key=lambda i: curve[i]["y"])
    peak = curve[ip]["y"]
    half = peak / 2
    l, r = curve[0]["x"], curve[-1]["x"]
    for i in range(ip, 0, -1):
        if curve[i - 1]["y"] <= half:
            den = curve[i]["y"] - curve[i - 1]["y"]
            t = (half - curve[i - 1]["y"]) / den if den else float("nan")  # JS: 0/0 -> NaN
            l = curve[i - 1]["x"] + t * (curve[i]["x"] - curve[i - 1]["x"])
            break
    for i in range(ip, len(curve) - 1):
        if curve[i + 1]["y"] <= half:
            den = curve[i]["y"] - curve[i + 1]["y"]
            t = (half - curve[i + 1]["y"]) / den if den else float("nan")
            r = curve[i + 1]["x"] + t * (curve[i]["x"] - curve[i + 1]["x"])
            break
    return dict(peak=peak, pos=curve[ip]["x"], fwhm=r - l)


def peak_at(P: dict, mat: dict, mass: float) -> float:
    P = clamp_P(P)
    mat = clamp_mat(mat)
    zref = ref_z(P, mat, mass)
    lim = min(P["Hmax"], max(10.0, 3 * P["Hk"] + 5))

    def ev(h):
        return 100 * (total_z(h, "up", P, mat, mass)["abs"] - zref) / zref

    bh, bv = 0.0, -1e18
    h = -lim
    while h <= lim + 1e-9:
        v = ev(h)
        if v > bv:
            bv, bh = v, h
        h += 0.5
    h = bh - 0.5
    while h <= bh + 0.5 + 1e-9:
        v = ev(h)
        if v > bv:
            bv = v
        h += 0.02
    return bv


def slope_at(curve: list, i: int) -> float:
    n = len(curve)

    def g(j):
        return curve[min(max(j, 0), n - 1)]

    if n < 5:
        a, b = g(i - 1), g(i + 1)
        return (b["y"] - a["y"]) / ((b["x"] - a["x"]) or 1)
    c0, c1, c3, c4 = g(i - 2), g(i - 1), g(i + 1), g(i + 2)
    d = (-c4["y"] + 8 * c3["y"] - 8 * c1["y"] + c0["y"]) / 12.0
    h = (c4["x"] - c0["x"]) / 4.0 or 1.0
    return d / h


def sigma_mir(dmdh: float, m: float, P: dict, axis: str = "read") -> float:
    sigH = (P.get("sigH") or 0) if axis == "set" else (P.get("sigHr") or 0)
    sZ = (P.get("sigZ") or 0) * math.sqrt(2) * (1 + m / 100)
    return max(math.hypot(abs(dmdh) * sigH, sZ), 1e-6)


def sensitivity_lod(P: dict, mat: dict, peak0: float | None = None) -> dict:
    P = clamp_P(P)
    mat = clamp_mat(mat)
    lam_cm = (P["wid"] * 1e-3 / math.pi) * 100
    dm = 0.02 if mat["type"] == "none" else max(1e-4, 0.05 * lam_cm * mat["rho"] * P["pf"] * P["Ac"])
    S = (peak_at(P, mat, dm) - peak_at(P, mat, 0.0)) / dm
    if peak0 is None:
        peak0 = peak_info(model_curve(P, mat, 0.0, "up", 0.1))["peak"]
    sig_pk = (1 + peak0 / 100) * P["sigZ"] * math.sqrt(2)
    lod = 3 * sig_pk / abs(S) if abs(S) > 1e-9 else float("inf")
    return dict(dm=dm, S=S, sig_pk=sig_pk, lod=lod)


# ---------------- 6. Utilitas analisis tambahan (tiruan fungsi non-UI JS) ----------------
def interp_model(curve: list, x: float) -> float:
    """Tiruan interpModel JS: interpolasi linear kurva model di H=x."""
    if not curve:
        return float("nan")
    if x <= curve[0]["x"]:
        return curve[0]["y"]
    if x >= curve[-1]["x"]:
        return curve[-1]["y"]
    for i in range(1, len(curve)):
        if x <= curve[i]["x"]:
            a, b = curve[i - 1], curve[i]
            t = (x - a["x"]) / ((b["x"] - a["x"]) or 1)
            return a["y"] + t * (b["y"] - a["y"])
    return float("nan")


def model_mir(P: dict, mat: dict, mass: float, H: float) -> float:
    """Tiruan modelMIR JS: MIR model tanpa noise di H, rata-rata sapuan naik+turun."""
    P = clamp_P(P)
    mat = clamp_mat(mat)
    zref = ref_z(P, mat, mass)
    zu = total_z(H, "up", P, mat, mass)["abs"]
    zd = total_z(H, "down", P, mat, mass)["abs"]
    return 100 * (0.5 * (zu + zd) - zref) / zref


def model_band(curve: list, P: dict, axis: str = "read") -> dict:
    """Tiruan modelBand JS: pita +-1 sigma dari noise Z + propagasi noise medan."""
    lo, hi = [], []
    for i, c in enumerate(curve):
        s = sigma_mir(slope_at(curve, i), c["y"], P, axis)
        lo.append(dict(x=c["x"], y=c["y"] - s))
        hi.append(dict(x=c["x"], y=c["y"] + s))
    return dict(lo=lo, hi=hi)


def _split_csv_line(ln: str) -> list:
    """Pisah baris CSV di luar tanda kutip; pemisah koma, titik-koma, atau tab."""
    cols, cur, q = [], [], False
    for ch in ln:
        if ch == '"':
            q = not q
            cur.append(ch)
        elif not q and ch in ",;\t":
            cols.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cols.append("".join(cur))
    return cols


_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _num_js(s) -> float:
    """Tiruan parseFloat JS plus dukungan desimal koma Indonesia."""
    t = str(s).strip().strip('"').strip("'")
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    elif "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    m = _NUM_RE.match(t.lstrip())
    return float(m.group(0)) if m else float("nan")


def parse_exp_csv(text: str) -> tuple:
    """Tiruan parseExpCsv JS (mapping 10/8/3/2 kolom). Kembalikan tuple (rows, bad)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return [], 0
    has_head = any(k in lines[0].lower() for k in ("mir", "h_", "sampel", "h ("))
    body = lines[1:] if has_head else lines
    out, bad = [], 0
    for ln in body:
        cc = [c.strip().strip('"') for c in _split_csv_line(ln)]
        if len(cc) < 2 or all(c == "" for c in cc):
            bad += 1
            continue
        H = mir = float("nan")
        label = "eksperimen"
        if len(cc) >= 10:
            H, mir, label = _num_js(cc[3]), _num_js(cc[9]), cc[0] or "eksperimen"
            if not math.isfinite(H):
                H = _num_js(cc[4])
        elif len(cc) >= 8:
            H, mir, label = _num_js(cc[3]), _num_js(cc[7]), cc[0] or "eksperimen"
            if not math.isfinite(H):
                H = _num_js(cc[4])
        elif len(cc) >= 3:
            label = cc[0] or "eksperimen"
            H, mir = _num_js(cc[1]), _num_js(cc[2])
            if not (math.isfinite(H) and math.isfinite(mir)):
                H, mir, label = _num_js(cc[0]), _num_js(cc[1]), "eksperimen"
        else:
            H, mir = _num_js(cc[0]), _num_js(cc[1])
        if math.isfinite(H) and math.isfinite(mir):
            out.append(dict(label=label, H=H, mir=mir))
        else:
            bad += 1
    return out, bad


def nelder_mead(fn, x0: list, step: list, n_max: int = 80) -> dict:
    """Tiruan nelderMead JS generik. Kembalikan dict(x, fun)."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += (step[i] if i < len(step) else None) or 1e-6
        simplex.append(p)
    val = [fn(p) for p in simplex]
    for _ in range(n_max or 80):
        order = sorted(range(n + 1), key=lambda i: val[i])
        simplex = [simplex[i] for i in order]
        val = [val[i] for i in order]
        best, worst = simplex[0], simplex[n]
        cent = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        refl = [c + (c - worst[j]) for j, c in enumerate(cent)]
        fr = fn(refl)
        if fr < val[0]:
            expand = [c + 2.0 * (refl[j] - c) for j, c in enumerate(cent)]
            fe = fn(expand)
            simplex[n], val[n] = (expand, fe) if fe < fr else (refl, fr)
        elif fr < val[n - 1]:
            simplex[n], val[n] = refl, fr
        else:
            cont = [c + 0.5 * (worst[j] - c) for j, c in enumerate(cent)]
            fc = fn(cont)
            if fc < val[n]:
                simplex[n], val[n] = cont, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j]) for j in range(n)]
                    val[i] = fn(simplex[i])
        if max(abs(val[i] - val[0]) for i in range(1, n + 1)) < 1e-7:
            break
    bi = val.index(min(val))
    return dict(x=simplex[bi], fun=val[bi])


def exp_cost(P: dict, mat: dict, label: str, mass: float, n_param: int, exp_data: list) -> float:
    """Tiruan expCost JS: chi2/DoF model tanpa noise vs data satu label (sumbu H_terbaca)."""
    arr = [d for d in exp_data if d["label"] == label]
    if len(arr) < 4:
        return float("nan")
    P = clamp_P(P)
    mat = clamp_mat(mat)
    curve = model_curve(P, mat, mass, "up", 0.25)
    s2, n = 0.0, 0
    for d in arr:
        m = model_mir(P, mat, mass, d["H"])
        if not math.isfinite(m):
            continue
        fi = next((i for i, p in enumerate(curve) if p["x"] >= d["H"]), -1)
        ii = len(curve) - 1 if fi < 0 else max(0, fi - 1)
        dmdh = slope_at(curve, ii)
        sig = sigma_mir(dmdh if math.isfinite(dmdh) else 0, m, P, "read")
        r = d["mir"] - m
        s2 += (r / sig) ** 2
        n += 1
    if not n:
        return float("nan")
    return s2 / max(n - (n_param or 0), 1)


def fit_sensor(P0: dict, mat: dict, label: str, exp_data: list, mass: float = 0.0, n_max: int = 60) -> dict:
    """Tiruan fitSensor JS: fit Hk dan sigHk ke satu label; mass = massa acuan label itu."""

    def cost(x):
        P = dict(P0, Hk=_clamp(x[0], 0.2, 40), sigHk=_clamp(x[1], 0, 5))
        v = exp_cost(P, mat, label, mass, 2, exp_data)
        return v if math.isfinite(v) else 1e12

    r = nelder_mead(cost, [P0.get("Hk") or 3, P0.get("sigHk") or 0], [0.5, 0.3], n_max)
    return _clamp_result(r, cost, [(0.2, 40), (0, 5)])


def fit_cd(P0: dict, mat: dict, exp_data: list, rows: list, row_peaks: list | None = None, n_max: int = 40) -> dict:
    """Tiruan fitCd JS: fit Cd global. rows = daftar dict(label, mass); row_peaks = puncak terukur per baris untuk auto-match label."""
    by_exp = {}
    for d in exp_data:
        by_exp.setdefault(d["label"], []).append(d)
    can_match = bool(row_peaks and len(row_peaks) == len(rows))
    unmatched = sorted(lb for lb in by_exp if not any(r["label"] == lb for r in rows))
    if unmatched and not can_match:
        warnings.warn(
            "fit_cd: label eksperimen tanpa pasangan massa di `rows` "
            f"({', '.join(unmatched)}); memakai massa 0.0. "
            "Label `rows` yang tersedia: "
            f"{', '.join(sorted(r['label'] for r in rows)) or '-'}.",
            UserWarning, stacklevel=2,
        )

    def mass_of(lb):
        same = next((r for r in rows if r["label"] == lb), None)
        if same is not None:
            return same["mass"]
        if row_peaks and len(row_peaks) == len(rows):
            pk_e = max(d["mir"] for d in by_exp[lb])
            bd, best = 1e18, 0.0
            for r, pk in zip(rows, row_peaks):
                dd = abs(pk - pk_e)
                if dd < bd:
                    bd, best = dd, r["mass"]
            return best
        # Fallback 0.0 hanya bila auto-match tak mungkin; sudah diperingatkan sekali di atas.
        return 0.0

    def cost(x):
        P = dict(P0, Cd=_clamp(x[0], 0, 1))
        s, n = 0.0, 0
        for lb, arr in by_exp.items():
            v = exp_cost(P, mat, lb, mass_of(lb), 1, exp_data)
            if math.isfinite(v):
                s += v * len(arr)
                n += len(arr)
        return s / n if n else 1e12

    r = nelder_mead(cost, [P0.get("Cd") or 0.1], [0.1], n_max)
    return _clamp_result(r, cost, [(0, 1)])


def auto_label(mass: float) -> str:
    """Tiruan autoLabel JS."""
    if mass <= 0:
        return "Tanpa sampel"
    return f"{mass:.3f}".rstrip("0").rstrip(".") + " gram"


def measure_csv(P: dict, mat: dict, masses: list, seed: int = 20261012) -> str:
    """Tiruan unduhan CSV pengukuran JS (dlCsv) sebagai string."""
    P = clamp_P(P)
    mat = clamp_mat(mat)
    meas = measure(P, masses, mat, seed)
    lines = [
        "# kcoil_teori_mTperA=%s,kcoil_cmd_mTperA=%s,kcoil_true_mTperA=%s,Boff_mT=%s,sigH_mT=%s,sigHr_mT=%s,showSet=%s"
        % (P.get("kcoilTheory", 0), P.get("kcoil", 0), P.get("kcoilTrue", 0), P.get("Boff") or 0,
           P.get("sigH") or 0, P.get("sigHr") or 0, str(bool(P.get("showSet"))).lower()),
        "sampel,massa_g,arah,H_terbaca_mT,H_atur_mT,H_aktual_mT,I_cmd_A,I_aktual_A,Z_ohm,MIR_persen",
    ]
    for r in meas:
        label = auto_label(r["mass"]).replace('"', '""')
        for p in r["pts"]:
            iact = p["Iact"]
            if iact != iact or iact == 0:
                iact = p["I"]
            lines.append('"%s",%s,%s,%.4f,%.4f,%.4f,%.5f,%.5f,%.6f,%.5f' % (
                label, r["mass"], "naik" if p["dir"] == "up" else "turun",
                p["H"], p["Hset"], p["Hact"], p["I"], iact, p["z"], p["mir"]))
    return "\n".join(lines) + "\n"


# ---------------- Uji mandiri (14 uji, tiruan selfTest JS) ----------------
def self_test(P0: dict | None = None) -> list:
    P0 = P0 or default_P()
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    def ex(v):
        return f"{v:.3e}"

    base = dict(P0, Hc=0, sigHk=0, sigHc=0, edgeC=0, dwAmp=0, Rc=0, RcSkin=0, Lp=0, axis="long")
    try:
        z = z_one(0, "up", base, 100)
        R = rdc(base)
        add("Batas DC: Z(100 Hz) -> Rdc", rel(z.real, R) < 1e-3 and abs(z.imag) < 1e-3 * R,
            f"Re={z.real:.6g} Rdc={R:.6g} Im/Re={ex(z.imag / z.real)}")
        sg = 10
        S = dict(base, Ms=0, sigF=sg, sigC=sg)
        fq, w = 50e6, 2 * math.pi * 50e6
        kk = csqrt(complex(0, w * MU0 * sg * 1e6))
        tt = (S["tcu"] / 2 + S["tf"]) * 1e-6
        kt = kk * tt
        zex = rdc(S) * kt * ccosh(kt) / csinh(kt)
        z = z_one(0, "up", S, fq)
        err = abs(z - zex) / abs(zex)
        add("Batas homogen: Z = Rdc k t coth(k t)", err < 1e-9, f"galat={ex(err)}")
        a, b = z_one(2.5, "up", base, 20e6), z_one(-2.5, "up", base, 20e6)
        add("Simetri Z(H)=Z(-H)", abs(a - b) / abs(a) < 1e-12, f"galat={ex(abs(a-b)/abs(a))}")
        S = dict(base, Hc=0.2)
        u, d = z_sensor(0.3, "up", S, 20e6), z_sensor(-0.3, "down", S, 20e6)
        add("Pembalikan waktu Z_up(H)=Z_down(-H)", abs(u - d) / abs(u) < 1e-9, f"galat={ex(abs(u-d)/abs(u))}")
        S = dict(base, edgeC=1, dH0=0, alpha=0)
        c = chi_layer(0, "up", 1e3, S)
        Nw = S["tf"] / (S["tf"] + S["wid"] * 1000)
        ch = S["Ms"] / (S["Hk"] * 1e-3 + Nw * S["Ms"])
        add("Koreksi tepi: chi(0)=Ms/(Hk+Nw Ms)", rel(c.real, ch) < 1e-4, f"chi={c.real:.3f} teori={ch:.3f}")
        Nw = S["tf"] / (S["tf"] + S["wid"] * 1000)
        Hk = S["Hk"] * 1e-3
        fr = GAM / (2 * math.pi) * math.sqrt((Hk + Nw * S["Ms"]) * (Hk + (1 - Nw) * S["Ms"]))
        lo = chi_layer(0, "up", 0.999 * fr, S).real
        hi = chi_layer(0, "up", 1.001 * fr, S).real
        add("Kittel: tanda chi' balik di fr", lo > 0 and hi < 0, f"fr={fr/1e9:.4f} GHz lo={lo:.1f} hi={hi:.1f}")
        S = dict(base, axis="trans", edgeC=20, dH0=0, alpha=0)
        Nw = S["tf"] / (S["tf"] + S["wid"] * 1000)
        D = 20 * Nw * S["Ms"] - S["Hk"] * 1e-3
        c = chi_layer(0, "up", 1e3, S).real
        add("Tepi melampaui Hk transversal: sumbu mudah berbalik", D > 0 and rel(c, S["Ms"] / D) < 1e-4,
            f"chi={c:.3f} teori={S['Ms']/D:.3f}")
        c0 = chi_layer(0, "up", 5e6, base)
        c1 = chi_layer(0, "up", 5e6, dict(base, dwAmp=10))
        dr, di = c1.real - c0.real, c1.imag - c0.imag
        add("Domain-wall Debye kausal", dr > 0 and abs(dr + di) < 1e-6 * dr, f"dRe={dr:.4f} dIm={di:.4f}")
        a = math.sqrt(20 / math.pi)
        n0, n1 = Nin(1e-9, 20), Nin(2 * 0.999 * a, 20)
        add("Nin film tipis -> 0", n0 < 1e-6, f"Nin={ex(n0)}")
        add("Nin kontinu ke bola 1/3", abs(n1 - 1 / 3) < 2e-3, f"Nin={n1:.5f}")
        m1 = dict(type="sp", Msp=400, d=1000, K=11, tau0=1, alphaP=0.05, chiV=0, rho=5, sigP=10, sigD=0)
        m0 = dict(m1, sigP=0)
        fq, w = 5e7, 2 * math.pi * 5e7
        dIm = particle_resp(0.01, "up", m1, fq, 300)["chi"].imag - particle_resp(0.01, "up", m0, fq, 300)["chi"].imag
        ex1 = -MU0 * 10e6 * w * 1e-12 / 40
        add("Eddy: chi''=-mu0 s w d^2/40", rel(dIm, ex1) < 1e-6, f"hitung={ex(dIm)} teori={ex(ex1)}")
        fm = dict(type="fm", Msp=350, d=20, K=200, tau0=1, alphaP=0.05, chiV=0, rho=5.3, sigP=0, sigD=0)
        Ms, K = 350e3, 200e3
        chiI = MU0 * Ms * Ms / (3 * K)
        c0 = particle_resp(0, "up", fm, 1e3, 300)["chi"].real
        add("chi terblokir B=0 = chi_i", rel(c0, chiI) < 1e-6, f"chi={c0:.5f} chi_i={chiI:.5f}")
        BK = 2 * K / Ms
        B = 100 * BK
        cH = particle_resp(B, "up", fm, 1e3, 300)["chi"].real
        r = cH * B / (MU0 * Ms)
        add("chi transversal terblokir menuju mu0 Ms/B medan tinggi", 0.98 < r < 1, f"chi B/(mu0 Ms) = {r:.4f}")
        sp = dict(type="sp", Msp=400, d=10, K=11, tau0=1, alphaP=0.05, chiV=0, rho=5, sigP=0, sigD=1e-6)
        a = particle_avg(0.01, "up", sp, 5e7, 300)
        b = particle_resp(0.01, "up", dict(sp, sigD=0), 5e7, 300)
        add("sigD->0 kembali monodispers", rel(a["Mp"], b["Mp"]) < 1e-6 and rel(a["chi"].real, b["chi"].real) < 1e-6,
            f"galat Mp={ex(rel(a['Mp'], b['Mp']))}")
    except Exception as e:  # noqa: BLE001
        add("Pengecualian saat uji", False, str(e))
    return out


# ---------------- CLI ----------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Port Python simulator magnetoimpedansi (satu file).")
    ap.add_argument("--masses", default="0,0.1,0.3", help="massa gram, koma-pisah")
    ap.add_argument("--mat", default="fe3o4", choices=sorted(MATS), help="preset partikel")
    ap.add_argument("--step", type=float, default=0.5, help="langkah kurva model (mT)")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--csv", default="", help="tulis kurva model ke CSV")
    ap.add_argument("--meas-csv", default="", help="tulis CSV pengukuran (mirip unduhan HTML)")
    ap.add_argument("--plot", action="store_true", help="simpan grafik PNG")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)
    if not (0 < args.step < math.inf):
        print(f"--step harus angka positif (mT); diberi {args.step}.", file=sys.stderr)
        return 2

    P = default_P()
    mat = default_mat(args.mat)
    masses = [float(s) for s in args.masses.split(",") if s.strip() != ""]
    if not masses:
        print("Daftar --masses kosong; beri minimal satu massa gram.", file=sys.stderr)
        return 2

    if not args.skip_test:
        print("== Uji mandiri ==")
        ok_all = True
        for name, ok, detail in self_test(P):
            print(f"[{'OK' if ok else 'GAGAL'}] {name} -- {detail}")
            ok_all = ok_all and ok
        if not ok_all:
            print("PERINGATAN: ada uji gagal, lanjut dengan hati-hati.")
    print(f"\n== Parameter: {mat['name']}, f={P['f']} MHz, Hmax={P['Hmax']} mT, sumbu={P['axis']} ==")
    print(f"Rdc = {rdc(P)*1000:.2f} mohm, kcoil = {P['kcoil']:.3f} mT/A")

    curves = [model_curve(P, mat, m, "up", args.step) for m in masses]
    for m, c in zip(masses, curves):
        info = peak_info(c)
        g = layer_geom(P, mat, m)
        print(f"massa {m:>6.3f} g: puncak {info['peak']:.3f} % @ {info['pos']:.2f} mT, "
              f"FWHM {info['fwhm']:.2f} mT, tp {g['tp']*1e4:.2f} um, teff {g['teff']*1e4:.2f} um")

    sl = sensitivity_lod(P, mat)
    print(f"\nSensitivitas awal S = {sl['S']:.3f} %/g (dm={sl['dm']:.5f} g)")
    print(f"LOD = {sl['lod']*1000:.3f} mg  (3*sigma/|S|, sigma_pk={sl['sig_pk']:.4f} %)")
    z0 = total_z(0.0, "up", P, mat, masses[-1] if masses else 0.0)
    print(f"Contoh Z(0) massa terakhir: {z0['re']:.4f} + j{z0['im']:.4f} ohm (|Z|={z0['abs']:.4f})")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["H_mT"] + [f"MIR_m{mass:g}g_pct" for mass in masses])
            for i in range(len(curves[0])):
                w.writerow([f"{curves[0][i]['x']:.4f}"] + [f"{c[i]['y']:.6f}" for c in curves])
        print(f"CSV ditulis: {args.csv}")

    if args.meas_csv:
        with open(args.meas_csv, "w", newline="", encoding="utf-8") as fh:
            fh.write(measure_csv(P, mat, masses, args.seed))
        print(f"CSV pengukuran ditulis: {args.meas_csv}")

    rc = 0
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for m, c in zip(masses, curves):
                ax.plot([p["x"] for p in c], [p["y"] for p in c], label=f"{m:g} g")
            ax.set_xlabel("H (mT)")
            ax.set_ylabel("MIR (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png = "mir_model.png"
            fig.savefig(png, dpi=130)
            print(f"Plot ditulis: {png}")
        except ImportError:
            print("matplotlib tidak ada; plot gagal.", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
