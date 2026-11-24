"""Simulador 2 de Física IV (ITBA): estados estacionarios del 'quantum bouncer'.

Sistema: U(x) = +infinito para x <= 0 y U(x) = alpha*x para x > 0.
Unidades de entrada y gráficos: masa en m_e, alpha en eV/nm, x en nm,
energía en eV, psi en nm**(-1/2), rho en nm**(-1).
Incertidumbres: Delta p en kg*m/s; Delta x Delta p en J*s.

La solución usa exclusivamente funciones de Airy, no diferencias finitas.
Las cuadraturas numéricas emplean una variable adimensional y un dominio
semiinfinito real, independiente de la discretización/zoom del dibujo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import e, isfinite, sqrt
from typing import Any

import numpy as np
from scipy.constants import electron_mass, hbar, physical_constants
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ai_zeros, airy

EV_J = physical_constants["electron volt-joule relationship"][0]
NM_M = 1e-9
HBAR_SI = float(hbar)
M_E = float(electron_mass)
# hbar^2/(2*m_e) expresado en eV*nm^2.
C_ELECTRON = HBAR_SI**2 / (2.0 * M_E * EV_J * NM_M**2)
LIMITE_N = 30
COLORES = ("#1d5fa9", "#19844b", "#7b48a5", "#c23d35")


class ErrorFisico(ValueError):
    """Parámetros imposibles o fuera de los límites didácticos elegidos."""


@dataclass(frozen=True)
class Config:
    masa: float = 1.0           # unidades de masa electrónica
    pendiente: float = 1.0      # eV/nm
    n: int = 1
    comparaciones: tuple[int, ...] = ()

    def validar(self) -> None:
        for nombre in ("masa", "pendiente"):
            v = getattr(self, nombre)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v):
                raise ErrorFisico(f"{nombre}: ingresá un número real finito.")
        if not 0.1 <= self.masa <= 10.0:
            raise ErrorFisico("La masa debe estar entre 0,1 y 10 mₑ.")
        if not 0.01 <= self.pendiente <= 10.0:
            raise ErrorFisico("La pendiente debe estar entre 0,01 y 10 eV/nm.")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or not 1 <= self.n <= LIMITE_N:
            raise ErrorFisico("n debe ser un entero entre 1 y 30.")
        if not isinstance(self.comparaciones, tuple) or len(self.comparaciones) > 3:
            raise ErrorFisico("Se permiten hasta tres estados adicionales (cuatro en total).")
        if len(set(self.comparaciones)) != len(self.comparaciones):
            raise ErrorFisico("No se permiten estados de comparación repetidos.")
        for otro in self.comparaciones:
            if isinstance(otro, bool) or not isinstance(otro, int) or not 1 <= otro <= LIMITE_N:
                raise ErrorFisico("Cada estado de comparación debe ser un entero entre 1 y 30.")
            if otro == self.n:
                raise ErrorFisico("El estado principal ya está representado: no lo agregues otra vez.")


def validar_numero(valor: Any, nombre: str) -> float:
    """No transforma silenciosamente bool, cadenas vacías ni NaN/Inf."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErrorFisico(f"{nombre}: ingresá un número válido.")
    x = float(valor)
    if not isfinite(x):
        raise ErrorFisico(f"{nombre}: ingresá un número finito.")
    return x


def configurar(datos: dict[str, Any]) -> Config:
    if not isinstance(datos, dict):
        raise ErrorFisico("Los parámetros deben enviarse como un objeto JSON.")
    masa = validar_numero(datos.get("masa", 1.0), "Masa")
    pendiente = validar_numero(datos.get("pendiente", 1.0), "Pendiente")
    n = datos.get("n", 1)
    otros = datos.get("comparaciones", [])
    if not isinstance(otros, list):
        raise ErrorFisico("Las comparaciones deben ser una lista de números cuánticos.")
    cfg = Config(masa, pendiente, n, tuple(otros))
    cfg.validar()
    return cfg


@lru_cache(maxsize=1)
def ceros_airy() -> tuple[float, ...]:
    """a_n>0 definidos por Ai(-a_n)=0. SciPy retorna ceros negativos."""
    raices, _, _, _ = ai_zeros(LIMITE_N)
    # Afinar las raíces: algunas versiones de SciPy devuelven la quinta
    # con residuo Ai(-a_5) del orden de 1e-11, visible al marcar nodos.
    afinadas = []
    for cero in -raices:
        a = float(cero)
        for _ in range(3):
            valor, derivada, _, _ = airy(-a)
            a += float(valor / derivada)  # Newton para f(a)=Ai(-a).
        afinadas.append(a)
    return tuple(afinadas)


def cero_airy(n: int) -> float:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= LIMITE_N:
        raise ErrorFisico("n debe ser entero entre 1 y 30.")
    return ceros_airy()[n - 1]


def longitud_caracteristica(masa: float, pendiente: float) -> float:
    """ell en nm. ell^3 = hbar^2/(2 m alpha), con unidades coherentes."""
    if not (isfinite(masa) and isfinite(pendiente) and masa > 0 and pendiente > 0):
        raise ErrorFisico("Se requieren masa y pendiente positivas y finitas.")
    return (C_ELECTRON / (masa * pendiente)) ** (1 / 3)


def energia_n(n: int, masa: float, pendiente: float) -> float:
    return pendiente * longitud_caracteristica(masa, pendiente) * cero_airy(n)


def normalizador(n: int, ell: float) -> float:
    """N_n>0, en nm^-1/2: 1/(sqrt(ell)*|Ai'(-a_n)|)."""
    derivada = float(airy(-cero_airy(n))[1])
    return 1.0 / (sqrt(ell) * abs(derivada))


def funcion_onda(x: np.ndarray | float, n: int, ell: float) -> np.ndarray | float:
    """Función real normalizada; se fija psi=0 en x<=0 por la pared.

    El cero de Airy en x=0 se impone exactamente para evitar redondeos.
    """
    arr = np.asarray(x, dtype=float)
    z = np.maximum(arr, 0.0) / ell - cero_airy(n)
    psi = normalizador(n, ell) * airy(z)[0]
    psi = np.where(arr <= 0, 0.0, psi)
    return float(psi) if psi.ndim == 0 else psi


def densidad(x: np.ndarray | float, n: int, ell: float) -> np.ndarray | float:
    return np.square(funcion_onda(x, n, ell))


def nodos_interiores(n: int, ell: float) -> list[float]:
    """Las n-1 raíces interiores son ell*(a_n-a_k), k=1,...,n-1."""
    a = cero_airy(n)
    return sorted(ell * (a - cero_airy(k)) for k in range(1, n))


def punto_retorno(n: int, ell: float) -> float:
    """x_t=E/alpha=ell*a_n."""
    return ell * cero_airy(n)


@lru_cache(maxsize=1)
def desplazamiento_airy_1e() -> float:
    """z>0 que satisface Ai(z)=Ai(0)/e, función decreciente para z>=0."""
    ai0 = float(airy(0.0)[0])
    return float(brentq(lambda z: float(airy(z)[0]) - ai0 / e, 0.0, 12.0,
                        xtol=2e-14, rtol=1e-14))


def penetracion_efectiva(n: int, ell: float) -> float:
    # n se valida expresamente, aunque el resultado escalado no depende de n.
    cero_airy(n)
    return ell * desplazamiento_airy_1e()


def integral_normalizacion(n: int) -> tuple[float, float]:
    """Integración numérica independiente en y=x/ell-a_n, desde -a_n a infinito.

    Integral = integral Ai(y)^2 dy / Ai'(-a_n)^2.
    No se normaliza la curva numérica con la integral calculada.
    """
    a = cero_airy(n)
    d = float(airy(-a)[1])
    integrando = lambda y: float(airy(y)[0]) ** 2 / d**2
    valor, error_quad = quad(integrando, -a, np.inf, epsabs=2e-12,
                            epsrel=2e-12, limit=350, points=None)
    return float(valor), float(error_quad)


def probabilidad_prohibida(n: int) -> float:
    """Integral desde x_t hasta infinito, sobre el dominio completo no visible."""
    a = cero_airy(n)
    d = float(airy(-a)[1])
    val, _ = quad(lambda y: float(airy(y)[0]) ** 2 / d**2,
                  0.0, np.inf, epsabs=2e-13, epsrel=2e-12, limit=200)
    return float(val)


def incertidumbres(n: int, masa: float, pendiente: float, ell: float) -> dict[str, float | bool]:
    """Momentos exactos del quantum bouncer (virial y relación de Airy).

    <x>=2*a_n*ell/3; <x²>=8*a_n²*ell²/15;
    <p>=0; <p²>=2*m*E/3. Se convierten SI para el panel.
    """
    a = cero_airy(n)
    en_ev = energia_n(n, masa, pendiente)
    x_medio_nm = 2.0 * a * ell / 3.0
    x2_medio_nm2 = 8.0 * (a * ell) ** 2 / 15.0
    dx_nm = sqrt(max(0.0, x2_medio_nm2 - x_medio_nm**2))
    dp_si = sqrt(2.0 * masa * M_E * en_ev * EV_J / 3.0)
    producto = dx_nm * NM_M * dp_si
    mitad_hbar = HBAR_SI / 2.0
    ratio = producto / mitad_hbar
    return {
        "delta_x_nm": dx_nm,
        "delta_p_si": dp_si,
        "producto_si": producto,
        "limite_si": mitad_hbar,
        "R": ratio,
        "cumple": ratio >= (1.0 - 1e-10),
    }


def dependencias(cfg: Config, puntos: int = 125) -> dict[str, Any]:
    """Curvas reales recalculadas para masa/pendiente, sin cálculo físico JS."""
    m = np.geomspace(0.1, 10.0, puntos)
    a = np.geomspace(0.01, 10.0, puntos)
    raiz = cero_airy(cfg.n)
    e_m = raiz * (C_ELECTRON / m) ** (1 / 3) * cfg.pendiente ** (2 / 3)
    e_a = raiz * (C_ELECTRON / cfg.masa) ** (1 / 3) * a ** (2 / 3)
    return {
        "masas": [float(v) for v in m],
        "energias_masa": [float(v) for v in e_m],
        "pendientes": [float(v) for v in a],
        "energias_pendiente": [float(v) for v in e_a],
    }


def simular(config: Config) -> dict[str, Any]:
    """Devuelve JSON serializable con *todos* los resultados calculados en Python."""
    config.validar()
    ell = longitud_caracteristica(config.masa, config.pendiente)
    seleccion = [config.n, *config.comparaciones]
    max_n = max(10, *seleccion)
    # Primero diez líneas horizontales; si n>10 se muestran hasta n.
    niveles = [{"n": j, "energia_ev": energia_n(j, config.masa, config.pendiente),
                "x_retorno_nm": punto_retorno(j, ell)} for j in range(1, max_n + 1)]
    e_n = energia_n(config.n, config.masa, config.pendiente)
    xt = punto_retorno(config.n, ell)
    delta = penetracion_efectiva(config.n, ell)
    prohibida = probabilidad_prohibida(config.n)
    integral, error_integral = integral_normalizacion(config.n)
    error_norm = abs(1.0 - integral)
    inc = incertidumbres(config.n, config.masa, config.pendiente, ell)

    # Dos dominios de dibujo diferentes evitan comprimir las curvas físicas
    # cuando el diagrama de energías contiene numerosos niveles.
    x_max_estados = max(punto_retorno(j, ell) for j in seleccion)
    xmax_onda = max(x_max_estados + 7 * ell, xt + delta + 5 * ell)
    xmax_pot = max(niveles[-1]["x_retorno_nm"] + 3.5 * ell, xmax_onda)
    xs = np.linspace(0.0, xmax_onda, 1751)
    # La recta se codifica con dos puntos físicos calculados en Python.
    # El punto remoto permite ampliar manualmente el eje x sin recalcular
    # ni extrapolar el potencial dentro de JavaScript.
    xp = np.array([0.0, max(1000.0, 20 * xmax_pot)])
    estados = []
    for indice, j in enumerate(seleccion):
        ps = np.asarray(funcion_onda(xs, j, ell), dtype=float)
        estados.append({
            "n": j, "color": COLORES[indice], "principal": indice == 0,
            "energia_ev": energia_n(j, config.masa, config.pendiente),
            "x_retorno_nm": punto_retorno(j, ell),
            "psi": ps.tolist(), "rho": np.square(ps).tolist(),
            "nodos_nm": nodos_interiores(j, ell),
        })
    return {
        "config": {**asdict(config), "comparaciones": list(config.comparaciones)},
        "constantes": {"hbar_si": HBAR_SI, "masa_electron_kg": M_E, "ev_j": EV_J},
        "escalas": {"ell_nm": ell, "x_onda_max_nm": xmax_onda,
                    "x_potencial_max_nm": xmax_pot,
                    "potencial_auto_y_max_ev": config.pendiente * xmax_pot * 1.09},
        "potencial": {"x_nm": xp.tolist(),
                      "u_ev": (config.pendiente * xp).tolist()},
        "niveles": niveles,
        "onda": {"x_nm": xs.tolist(), "estados": estados},
        "propiedades": {
            "energia_ev": e_n,
            "x_retorno_nm": xt,
            "penetracion_nm": delta,
            "x_penetracion_nm": xt + delta,
            "probabilidad_prohibida": prohibida,
            "nodos_interiores": config.n - 1,
        },
        "incertidumbre": inc,
        "normalizacion": {
            "integral": integral, "error_absoluto": error_norm,
            "error_cuadratura_estimado": error_integral,
            "tolerancia": 1e-8,
            "correcta": error_norm <= 1e-8,
        },
        "dependencias": dependencias(config),
    }


if __name__ == "__main__":
    import json
    import sys
    if "--test" in sys.argv:
        import runpy
        from pathlib import Path
        archivo = Path(__file__).resolve().with_name("Simulador 2 - Pruebas.py")
        sys.argv = [str(archivo)]
        runpy.run_path(str(archivo), run_name="__main__")
        raise SystemExit(0)
    print(json.dumps(simular(Config()), ensure_ascii=False, indent=2)[:2600])
