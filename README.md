# Pore-Scale Reactive Transport Modeling of Nano Zero-Valent Iron (nZVI) Filters for Chromium Remediation & Resource Recovery

> **Project Status:** On Hold (Core Geometry Reconstruction, REV Analysis, Hydrodynamic LBM Verification, and Geochemical Speciation Modules Completed; Full Multi-Ion Reactive Coupling Pending)

## Overview
This repository contains a Python-based, JAX-accelerated pore-scale simulation framework designed to model fluid flow, solute transport, and geochemical reactions within **Nano Zero-Valent Iron (nZVI)** porous filtration barriers. By combining **Digital Rock Physics (DRP)**, the **Lattice Boltzmann Method (D3Q19 LBM)**, and **PHREEQC** thermodynamic speciation (`minteq.v4.dat`), this model simulates the reduction of toxic Hexavalent Chromium ($\text{Cr(VI)}$) into stable Trivalent Chromium ($\text{Cr(III)}$) under realistic, multi-ion groundwater and industrial wastewater conditions.

---

## Why This Matters: Industrial Context & Impact (Birgunj, Nepal)

Industrial corridors such as the **Birgunj–Pathlaiya industrial belt in Nepal** (home to numerous tanneries, electroplating units, textile dyeing factories, and metal-processing industries) face severe environmental and economic challenges regarding heavy metal wastewater discharge - specifically **Hexavalent Chromium ($\text{Cr(VI)}$)**.

### The Economic & Technological Bottleneck
* **High Cost of Conventional Systems:** While advanced remediation technologies exist (such as ion-exchange resins, reverse osmosis, and automated chemical dosing plants), their high capital expenditure (CAPEX), continuous chemical costs, and energy demands make them financially unviable for small- and medium-scale enterprises (SMEs) struggling with tight profit margins.
* **Premature Filter Failure:** Passive, low-cost permeable reactive barriers (PRBs) using Zero-Valent Iron offer an affordable alternative. However, real industrial wastewater contains competing background ions ($\text{SO}_4^{2-}$, $\text{HCO}_3^-$, $\text{Ca}^{2+}$, $\text{Mg}^{2+}$, $\text{PO}_4^{3-}$) that cause mineral precipitation, surface passivation, and pore clogging—leading to premature filter failure and wasted capital.

### How This Research Helps
1. **Virtual Prototyping Before Capital Investment:** Instead of expensive trial-and-error pilot plants, this digital twin predicts exact filter lifespan, clogging rates, and $\text{Cr(VI)}$ breakthrough times using specific local water chemistries.
2. **Optimizing Low-Cost Material Mixes:** By identifying the exact Representative Elementary Volume (REV) and pore structure needed to prevent early passivation, industries can design cheaper, longer-lasting nZVI-sand composite filters.
3. **Pathway to Resource Recovery:** Controlled reduction precipitates Cr(III) hydroxides and mixed Fe(III)-Cr(III) phases within the pore matrix. Mapping where and how these precipitates accumulate lays the groundwork for selective chromium sludge recovery and reuse in tanning or pigment manufacturing—turning a toxic liability into a recoverable resource.

---

## Current Repository Architecture & Progress

The modeling pipeline is modularized into sequential verification and validation steps:

### Completed & Verified Modules
1. **Microstructure & REV Characterization (`Step_1` to `Step_3.4`):**
   * Extracted spatial statistics and **Two-Point Correlation Functions ($S_2(r)$)** from TEM micrographs using Multi-Otsu thresholding (`skimage` & `PoreSpy`).
   * Reconstructed a statistically equivalent **$512^3$ voxel 3D nZVI geometry** (`0.6349 nm/voxel` resolution, `44.52%` porosity) via JAX-accelerated 3D Fast Fourier Transforms, safely exceeding the physical correlation length (`~150 nm`) to guarantee a transport-valid REV.
   * Generated 3D surface meshes and cutaway visualizations via `PyVista`.
2. **Hydrodynamic Flow Validation (`Step_4.1` & `Step_4.2`):**
   * Implemented a D3Q19 BGK Lattice Boltzmann solver in **JAX** with Guo body-forcing and on-site no-slip bounce-back boundary conditions.
   * Validated steady-state convergence on a $64^3$ REV subdomain (converged at iteration 400 with `< 0.05%` relative velocity derivative change and `< 1e-5` mass conservation error).
3. **Reactive Surface Mapping & Geochemical Setup (`Step_5` & `Step_5.1`):**
   * Mapped active fluid-solid interface voxels (`40.82%` of pore space) and calculated physical reactive surface areas (`m²/node`).
   * Verified multi-ion aqueous speciation (`Cl-`, `SO4--`, `HCO3-`, `Ca++`, `Mg++`, `PO4---`, `Cr(VI)`, `Cr(III)`) using `phreeqpython` linked to the `minteq.v4.dat` thermodynamic database.

### Pending Next Steps (When Resumed)
* [ ] **Step 6 (`Step_6_Tracer_Breakthrough.py`):** Execute the conservative tracer ($\text{Cl}^-$) advection-diffusion breakthrough curve to validate numerical dispersion limits.
* [ ] **Step 7 (Single-Species Kinetics):** Couple surface-area-scaled $\text{Cr(VI)} \rightarrow \text{Cr(III)}$ reduction kinetics to the LBM velocity field and benchmark against literature rate constants ($k_{obs}$).
* [ ] **Step 8 (Full Multi-Ion Coupling & Passivation):** Introduce competing anions/cations ($\text{PO}_4^{3-}$, $\text{HCO}_3^-$, $\text{Ca}^{2+}$) via operator-splitting with PHREEQC to simulate mineral precipitation, dynamic porosity loss, and chromium recovery yield.

---

## Environment & Dependencies
* **Python:** `3.12.10` (Recommended for pre-compiled Windows scientific wheels)
* **Core Libraries:** `jax`, `jaxlib`, `numpy`, `scipy`, `porespy`, `scikit-image`, `opencv-python`, `pyvista`, `matplotlib`, `phreeqpython`
* **Thermodynamic Database:** `minteq.v4.dat` (included in root directory)
