# Technical Brief: AI-Powered Traffic Signal Optimization for San Diego Corridors

**Prepared for:** Caltrans Division of Traffic Operations | City of San Diego Transportation Department
**Author:** Samarth Vaka | San Diego, CA
**Date:** March 2026

---

## Executive Summary

TrafficAI is an artificial intelligence platform that optimizes traffic signal timing using reinforcement learning and predictive analytics. Tested on simulated San Diego corridors (El Camino Real, University Ave, I-5 surface intersections), the system demonstrates **25-35% reductions in intersection delay**, **20-30% improvements in corridor speed**, and **measurable reductions in vehicle emissions** compared to fixed-timing baselines — all achievable through software-only deployment to existing NTCIP-compliant signal controllers.

---

## Problem Statement

- San Diego County operates **3,000+ signalized intersections** across 18 cities
- U.S. urban congestion costs drivers **$87 billion annually** in lost productivity (INRIX 2023)
- Fixed-timing signals account for ~70% of urban signals and cannot adapt to real-time demand
- Emergency vehicle response times are impacted by signal delays at intersections

## Proposed Solution

A software-based AI controller that:

1. **Learns optimal signal timing** through reinforcement learning (Dueling Double DQN)
2. **Predicts congestion** 5-15 minutes ahead using historical pattern analysis
3. **Preempts signals for emergency vehicles** with cascading green waves
4. **Quantifies emissions reductions** using EPA-standard emission factors
5. **Requires no new hardware** — deploys as timing plan updates to existing controllers

---

## Technical Approach

### AI Controller: Dueling Double Deep Q-Network

| Component | Specification |
|-----------|--------------|
| Architecture | Dueling DQN with separate value/advantage streams |
| Training Method | Double DQN (policy net selects, target net evaluates) |
| State Space | Queue lengths (NS/EW), phase, elapsed time, wait time |
| Action Space | Phase selection (NS green / EW green) |
| Reward Function | Multi-objective: wait time, throughput, emissions, emergency priority |
| Experience Replay | 30,000-step buffer with uniform sampling |
| Training Stability | Gradient clipping (max_norm=10), target network soft updates |

### Multi-Objective Reward Function

$$R(s, a) = -0.15 \cdot Q_{total} + 0.10 \cdot D_{departed} - 0.08 \cdot |Q_{NS} - Q_{EW}| - 2.5 \cdot \mathbb{1}_{switch} - 0.05 \cdot Q_{total} + 5.0 \cdot \mathbb{1}_{EV\_clear}$$

### Simulation Environment

- **Corridor model:** 5-intersection linear arterial with realistic spacing (350-510m)
- **Demand model:** Poisson arrivals with Gaussian AM/PM peak profiles calibrated to Caltrans PeMS District 11
- **Vehicle propagation:** 60% platoon transfer between intersections (Robertson's dispersion model)
- **Saturation flow:** 0.45-0.50 veh/sec/lane (Highway Capacity Manual 6th Edition)

### Emissions Methodology

Based on EPA AP-42 Section 13.2.1 and MOVES3 model:
- Idle fuel consumption: 0.16 gallons/hour per vehicle
- CO₂ emission factor: 8.887 kg per gallon of gasoline
- Annualized to 16 operational hours/day, 365 days/year

---

## Key Results

| Metric | Fixed Timing | Actuated | AI-Optimized | AI Improvement |
|--------|-------------|----------|--------------|----------------|
| Avg Wait Time | Baseline | ~12% reduction | ~31% reduction | ✓ |
| Avg Queue Length | Baseline | ~10% reduction | ~28% reduction | ✓ |
| Corridor Speed | Baseline | ~8% increase | ~23% increase | ✓ |
| CO₂ Emissions | Baseline | ~9% reduction | ~25% reduction | ✓ |
| EV Response Delay | Baseline | No change | ~67% reduction | ✓ |

*Results from 5-intersection corridor simulation with identical random seeds and emergency vehicle injection.*

---

## Validation & Statistical Rigor

- **5-fold cross-validation** across 10 controller types
- **Mann-Whitney U tests** (α=0.05) for pairwise significance
- **Bootstrap confidence intervals** (95% CI, 300 resamples)
- **Ablation study** for hyperparameter sensitivity
- **66 unit tests** for simulation engine and controllers

---

## Implementation Pathway

### Phase 1: Pilot Corridor (3-6 months)
- Select one San Diego corridor (recommend: El Camino Real, 5 intersections)
- Deploy AI timing plans to existing controllers via NTCIP interface
- Monitor performance with before/after data collection via PeMS

### Phase 2: Validation (3-6 months)
- Compare AI-optimized vs existing timing plans using real traffic data
- Quantify delay reduction, speed improvements, emissions impacts
- Refine model using field data feedback loop

### Phase 3: Expansion
- Scale to additional corridors based on validated results
- Integrate with Caltrans ATMS for centralized monitoring
- Connect to real-time data feeds (PeMS, Bluetooth/WiFi travel time)

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| AI/ML | PyTorch, NumPy, scikit-learn |
| Simulation | Custom Python engine (Poisson, network propagation) |
| Dashboard | Streamlit (interactive web application) |
| Data Sources | Caltrans PeMS, SANDAG Open Data, EPA MOVES3 |
| Deployment | Compatible with NTCIP 1202 signal controllers |

---

## Contact

**Samarth Vaka**
Email: vsamarth2010@gmail.com
GitHub: github.com/svaka2000
San Diego, California

*Previously presented to SANDAG Regional Planning Committee. GSDSEF 2nd Place + Special Recognition.*
