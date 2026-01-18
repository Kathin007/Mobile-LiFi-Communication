# Mobile LiFi Communication via Autonomous Drone Technology
## 📖 Overview
This project presents a **prototype simulation and system design** for enabling **secure, mobile LiFi-based communication** using an **autonomous drone platform**. 

The primary objective is to demonstrate the **feasibility of maintaining line-of-sight (LoS) optical communication** in dynamic and obstacle-rich environments through **AI-assisted navigation and decision-making**. This work serves as a proof-of-concept to validate system architecture, autonomy logic, and communication reliability under constrained resources.

---

## ⚠️ Problem Statement
Conventional **RF-based communication systems** used in defense and disaster-response scenarios face significant vulnerabilities:
* **Jamming and Interception:** High susceptibility to electronic warfare and signal intelligence.
* **Infrastructure Damage:** Dependence on static towers that may be destroyed or unavailable.
* **Terrain Constraints:** Poor performance in mountainous, forested, or disaster-hit terrains due to multipath fading.

While **LiFi (Light Fidelity)** offers a secure and interference-resistant alternative, its **strict line-of-sight dependency** limits real-world deployment. This project explores how **autonomous drones** can act as **mobile LiFi relays**, dynamically repositioning themselves to maintain connectivity.

---

## 💡 Proposed Solution
* **Mobile Relay:** A drone-mounted LiFi transmitter providing mobile optical communication.
* **Autonomous Navigation:** Intelligent positioning to preserve LoS with a base station.
* **Obstacle Avoidance:** Computer vision-based detection to navigate complex environments.
* **AI-Assisted Decision Making:** Real-time logic to ensure safe movement without breaking the communication link.
* **Hybrid Reliability:** An RF fallback channel for control-plane redundancy.

---

## 🏗️ System Architecture



The following flow represents the integrated logic from perception to motion:

1.  **Environment:** AirSim / Simulation Environment
2.  **Input:** Camera Feed & Drone State
3.  **Perception:** Obstacle Detection (**YOLOv8**)
4.  **Processing:** Feature Extraction (Bounding Boxes, Position, Angle)
5.  **Intelligence:** Logistic Regression (LiFi-Safe Direction Prediction)
6.  **Pathing:** Backtracking-Based Path Planner
7.  **Actuation:** Drone Motion Controller

---

## 🛠️ Key Components

### 1. Obstacle Detection
* Utilizes **YOLOv8** (or simulated YOLO outputs).
* Processes bounding box data to identify blocked directions.
* Prioritizes **reliability and precision** over raw frame-rate in this prototype stage.

### 2. Decision Making
* **Logistic Regression** predicts whether a specific movement direction preserves LiFi connectivity.
* **Why Logistic Regression?**
    * Extremely low computational overhead.
    * High explainability (critical for defense applications).
    * Suited for real-time inference on edge hardware.

### 3. Path Planning

* **Backtracking-based directional exploration** to recover from dead-ends.
* **Priority Heuristic:** Forward → Left → Right → Up.
* **Vertical Recovery:** Upward movement acts as a failsafe when horizontal paths are occluded.

### 4. LiFi Connectivity Model
* Base station treated as a fixed coordinate in 3D space.
* Signal strength estimated via distance-decay and angular alignment logic.
* Focused on **conceptual validation** of the link budget rather than physical-layer hardware optimization.

---

## 📐 Design Philosophy
* **Reliability over Speed:** Consistent connectivity is valued over rapid transit.
* **Explainability:** Avoids "black-box" AI; decisions can be traced back to sensor inputs.
* **Prototype Feasibility:** Designed to be implementable on current-generation edge compute (e.g., Raspberry Pi/Jetson).
* **Safety-First:** Autonomous behaviors prioritize collision avoidance.

---

## 🖥️ Simulation Environment
The system is validated using:
* **Python-based** control logic.
* **AirSim** for realistic drone physics and high-fidelity camera feeds.
* **Visualization Dashboards** to monitor connectivity status and obstacle proximity in real-time.

---

## 🔌 Hardware (Prototype-Level)

| Component | Specification (Conceptual) |
| :--- | :--- |
| **Drone Platform** | Simulated / Ryze Tello class |
| **Transmitter** | ESP32-based LiFi module |
| **Receiver** | Raspberry Pi-based optical sensor |
| **Optics** | Lightweight lenses and filtering components |

> **Note:** The current implementation focuses on **system logic validation** and software architecture, not custom hardware fabrication.

---

## 📊 Evaluation Metrics
* **Connectivity Stability:** Percentage of mission time with active LoS.
* **Obstacle Recovery Time:** Duration taken to find a clear path after encountering an obstruction.
* **Decision Consistency:** Statistical variance in directional choices under similar conditions.
* **System Latency:** Responsiveness of the control loop under dynamic conditions.

---

## 🛑 Limitations & Future Scope

### Current Limitations
* Optical communication is susceptible to high ambient light (sunlight).
* LoS remains a hard constraint; the system cannot communicate through walls.
* Current prototype focuses on single-drone scenarios.

### Future Work
* **Swarm Networks:** Multi-drone LiFi relay chains for extended range.
* **Sensor Fusion:** Integrating Depth/Infrared sensors for better night-time performance.
* **Field Trials:** Transitioning from AirSim to Hardware-in-the-Loop (HITL) testing.

---

## 🛡️ Applications
* **Defense:** Secure, stealthy communication in contested electronic environments.
* **Disaster Response:** Rapid deployment of comms in areas with downed towers.
* **Border Security:** Mobile, low-intercept surveillance links.
* **RF-Denied Zones:** Communication in hospitals or industrial plants sensitive to RF interference.

---

## 📜 Disclaimer
This project is a **research and hackathon prototype** intended to demonstrate feasibility and system design concepts. It is **not a deployment-ready solution** and should be treated as a proof-of-concept.

---

## 📚 References
1.  H. Haas et al., “What is LiFi?”, *Journal of Lightwave Technology*.
2.  Ryze Tech (DJI), Tello Drone Documentation.
3.  Espressif Systems, ESP32 Technical Reference.
4.  IEEE Surveys on Hybrid LiFi–WiFi Networks.

---

## 👥 Team
**Team Name:** NeuroMinds  
**Domain:** Open Innovation / Defense Technology