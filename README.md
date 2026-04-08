# Autonomous Combat Identification Engine
**Tactical Reasoning Engine** implementing the **ID3** algorithm from scratch to automate **Combat Identification (CID)**.

* **Architecture:** Non-parametric hierarchical classifier designed for **sensor fusion** and explainable decision-making.
* **Data Flow:**
  * **Source:** Synthetic tactical dataset (10k detections) simulating a high-saturation air combat environment (e.g., Iran). *Generated via `python dataset/script_generator.py`.*
  * **Dimensions:** Symbolic feature vectors (IFF Response, Radar Signature/RCS, Flight Profile, and BFT Proximity).
  * **Target:** Binary classification (1 Friend / 0 Foe) for weapon release authorization.
* **Objective:** Minimize **Shannon Entropy** through **Information Gain** to achieve robust identification—even during primary transponder failure—preventing blue-on-blue incidents (e.g., Kuwait).


## Dependencies installation
Write these commands one by one in the terminal of your editor to install the dependencies:

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
