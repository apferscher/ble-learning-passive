# Active or Passive? A Comparison of Automata Learning Paradigms in Practice

This repository contains the supplemental material to the paper ' Active or Passive? A Comparison of Automata Learning Paradigms in Practice' of Bernhard K. Aichernig<sup>1</sup>, Edi Muškardin<sup>1,2</sup>, and Andrea Pferscher<sup>1</sup> (<sup>1</sup>Institute of Software Technology, Graz University of Technology; <sup>2</sup> Silicon Austria Labs, TU Graz - SAL DES Lab).

## Content
- BLE models ([automata/](automata/)):
    - [CYBLE-416045-02](automata/CYBLE-416045-02.dot) \[[PDF](automata/pdf/cyble-416045-02.pdf)\]
    - [nRF52832](automata/nRF52832.dot) \[[PDF](automata/pdf/nRF52832.pdf)\]
    - [CC2650](automata/CC2650.dot) \[[PDF](automata/pdf/cc2650.pdf)\]
    - [CYW43455](automata/CYW43455.dot) \[[PDF](automata/pdf/cyw43455.pdf)\]
    - [CC2640R2 (no pairing request)](automata/CC2640R2-no-pairing-req.dot) \[[PDF](automata/pdf/CC2640R2-no-pairing-req.pdf)\]
    - [CC2640R2 (no feature request)](automata/CC2640R2-no-feature-req.dot) \[[PDF](automata/pdf/CC2640R2-no-feature-req.pdf)\]
    - [CC2652R1](automata/CC2652r1.dot) \[[PDF](automata/pdf/CC2652r1.pdf)\]
- Experiment execution ([main.py](main.py)):

## Prerequisites

Python library  [Aalpy >=1.2.8](https://github.com/DES-Lab/AALpy)

**Requirements installation:** 

```bash
sudo pip3 install -r requirements.txt
```
## Experiment Execution

    python3 main.py

## Acknowledgement
- [AALpy](https://github.com/DES-Lab/AALpy): active automata learning library