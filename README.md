# Active vs. Passive? A Comparison of Automata Learning Paradigms for Network Protocols

This repository contains the supplemental material to the paper ' Active vs. Passive? A Comparison of Automata Learning Paradigms for Network Protocols' of Bernhard K. Aichernig<sup>1</sup>, Edi Muškardin<sup>1,2</sup>, and Andrea Pferscher<sup>1</sup> (<sup>1</sup>Institute of Software Technology, Graz University of Technology; <sup>2</sup> Silicon Austria Labs, TU Graz - SAL DES Lab).

## Content
- BLE models ([automata/BLE/](automata/BLE/)):
    - [CYBLE-416045-02](automata/BLE/CYBLE-416045-02.dot) \[[PDF](automata/BLE/pdf/cyble-416045-02.pdf)\]
    - [nRF52832](automata/BLE/nRF52832.dot) \[[PDF](automata/BLE/pdf/nRF52832.pdf)\]
    - [CC2650](automata/BLE/CC2650.dot) \[[PDF](automata/BLE/pdf/cc2650.pdf)\]
    - [CYW43455](automata/BLE/CYW43455.dot) \[[PDF](automata/BLE/pdf/cyw43455.pdf)\]
    - [CC2640R2 (no pairing request)](automata/BLE/CC2640R2-no-pairing-req.dot) \[[PDF](automata/BLE/pdf/CC2640R2-no-pairing-req.pdf)\]
    - [CC2640R2 (no feature request)](automata/BLE/CC2640R2-no-feature-req.dot) \[[PDF](automat/BLEa/pdf/CC2640R2-no-feature-req.pdf)\]
    - [CC2652R1](automata/BLE/CC2652r1.dot) \[[PDF](automata/BLE/pdf/CC2652r1.pdf)\]
- MQTT models ([automata/MQTT/](automata/MQTT/)):
    - [ActiveMQ](automata/MQTT/ActiveMQ__two_client_will_retain.dot) \[[PDF](automata/MQTT/pdf/ActiveMQ__two_client_will_retain.pdf)\]
    - [emqtt](automata/MQTT/emqtt__two_client_will_retain.dot) \[[PDF](automata/MQTT/pdf/emqtt__two_client_will_retain.pdf)\]
    - [HBMQTT](automata/MQTT/hbmqtt__two_client_will_retain.dot) \[[PDF](automata/MQTT/pdf/hbmqtt__two_client_will_retain.pdf)\]
    - [Mosquitto](automata/MQTT/mosquitto__two_client_will_retain.dot) \[[PDF](automata/MQTT/pdf/mosquitto__two_client_will_retain.pdf)\]
    - [Mosquitto](automata/MQTT/VerneMQ__two_client_will_retain.dot) \[[PDF](automata/MQTT/pdf/VerneMQ__two_client_will_retain.pdf)\]
- Experiment execution ([main.py](main.py)):

## Prerequisites

Python library  [Aalpy >=1.2.8](https://github.com/DES-Lab/AALpy)

**Requirements installation:** 

```bash
sudo pip3 install -r requirements.txt
```
## Experiment Execution

    python3 main.py <BENCHMARK>

Possible benchmarks are "MQTT" or "BLE". The script generates three different csv-files containing the learning results for active learning, passive learning, and active learning with cache.

## Acknowledgement
- [AALpy](https://github.com/DES-Lab/AALpy): active automata learning library