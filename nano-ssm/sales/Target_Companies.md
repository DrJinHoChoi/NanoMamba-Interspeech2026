# NC-SSM Target Company List

> **Last updated:** 2026-03-28
>
> **Technology:** NC-SSM (Noise-Conditioned State Space Model) -- Ultra-compact edge AI keyword spotting
>
> **Key Value Propositions:**
> - 7,443 parameters / 7.3 KB INT8 -- fits on any $5 MCU
> - 7.1 ms inference on Cortex-M7 (350 ms end-to-end streaming response)
> - Noise-robust by design -- no denoising pipeline required (factory, babble, wind)
> - 720 bytes runtime state -- streaming SSM architecture
> - Patent-pending (US + KR)

---

## 1. Semiconductor / MCU Makers

These companies integrate third-party AI/ML IP into their chip platforms. NC-SSM's tiny footprint and Cortex-M compatibility make it a natural fit for on-chip KWS offerings.

| Company | Why NC-SSM Fits | Target Contact | Engagement Approach | Priority |
|---|---|---|---|---|
| **STMicroelectronics** | STM32 MCU line dominates industrial and consumer edge. NC-SSM at 7.3 KB runs natively on STM32U5/H7 with room to spare; strengthens their STM32Cube.AI ecosystem with a noise-robust KWS reference. | VP of AI/ML Solutions; Director of Edge AI Ecosystem | Conference intro at Embedded World / ST Partner Program application | **High** |
| **NXP Semiconductors** | i.MX RT crossover MCUs target voice-enabled IoT. NC-SSM's streaming architecture and 720-byte state complement NXP's eIQ ML platform with a production-ready KWS model. | Director of Edge Processing; VP of AI/ML Strategy | Partnership proposal via NXP eIQ partner program | **High** |
| **Infineon Technologies** | PSoC and AIROC families serve industrial and automotive voice use cases. NC-SSM's noise robustness is ideal for their noisy-environment customers (motor control, HVAC). | Head of AI/ML, Sense & Control Division; CTO | Cold email + Embedded World demo | **High** |
| **Renesas Electronics** | RA and RX MCU families are widely used in Japanese automotive and industrial. NC-SSM fits their Reality AI strategy and addresses demand for low-power voice triggers. | VP of AI/IoT Business Division | Conference intro at CEATEC; partnership proposal | **Medium** |
| **Ambiq Micro** | Apollo4 SoCs are purpose-built for ultra-low-power always-on sensing. NC-SSM's 7.1 ms inference and tiny state are a perfect match for their sub-mW power budgets in wearables and hearables. | CTO; VP of AI/ML | Direct cold email (small company, CTO accessible) | **High** |
| **Nordic Semiconductor** | nRF series is the go-to for BLE wearables and IoT. Adding on-device KWS via NC-SSM enables voice-triggered BLE commands without a cloud round-trip. | Director of Product Management; Head of Edge AI | Partnership proposal + demo at Nordic DevZone | **Medium** |
| **Espressif Systems** | ESP32-S3 has ML acceleration but lacks a noise-robust KWS reference. NC-SSM at 7.3 KB runs easily alongside Wi-Fi/BLE stacks on ESP32, enabling voice-first IoT devices at the $3 BOM level. | CTO; VP of Software Platform | Open-source demo on ESP-IDF + cold email | **Medium** |

---

## 2. Consumer Electronics / Smart Home

These companies ship millions of devices that need reliable wake-word detection. NC-SSM eliminates the need for expensive DSPs or cloud fallback.

| Company | Why NC-SSM Fits | Target Contact | Engagement Approach | Priority |
|---|---|---|---|---|
| **Samsung Electronics** | Galaxy, SmartThings, Bixby ecosystem spans phones, TVs, and home appliances. NC-SSM enables always-on Bixby wake-word on low-cost appliance MCUs without a dedicated voice chip. | VP of Bixby / On-Device AI; Head of SmartThings Platform | Internal referral (Korean patent filing builds credibility); partnership proposal | **High** |
| **LG Electronics** | ThinQ smart home appliances and webOS TVs need cost-effective voice triggers. NC-SSM's noise robustness handles kitchen/laundry noise where current solutions struggle. | VP of AI Research (LG AI Research); Director of ThinQ Platform | Conference intro at CES; partnership proposal | **High** |
| **Xiaomi** | Massive IoT ecosystem (Mi Home) with aggressive BOM targets. NC-SSM at 7.3 KB on a $5 MCU enables voice wake on sub-$20 smart home devices at Xiaomi's scale. | Head of AI Platform; VP of IoT Division | Cold email + technical white paper | **Medium** |
| **Bose** | Premium headphones and speakers need low-latency, noise-robust voice activation. NC-SSM's streaming architecture delivers 350 ms response -- faster than cloud-based alternatives. | VP of Research (Hear Division); Director of Voice Engineering | Conference intro at AES Convention; partnership proposal | **Medium** |
| **Sonos** | Multi-room speakers need always-on wake-word with minimal power draw. NC-SSM can run on existing Sonos MCU hardware, reducing reliance on far-field DSP arrays for initial wake detection. | VP of Product; Head of Voice Platform | Cold email + technical demo | **Medium** |
| **Harman International** (Samsung subsidiary) | JBL/Harman Kardon speakers and automotive audio systems. NC-SSM bridges their consumer and automotive audio lines with a single KWS solution that handles both home and in-cabin noise. | VP of Software & Services; Director of Connected Services | Internal referral via Samsung relationship | **High** |

---

## 3. Automotive

In-cabin voice commands operate in high-noise environments (road, HVAC, wind). NC-SSM's noise-conditioned design was built for exactly this.

| Company | Why NC-SSM Fits | Target Contact | Engagement Approach | Priority |
|---|---|---|---|---|
| **Hyundai Mobis** | Tier-1 supplier for Hyundai/Kia in-cabin systems. NC-SSM's noise robustness handles road and HVAC noise for always-on wake-word, running on existing in-cabin MCUs without added BOM cost. | Director of In-Cabin AI; VP of Infotainment Platform | Direct outreach (Korean company, KR patent builds trust); partnership proposal | **High** |
| **Continental AG** | Major Tier-1 for cockpit electronics. NC-SSM enables voice-triggered ADAS alerts and cabin controls on their existing Cortex-M based ECUs without requiring a separate AI accelerator. | Head of Human-Machine Interaction; VP of Interior Electronics | Conference intro at CES Automotive / AutoSens | **High** |
| **Robert Bosch** | Broad automotive and industrial portfolio. NC-SSM's dual applicability (in-cabin voice + factory voice commands) lets Bosch deploy one KWS IP across automotive and Industry 4.0 divisions. | VP of Cross-Domain Computing; Director of AI & Automation | Partnership proposal via Bosch Open Innovation | **High** |
| **Cerence Inc.** | Pure-play automotive voice AI. NC-SSM can serve as their ultra-low-power wake-word front-end, replacing heavier neural models and reducing latency in their voice assistant pipeline. | CTO; VP of Product Engineering | Cold email + technical benchmark comparison | **High** |

---

## 4. IoT / Industrial

Always-on voice in factories, warehouses, and field equipment demands noise robustness and minimal compute. NC-SSM's noise-conditioned architecture eliminates the denoising stage entirely.

| Company | Why NC-SSM Fits | Target Contact | Engagement Approach | Priority |
|---|---|---|---|---|
| **Siemens** | Factory automation (SIMATIC) and building tech. NC-SSM enables hands-free voice commands on PLCs and HMIs in 90+ dB factory noise -- a capability their current edge offerings lack. | VP of Industrial Edge; Head of AI & Digitalization | Conference intro at Hannover Messe; partnership proposal | **High** |
| **ABB** | Robotics and industrial automation. NC-SSM allows voice-triggered e-stop and mode switching on ABB robot controllers running Cortex-M, improving operator safety in noisy environments. | Director of Digital; VP of Robotics AI | Cold email + safety use-case demo | **Medium** |
| **Honeywell** | Connected buildings and warehouse automation (Intelligrated). NC-SSM enables voice-picking and voice-controlled HVAC on existing sensor MCUs without a separate voice module. | VP of Connected Enterprise; Head of Warehouse Automation Tech | Partnership proposal via Honeywell Forge ecosystem | **Medium** |
| **Rockwell Automation** | Allen-Bradley PLCs and FactoryTalk platform. NC-SSM voice triggers on PLC-adjacent MCUs enable hands-free operation in noisy production lines -- differentiator for their smart manufacturing story. | VP of Innovation; Director of Edge & IoT | Conference intro at Automation Fair; cold email | **Medium** |

---

## 5. Voice AI / Speech Technology

These companies are potential partners, integrators, or acquirers. NC-SSM fills a gap in their product lines for ultra-low-resource, noise-robust KWS.

| Company | Why NC-SSM Fits | Target Contact | Engagement Approach | Priority |
|---|---|---|---|---|
| **Sensory Inc.** | TrulyHandsfree KWS is their core product. NC-SSM's 7.3 KB model is 10-100x smaller than their current offerings, opening new markets on ultra-constrained MCUs they cannot currently address. | CEO (Todd Mozer); VP of Engineering | Direct CEO outreach; licensing / acquisition discussion | **High** |
| **Picovoice** | Porcupine wake-word engine targets edge devices. NC-SSM's noise-conditioned approach outperforms in noisy environments without needing their separate noise suppression module (Cobra). | CEO (Alireza Kenarsari); CTO | Cold email with benchmark comparison; partnership or licensing | **High** |
| **Fluent.ai** | Speech-to-intent on edge. NC-SSM can serve as a lightweight wake-word front-end that triggers their heavier intent model, reducing always-on power consumption in their pipeline. | CEO; VP of Product | Conference intro at Interspeech 2026; partnership proposal | **Medium** |
| **Syntiant** | NDP chips are purpose-built for always-on audio AI. NC-SSM's architecture could be optimized for Syntiant's neural decision processors, offering their customers a noise-robust KWS option. | CTO; VP of Partnerships | Technical partnership proposal; co-development pitch | **High** |
| **Knowles Corporation** | MEMS microphone + edge AI (IA8201 processor). NC-SSM bundled with Knowles mics creates a complete voice-trigger module -- mic + ML in a single package, simplifying customer integration. | VP of Intelligent Audio; Director of AI Solutions | Partnership proposal for integrated mic+KWS module | **High** |

---

## 6. Wearables / Hearables

Battery life is king. NC-SSM's ultra-low compute (7.1 ms on Cortex-M7) and tiny state (720 bytes) enable always-on voice wake without draining coin-cell or earbud batteries.

| Company | Why NC-SSM Fits | Target Contact | Engagement Approach | Priority |
|---|---|---|---|---|
| **Apple** (AirPods / Watch) | AirPods and Apple Watch need ultra-efficient on-device "Hey Siri" detection. NC-SSM's 720-byte state and sub-10 ms inference could extend battery life while improving wind-noise robustness during outdoor use. | Audio Engineering Lead; Senior Director of Siri On-Device ML | Conference intro at ICASSP / Interspeech; research paper as credibility signal | **Low** |
| **Jabra** (GN Audio) | Elite and Evolve headsets serve enterprise and consumer. NC-SSM enables custom wake-words on their existing Cortex-M audio SoCs without firmware bloat, differentiating their enterprise UC headsets. | VP of Product; Director of Audio AI R&D | Cold email + demo on Jabra dev kit | **Medium** |
| **Sony Audio** (Headphones Division) | WH-1000XM series and LinkBuds need low-latency voice activation. NC-SSM's noise robustness handles the very noise their ANC systems create, reducing false rejections during playback. | Head of Audio AI; VP of Product Planning | Conference intro at CES / IFA; partnership proposal | **Medium** |
| **Bragi** | Pioneer in hearable computing, now focused on B2B audio AI platform. NC-SSM fits their Bragi OS as a licensable KWS module for OEM partners building next-gen hearables on constrained hardware. | CEO (Nikolaj Hviid); CTO | Direct CEO outreach (small company); licensing discussion | **Medium** |

---

## Priority Summary

### High Priority (14 companies) -- Pursue immediately

| Company | Category |
|---|---|
| STMicroelectronics | Semiconductor |
| NXP Semiconductors | Semiconductor |
| Infineon Technologies | Semiconductor |
| Ambiq Micro | Semiconductor |
| Samsung Electronics | Consumer Electronics |
| LG Electronics | Consumer Electronics |
| Harman International | Consumer Electronics |
| Hyundai Mobis | Automotive |
| Continental AG | Automotive |
| Robert Bosch | Automotive |
| Cerence Inc. | Automotive |
| Siemens | Industrial IoT |
| Sensory Inc. | Voice AI |
| Picovoice | Voice AI |
| Syntiant | Voice AI |
| Knowles Corporation | Voice AI |

### Medium Priority (12 companies) -- Pursue after initial traction

| Company | Category |
|---|---|
| Renesas Electronics | Semiconductor |
| Nordic Semiconductor | Semiconductor |
| Espressif Systems | Semiconductor |
| Xiaomi | Consumer Electronics |
| Bose | Consumer Electronics |
| Sonos | Consumer Electronics |
| ABB | Industrial IoT |
| Honeywell | Industrial IoT |
| Rockwell Automation | Industrial IoT |
| Fluent.ai | Voice AI |
| Jabra (GN Audio) | Wearables |
| Sony Audio | Wearables |
| Bragi | Wearables |

### Low Priority (1 company) -- Long-term / credibility play

| Company | Category |
|---|---|
| Apple | Wearables |

---

## Recommended Engagement Timeline

| Phase | Timeframe | Focus |
|---|---|---|
| **Phase 1: Validation** | Q2 2026 | Sensory, Picovoice, Syntiant, Knowles (voice AI peers validate the tech; potential early licensing) |
| **Phase 2: Semiconductor Partners** | Q2-Q3 2026 | STMicro, NXP, Infineon, Ambiq (get NC-SSM into chip reference designs) |
| **Phase 3: Korean Anchors** | Q3 2026 | Samsung, LG, Hyundai Mobis (leverage KR patent and domestic network) |
| **Phase 4: Industrial + Automotive** | Q3-Q4 2026 | Siemens, Continental, Bosch, Cerence (noise-robust differentiator) |
| **Phase 5: Scale** | 2027 | Consumer electronics, wearables, remaining targets |

---

## Key Conferences for In-Person Engagement

| Conference | Date (Typical) | Best Targets |
|---|---|---|
| **Embedded World** | March | STMicro, NXP, Infineon, Renesas, Ambiq |
| **Interspeech 2026** | September | Sensory, Picovoice, Fluent.ai, Cerence |
| **CES** | January | Samsung, LG, Bose, Sonos, Continental |
| **Hannover Messe** | April | Siemens, ABB, Bosch, Rockwell |
| **AutoSens** | September | Continental, Bosch, Cerence, Hyundai Mobis |
| **ICASSP** | June | Apple, Sony, Knowles (research credibility) |
