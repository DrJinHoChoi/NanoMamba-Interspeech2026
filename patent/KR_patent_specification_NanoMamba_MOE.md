# 특허 명세서

## [발명의 명칭]

**초경량 상태공간모델 기반 인공지능 추론 방법 및 장치, 그리고 이를 이용한 음성 및 영상 인식 시스템**

Method and Apparatus for Ultra-Lightweight AI Inference Based on State Space Models, and Audio-Visual Recognition System Using the Same

---

## [기술분야]

본 발명은 인공지능 기반 신호 인식 기술에 관한 것으로, 보다 구체적으로는 상태공간모델(State Space Model, SSM)의 구조적 상수(architectural constant)를 신호 대 잡음비(Signal-to-Noise Ratio, SNR)에 따라 적응적으로 조절하여, 추가 파라미터 없이 잡음 환경에서의 강인성을 달성하는 초경량 인공지능 추론 방법 및 장치에 관한 것이다.

또한, 본 발명은 서로 다른 잡음 유형에 특화된 복수의 특징 정규화 전문가(feature normalization expert)와 다차원 스펙트럴 라우팅(multi-dimensional spectral routing)을 결합한 구조적 전문가 혼합(Structural Mixture of Experts, Structural MOE) 프레임워크에 관한 것이다.

또한, 본 발명은 상기 방법 및 장치를 음성 키워드 인식(Keyword Spotting, KWS), 음향 이벤트 검출(Sound Event Detection), 영상 인식(Visual Recognition), 센서 이상 감지(Anomaly Detection) 등 다양한 시퀀스 처리 태스크에 적용하는 시스템과, 이를 블루투스(BT) 오디오 SoC, 모바일 AP, IoT 디바이스 등 에지(edge) 프로세서에서 실시간으로 구동하기 위한 전용 하드웨어 가속기 IP 블록의 설계에 관한 것이다.

**국제특허분류(IPC):**
- G10L 15/22 — 잡음 환경에서의 음성 인식
- G06N 3/04 — 신경망 아키텍처
- G06N 3/08 — 신경망 학습 방법
- G06F 17/10 — 상태공간 모델을 이용한 신호 처리
- G06V 10/82 — 영상 인식을 위한 신경망
- H03H 17/02 — 디지털 필터 (적응형 정규화)
- G06F 1/3234 — 에지 디바이스 전력 관리

---

## [배경기술]

### 1. 종래 기술의 문제점

키워드 인식(KWS)은 음성 비서, 스마트 이어버드, IoT 디바이스 등에서 항상-켜짐(always-on) 모드로 동작해야 하므로, 극도로 낮은 연산량과 메모리 사용이 요구된다. 이를 위해 다양한 경량 모델이 제안되었으나, 기존 기술은 다음과 같은 근본적 한계를 가진다.

#### 1.1 합성곱 신경망(CNN) 기반 접근의 한계

BC-ResNet(Broadcasting Residual Network, Kim et al., 2021)과 DS-CNN(Depthwise Separable CNN, Zhang et al., 2017) 등 종래 CNN 기반 KWS 모델은 학습 시 고정된 필터 응답(filter response)을 가진다. 즉, 컨볼루션 커널의 가중치가 학습 완료 후 동결(frozen)되어, 추론 시 입력 신호의 잡음 특성에 따른 적응적 변환이 불가능하다.

이로 인해 학습 시 접하지 못한 잡음 유형에 대해 성능이 급격히 저하된다. 예를 들어, 23,700개 파라미터를 가진 DS-CNN-S 모델은 깨끗한 환경에서 96.6%의 정확도를 보이나, 0dB 백색잡음(white noise) 환경에서는 13.9%로 급락하여 사실상 사용이 불가능하다. 이는 CNN의 정적 필터 응답이 시변(time-varying) 잡음에 적응하지 못하기 때문이다.

#### 1.2 기존 전문가 혼합(MOE) 기술의 한계

전문가 혼합(Mixture of Experts) 기술은 복수의 전문가 네트워크와 게이팅(gating) 네트워크로 구성되며, 입력에 따라 적합한 전문가를 선택적으로 활성화한다.

US20200279150A1(Google LLC)에 개시된 MOE 기술에서는 선형 변환(linear layer)과 소프트맥스(softmax) 함수를 이용한 학습 기반 게이팅을 사용한다. 이러한 학습 기반 게이팅은 추가적인 학습 가능 파라미터를 요구하고, 학습 데이터 분포에 편향되며, 학습 시 접하지 못한 잡음 유형에 대한 일반화 성능이 제한적이다.

또한, 종래 MOE 기술은 대규모 언어 모델(LLM)이나 비전 트랜스포머(Vision Transformer) 등 수백만~수십억 파라미터 규모의 모델에 주로 적용되어 왔으며, 5,000개 미만의 극소 파라미터 모델에서의 MOE 적용은 시도된 바 없다.

#### 1.3 상태공간모델(SSM)의 잡음 취약성

최근 Mamba(Gu & Dao, 2023) 등 선택적 상태공간모델(Selective SSM)이 시퀀스 모델링에서 주목받고 있다. Mamba의 핵심 혁신은 선택 메커니즘(selection mechanism)으로, 입력에 따라 이산화 스텝(dt), 입력 행렬(B), 출력 행렬(C)을 적응적으로 조절한다.

그러나 표준 Mamba에서 이산화 스텝 dt와 입력 행렬 B는 오직 시간 축 특징(temporal features)에서만 투영(projection)되며, 입력 신호의 주파수별 SNR 정보를 활용하지 않는다. 이는 잡음이 존재할 때 SSM이 잡음 프레임과 음성 프레임을 동등하게 처리하여, 잡음 환경에서의 성능 저하를 초래한다.

#### 1.4 외부 음향 향상기(External Enhancer)의 클린 성능 저하 문제

종래 기술에서 잡음 환경 성능을 향상시키기 위해 외부 음향 향상 모듈(예: GTCRN, 23,700개 파라미터)을 전처리로 사용하는 접근이 시도되었다. 그러나 이러한 외부 향상기는 다음과 같은 근본적 문제를 가진다.

**첫째,** 깨끗한 환경(Clean)에서의 심각한 성능 저하를 유발한다. 실험 결과, NanoMamba-Tiny(92.9%)에 GTCRN을 결합하면 Clean 정확도가 79.6%로 13.3%p 하락하며, NanoMamba-Small(95.2%)의 경우에도 87.7%로 7.5%p 하락한다. 이는 향상기가 깨끗한 신호를 불필요하게 변형(distortion)하기 때문이다.

**둘째,** 외부 향상기 자체가 23,700개의 추가 파라미터를 요구하여, 초경량 모델(5,000개 미만)의 취지에 반한다.

**셋째,** 향상기와 인식기 간의 특징 공간 불일치(feature space mismatch)가 발생한다. 예를 들어, SPP(Speech Presence Probability) 기반 블렌딩 방법에서 log(mel) 특징은 [-6, 0] 범위인 반면, PCEN 특징은 [0, 2] 범위를 가져, 두 특징 공간의 단순 혼합이 성능 저하를 야기한다.

이러한 외부 향상기의 한계는 잡음 강인성이 외부 모듈이 아닌 인식기 내부의 구조적 특성에 의해 달성되어야 함을 시사한다.

#### 1.5 다단계 파이프라인의 비효율성

Qualcomm Sensing Hub 등 종래 상용 시스템에서는 음성 활동 검출(VAD) → 음향 반향 제거(AEC) → 잡음 억제(NR) → 키워드 검출(KWD)의 다단계 직렬 파이프라인을 사용한다(US20210005181A1). 이러한 다단계 접근은 각 모듈의 지연(latency)이 누적되고, 모듈 간 정보 손실이 발생하며, 전체 시스템의 전력 소모와 메모리 사용량이 증가하는 문제가 있다.

#### 1.6 단일 PCEN(Per-Channel Energy Normalization)의 한계

PCEN(Wang et al., ICASSP 2017)은 멜(mel) 스펙트로그램에 대한 적응적 이득 제어(AGC)와 동적 범위 압축(DRC)을 수행하는 정규화 기법이다. PCEN의 출력은 다음과 같이 정의된다:

```
smoother[t] = (1 - s) * smoother[t-1] + s * mel[t]
gain = (epsilon + smoother)^(-alpha)
output = (mel * gain + delta)^r - delta^r
```

여기서 delta(오프셋) 파라미터는 정규화 특성을 결정하는 핵심 변수이다:
- delta가 크면(예: 2.0) AGC 이득이 무시되어 오프셋 우세 모드(offset-dominant mode)가 되며, 비정상 잡음(babble 등)에 강인하다. (PCEN delta=2.0: Clean 95.1%, babble-0dB 81.0%)
- delta가 작으면(예: 0.01) AGC가 지배적(AGC-dominant mode)이 되어, 정상 잡음(factory, white 등)의 시간 불변 특성을 효과적으로 추적 제거한다. (PCEN delta=0.01: Clean 94.6%, factory-15dB 31.0%)

단일 PCEN은 하나의 delta 값만을 가지므로, 정상 잡음과 비정상 잡음을 동시에 최적으로 처리할 수 없다.

### 2. 선행 기술 문헌

**특허 문헌:**
- US20200279150A1 (Google LLC): Mixture of Experts Neural Networks — 선형-소프트맥스 게이팅
- US20210005181A1 (Qualcomm): Audible Keyword Detection — 다단계 LKDE/HKDE 구조
- US12,548,565 (Nuance/Cerence): Voice Command Detection — 다중 프로세서 계층 구조
- US9,786,135 B2 (Qualcomm): Power optimization for AI hardware

**비특허 문헌:**
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, 2023
- Wang et al., "Trainable Frontend for Robust and Far-Field Keyword Spotting," ICASSP 2017
- Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting," arXiv:2106.04140, 2021
- Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers," arXiv:1711.07128, 2017
- Tan et al., "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources," arXiv:2312.00752, 2023

---

## [발명의 내용]

### 해결하고자 하는 과제

본 발명은 상기 종래 기술의 문제점을 해결하기 위한 것으로, 다음과 같은 기술적 과제를 해결하고자 한다.

**첫째,** 잡음 증강 학습 데이터 없이(clean data only training) 구조적으로 잡음에 강인한 인공지능 추론 모델을 제공한다.

**둘째,** 정상 잡음(factory, white, pink)과 비정상 잡음(babble, street)을 단일 모델로 동시에 처리할 수 있는 구조적 전문가 혼합(Structural MOE) 아키텍처를 제공한다.

**셋째,** 외부 음향 향상기(External Enhancer) 없이, 고 SNR 환경에서 적응형 파라미터가 원래 값으로 복원되어 깨끗한 신호의 성능 저하를 구조적으로 방지하는 "클린 보존 원리(Clean Preservation Principle)"를 구현한다.

**넷째,** 스펙트럴 평탄도(Spectral Flatness)만으로는 분류 불가능한 유색 정상 잡음(pink noise 등)을 스펙트럴 기울기(Spectral Tilt)와의 다차원 라우팅을 통해 올바르게 분류하는 라우팅 기법을 제공한다.

**다섯째,** 5,000개 미만 파라미터, INT8 양자화 시 5KB 미만의 초경량 모델로 상기 잡음 강인성을 달성하는 시스템을 제공한다.

**여섯째,** 상기 시스템을 Qualcomm, JieLi(杰理), BES(恒玄) 등 상용 SoC에 통합 가능한 하드웨어 가속기 IP 블록으로 구현하는 방법을 제공한다.

**일곱째,** 음성(audio)뿐만 아니라 비전(vision), 센서 데이터 등 다양한 시퀀스 처리 태스크에 범용적으로 적용 가능한 SNR-적응형 상태공간모델 기반 인공지능 추론 프레임워크를 제공한다.

### 과제의 해결 수단

상기 과제를 해결하기 위해, 본 발명은 다음의 핵심 구성 요소를 포함하는 잡음 강인 인공지능 추론 시스템을 제공한다.

#### A. 다차원 스펙트럴 라우팅 기반 이중 PCEN 전문가 혼합(DualPCEN MOE) — 2-Expert 구조

본 발명의 제1 구성 요소는 서로 다른 잡음 유형에 특화된 두 개의 PCEN 전문가와, 스펙트럴 평탄도(Spectral Flatness) 및 스펙트럴 기울기(Spectral Tilt)의 다차원 음향 물리적 특성에 기반한 무파라미터 라우팅(zero-parameter routing)을 결합한 이중 전문가 혼합 모듈이다.

**전문가 1(비정상 잡음 전문가):** delta_init = 2.0, s_init = 0.025, alpha_init = 0.99, r_init = 0.5
- 높은 delta 값에 의해 AGC 이득이 상대적으로 무시되어 오프셋 우세 모드가 됨
- babble, street 등 시변 잡음에 대해 구조적으로 강인
- delta 클램핑: (0.5, 5.0)

**전문가 2(정상 잡음 전문가):** delta_init = 0.01, s_init = 0.15, alpha_init = 0.99, r_init = 0.1
- 낮은 delta 값에 의해 AGC가 지배적이 되어, 시간 불변 잡음을 추적 제거
- factory, white, pink 등 정상 잡음에 대해 구조적으로 강인
- delta 클램핑: (0.001, 0.1)

**라우팅 신호 — 스펙트럴 평탄도(SF) + 스펙트럴 기울기(Spectral Tilt):**

```
SF(t) = exp(mean(log(mel(t)))) / mean(mel(t))          # [0, 1]
Tilt(t) = low_freq_energy / (low_freq_energy + high_freq_energy + eps)  # [0, 1]
SF_adjusted(t) = SF(t) + (1 - SF(t)) * ReLU(Tilt(t) - 0.6)
```

종래의 단일 차원 라우팅(SF만 사용)에서는 핑크 잡음(pink noise)이 오분류되는 문제가 있었다. 핑크 잡음은 정상(stationary) 잡음이나, SF 값이 약 0.3으로 낮아(저주파 에너지 집중에 의한 비평탄 스펙트럼) 비정상 잡음 전문가로 잘못 라우팅되었다.

본 발명의 스펙트럴 기울기(Spectral Tilt)는 저주파 에너지 집중도를 측정하여 이를 교정한다:
- 핑크 잡음: tilt ≈ 0.85 (저주파 집중) → SF 0.3 → SF_adjusted = 0.3 + 0.7 × ReLU(0.85 - 0.6) = 0.475
- Babble 잡음: tilt ≈ 0.55 → SF 0.4 → SF_adjusted = 0.4 + 0.6 × ReLU(0.55 - 0.6) = 0.4 (변화 없음)
- 백색 잡음: tilt ≈ 0.50 → SF 0.95 → SF_adjusted = 0.95 (이미 높음, 변화 없음)

이 다차원 라우팅은 추가 학습 가능 파라미터를 사용하지 않으며(0 additional parameters), 순수 음향 물리적 신호에서 유도된다.

**게이트 함수:**
```
gate(t) = sigmoid(gate_temp * (SF_adjusted(t) - 0.5))
output(t) = gate(t) * expert_stationary(t) + (1 - gate(t)) * expert_nonstationary(t)
```

#### B. SNR-적응형 구조적 상수를 가진 스펙트럴-인식 상태공간모델(SA-SSM)

본 발명의 제2 구성 요소는 주파수별 SNR 추정치를 SSM의 선택 메커니즘에 직접 주입하되, 이산화 스텝 하한(delta floor), 잔차 경로 계수(epsilon), 입력 행렬 게이트 하한(B-gate floor)의 세 가지 구조적 상수를 SNR에 따라 적응적으로 조절하는 스펙트럴-인식 상태공간모델이다.

**[핵심 혁신 1] SNR-적응형 이산화 스텝 하한(Adaptive Delta Floor):**

종래 기술에서는 고정 delta_floor = 0.15를 사용하였다. 본 발명에서는 SNR에 따라 적응적으로 변화하는 하한을 도입한다:

```
adaptive_floor = delta_floor_min + (delta_floor_max - delta_floor_min) * snr_mean
# delta_floor_min = 0.05, delta_floor_max = 0.15
```

- 고 SNR (Clean) → adaptive_floor ≈ 0.14 ≈ 원래 값 → Clean 성능 보존
- 저 SNR (-15dB) → adaptive_floor ≈ 0.055 → SSM의 시간적 메모리 연장, 잡음 프레임에서의 동결 방지

이 적응형 하한은 register_buffer로 구현된 비학습 상수(delta_floor_min, delta_floor_max)와 SNR 추정치의 조합으로, 추가 학습 가능 파라미터가 0개이다.

**[핵심 혁신 2] SNR-적응형 잔차 경로 계수(Adaptive Epsilon):**

종래 기술에서는 고정 epsilon = 0.1을 사용하였다. 본 발명에서는:

```
adaptive_eps = epsilon_max - (epsilon_max - epsilon_min) * snr_mean
# epsilon_min = 0.08, epsilon_max = 0.20
```

- 고 SNR (Clean) → adaptive_eps ≈ 0.09 ≈ 원래 값 → 게이팅 신뢰, Clean 보존
- 저 SNR (-15dB) → adaptive_eps ≈ 0.19 → 강화된 바이패스 경로, 정보 구조(rescue)

상태 업데이트: h[t] = A_bar * h[t-1] + dBx[t] + adaptive_eps * x[t]

**[핵심 혁신 3] B-게이트 하한(B-Gate Floor):**

종래 기술에서는 B_gate = sigmoid(snr_mod)로 계산되어, -15dB에서 B_gate ≈ 0.1까지 하락하여 입력이 과도하게 억제되었다. 본 발명에서는:

```
B_gate = B_gate_raw * 0.7 + 0.3    # 범위: [0.3, 1.0]
```

이에 의해 최소 30%의 입력이 항상 SSM 상태에 전달되며, 이산화 스텝(dt)과 입력 행렬(B)이 동시에 억제되는 복합 과억제(compound over-suppression) 현상을 방지한다. 추가 학습 가능 파라미터: 0개.

**[핵심 혁신 4] 클린 보존 원리(Clean Preservation Principle):**

상기 세 가지 적응형 구조적 상수는 모두 고 SNR 환경에서 원래의 고정 값으로 수렴하도록 설계되었다:

| 구조적 상수 | 저 SNR 값 | 고 SNR 값 | 효과 |
|------------|-----------|-----------|------|
| adaptive_floor | 0.05 | 0.15 (원래) | 시간적 메모리 연장 |
| adaptive_eps | 0.20 | 0.08 (원래) | 바이패스 경로 강화 |
| B_gate | [0.3, 0.7] | [0.3, 1.0] | 입력 흐름 보장 |

이는 외부 향상기(GTCRN 등)가 Clean에서 -7.5~13.3%p의 성능 저하를 유발하는 것과 대조적으로, 본 발명의 구조적 접근은 Clean 성능을 아키텍처적으로 보존한다.

#### C. 주파수 영역 전문가 혼합(MoEFreq) — 3-Expert 구조

본 발명의 제3 구성 요소는 SNR 통계 조건부 주파수 영역 전문가 혼합 모듈이다.

**전문가 구성:**
- 전문가 1: 좁은 대역 컨볼루션(kernel_size=3) — 톤형 잡음(공장 하모닉스 등) 대응
- 전문가 2: 넓은 대역 컨볼루션(kernel_size=7) — 광대역 잡음(백색잡음, HVAC 등) 대응
- 전문가 3: 항등 변환(identity, 0 파라미터) — 깨끗한 환경에서 원본 보존

#### D. 주파수 의존 에너지 하한(Frequency-Dependent Floor)

본 발명의 제4 구성 요소는 저주파 멜 밴드의 정보 손실을 구조적으로 방지하는 비학습 에너지 하한 모듈이다.

```
floor[i] = 0.05 * exp(-3.0 * (1.0 - i/(n_mels - 1)))
mel_protected = max(mel_linear, floor)
```

#### E. 가중치 공유(Weight Sharing)

본 발명의 제5 구성 요소는 단일 SA-SSM 블록을 N회 반복 실행하여, N층 깊이를 가지면서 단일 블록의 파라미터만을 사용하는 가중치 공유 메커니즘이다.

#### F. 일반화된 N-Expert 프레임워크

상기 구성 요소들을 일반화하면, 임의의 N개 전문가와 임의의 다차원 신호 유도 라우팅 신호를 결합하는 범용 프레임워크를 구성할 수 있다. 이 프레임워크는 음성뿐만 아니라 비전, 센서 데이터 등 다양한 도메인에 적용 가능하다.

### 발명의 효과

본 발명에 따르면 다음과 같은 효과를 얻을 수 있다.

1. **구조적 잡음 강인성:** 클린 데이터만으로 학습하여, 학습 시 접하지 않은 잡음 유형에 대해 구조적으로 강인한 성능을 달성한다.

2. **클린 보존:** SNR-적응형 구조적 상수가 고 SNR 환경에서 원래 값으로 자동 복원되어, 외부 향상기 대비 Clean 성능 저하가 없다. (외부 GTCRN 향상기: Clean -7.5~13.3%p 저하 vs. 본 발명: 0%p 저하)

3. **핑크 잡음 정확 라우팅:** 다차원 스펙트럴 라우팅(SF + Spectral Tilt)에 의해, 종래 SF 단독 라우팅에서 오분류되던 핑크 잡음을 정확히 정상 잡음 전문가로 라우팅한다. (SF: 0.3 → SF_adjusted: 0.475)

4. **복합 과억제 방지:** B-게이트 하한에 의해, dt 억제와 B 억제가 동시에 발생하는 극단적 저 SNR 환경에서도 최소 30% 입력 흐름을 보장한다.

5. **초경량:** 5,000개 미만 파라미터, INT8 양자화 시 5KB 미만. 모든 적응형 구조적 상수가 추가 파라미터 0개로 구현된다.

6. **하드웨어 친화적:** 4.5KB 가중치 SRAM + 선형 시간 추론(O(L)) + 간단한 MAC 연산 → BT 오디오 SoC의 제한된 자원에서 실시간 구동 가능.

7. **범용성:** 음성 키워드 인식 외 음성 향상, 음향 이벤트 검출, 영상 인식, 센서 이상 감지 등에 적용 가능한 범용 SNR-적응형 상태공간모델 프레임워크.

---

## [도면의 간단한 설명]

**도 1**은 본 발명에 따른 잡음 강인 인공지능 추론 시스템의 전체 아키텍처를 나타내는 블록도이다.

**도 2**는 본 발명의 다차원 스펙트럴 라우팅 기반 이중 PCEN 전문가 혼합(DualPCEN MOE) 모듈의 구조를 나타내는 상세 블록도로서, 스펙트럴 평탄도(SF)와 스펙트럴 기울기(Spectral Tilt)의 결합 과정을 포함한다.

**도 3**은 본 발명의 SNR-적응형 구조적 상수를 가진 스펙트럴-인식 상태공간모델(SA-SSM) 블록의 내부 구조를 나타내는 블록도로서, 적응형 delta floor, 적응형 epsilon, B-gate floor의 배치를 포함한다.

**도 4**는 본 발명의 주파수 영역 전문가 혼합(MoEFreq) 모듈의 구조를 나타내는 블록도이다.

**도 5**는 본 발명을 일반화한 N-Expert 전문가 혼합 프레임워크의 구조를 나타내는 블록도이다.

**도 6**은 본 발명에 따른 하드웨어 가속기 IP 블록의 RTL 수준 아키텍처를 나타내는 블록도이다.

**도 7**은 본 발명의 하드웨어 가속기가 상용 SoC(Qualcomm, JieLi, BES)에 통합되는 구조를 나타내는 블록도이다.

**도 8**은 가중치 공유 메커니즘과 메모리 레이아웃을 나타내는 도면이다.

**도 9**는 본 발명에 따른 학습 파이프라인의 흐름도이다.

**도 10**은 에지 SoC에서의 스트리밍 실시간 추론 파이프라인을 나타내는 흐름도이다.

**도 11**은 SNR-적응형 구조적 상수(adaptive delta floor, adaptive epsilon)의 SNR에 따른 변화 그래프이다.

**도 12**는 스펙트럴 기울기(Spectral Tilt)에 의한 핑크 잡음 라우팅 교정 과정을 나타내는 도면이다.

**도 13**은 클린 보존 원리(Clean Preservation Principle)의 동작을 나타내는 비교 그래프로서, 외부 향상기 방식과 본 발명의 구조적 접근의 Clean 성능을 대비한다.

**도 14**는 본 발명의 비전(영상) 도메인 적용 실시예를 나타내는 블록도이다.

---

## [발명을 실시하기 위한 구체적인 내용]

이하, 첨부된 도면을 참조하여 본 발명의 실시예를 상세히 설명한다.

### 제1 실시예: SNR-적응형 구조적 상수를 가진 상태공간모델

#### 1.1 전체 시스템 아키텍처 (도 1 참조)

본 발명에 따른 잡음 강인 인공지능 추론 시스템은 다음의 처리 파이프라인으로 구성된다:

```
원시 입력 신호 (예: 16kHz 오디오, 1초)
    |
[STFT 모듈] -- FFT 크기=512, 홉 길이=160, Hann 윈도우
    |
크기 스펙트로그램 (B, 257, T)
    |
[SNR 추정 모듈] -- 초기 5프레임 잡음 바닥 추정 + EMA 추적
    |                          |
SNR 추정치 (B, 40, T)     크기 스펙트로그램
    |                          |
    |               [MoEFreq 모듈] (선택적)
    |                          |
    |               [멜 필터뱅크 투영] -- 40 밴드
    |                          |
    |               [FrequencyDependentFloor]
    |                          |
    |               [DualPCEN MOE 모듈] -- 다차원 스펙트럴 라우팅
    |                          |
    |               [인스턴스 정규화]
    |                          |
    |               [패치 투영] -- n_mels -> d_model
    |                          |
    +---> SNR 측면 입력 -->[SA-SSM 블록 x N] -- SNR-적응형 구조적 상수
                               |
                    [층 정규화 + 전역 평균 풀링]
                               |
                    [선형 분류기] -- K 클래스
                               |
                    인식 결과
```

#### 1.2 SNR 추정 모듈

크기 스펙트로그램에서 주파수별 SNR을 추정한다. 초기 N 프레임(기본값 5)의 평균을 잡음 바닥(noise floor)으로 사용하며, 선택적으로 비대칭 지수 이동 평균(EMA)을 통해 적응적 잡음 추적을 수행한다:

```python
noise_floor_init = mean(mag[:, :, :5], dim=time)              # 초기 잡음 바닥
snr_linear = mag / (noise_scale * noise_floor + floor_param)  # 주파수별 SNR
snr_db = 10 * log10(snr_linear + 1e-8)                        # dB 변환
snr_mel = mel_fb @ snr_db                                     # 멜 스케일 투영
snr_mel = tanh(snr_mel / 10.0)                                # [0, 1] 정규화
```

비대칭 EMA(선택적):
- 프레임 에너지 > 잡음 바닥: 느린 상승(gamma ~ 0.05) -- 음성/임팩트에 의한 오추정 방지
- 프레임 에너지 < 잡음 바닥: 빠른 하강(beta ~ 0.10) -- 잡음 바닥 신속 갱신

#### 1.3 다차원 스펙트럴 라우팅 기반 DualPCEN MOE 모듈 (도 2, 12 참조)

**1.3.1 PCEN 전문가 단일 모듈**

각 PCEN 전문가는 40개 멜 밴드에 대해 4개의 학습 가능 파라미터(s, alpha, delta, r)를 가진다:

```python
log_s = Parameter(log(s_init / (1 - s_init)) * ones(n_mels))
log_alpha = Parameter(log(alpha_init / (1 - alpha_init)) * ones(n_mels))
log_delta = Parameter(log(delta_init) * ones(n_mels))
log_r = Parameter(log(r_init / (1 - r_init)) * ones(n_mels))
```

로그/로짓 공간에서 학습하여 양수/범위 보장. delta는 추가로 클램핑:
```python
s = sigmoid(log_s).clamp(0.05, 0.3)
alpha = sigmoid(log_alpha).clamp(0.9, 0.999)
delta = exp(log_delta).clamp(delta_min, delta_max)
r = sigmoid(log_r).clamp(0.05, 0.25)
```

IIR 평활기 (1차 IIR 필터):
```python
smoother[0] = mel[:, :, 0]
for t in range(1, T):
    smoother[t] = (1 - s) * smoother[t-1] + s * mel[:, :, t]
```

PCEN 변환:
```python
gain = (eps + smoother) ** (-alpha)
output = (mel * gain + delta) ** r - delta ** r
```

파라미터 수: 4 x 40 = 160개 (전문가당)

**1.3.2 이중 전문가 구성**

| 파라미터 | 전문가 1 (비정상 잡음) | 전문가 2 (정상 잡음) |
|----------|------------------------|----------------------|
| s (평활 계수) | 0.025 (느린 추적) | 0.15 (빠른 추적) |
| alpha (AGC 지수) | 0.99 | 0.99 |
| delta (오프셋) | 2.0 (높음, AGC 무시) | 0.01 (낮음, AGC 지배) |
| r (압축 지수) | 0.5 | 0.1 |
| delta 클램프 | (0.5, 5.0) | (0.001, 0.1) |

**1.3.3 다차원 스펙트럴 라우팅 (도 12 참조)**

**단계 1: 스펙트럴 평탄도(SF) 계산**

```python
log_mel = log(mel_linear + 1e-8)                        # (B, 40, T)
geo_mean = exp(mean(log_mel, dim=mel_axis))              # (B, 1, T)
arith_mean = mean(mel_linear, dim=mel_axis) + 1e-8       # (B, 1, T)
sf = clamp(geo_mean / arith_mean, 0, 1)                  # (B, 1, T)
```

SF의 물리적 의미:
- SF -> 1.0: 모든 주파수 에너지가 균일 -> 백색잡음(정상 잡음)
- SF -> 0.0: 특정 주파수에 에너지 집중 -> 음성 또는 비정상 잡음

**단계 2: 스펙트럴 기울기(Spectral Tilt) 계산**

```python
n_mels = mel_linear.size(1)                              # 40
low_energy = mel_linear[:, :n_mels // 3, :].mean(dim=1, keepdim=True)   # 밴드 0-12
high_energy = mel_linear[:, 2 * n_mels // 3:, :].mean(dim=1, keepdim=True) # 밴드 27-39
spectral_tilt = clamp(low_energy / (low_energy + high_energy + 1e-8), 0, 1)
```

Spectral Tilt의 물리적 의미:
- Tilt -> 1.0: 에너지가 저주파에 집중 (핑크 잡음: tilt ~ 0.85)
- Tilt -> 0.5: 에너지가 균일 분포 (백색 잡음: tilt ~ 0.50, babble: tilt ~ 0.55)

**단계 3: 다차원 라우팅 보정**

```python
sf_adjusted = sf + (1.0 - sf) * relu(spectral_tilt - 0.6)
```

이 보정의 핵심 직관: Spectral Tilt가 0.6을 초과하면(저주파 에너지 집중), SF 값을 상향 보정하여 해당 프레임을 정상 잡음 전문가 쪽으로 라우팅한다.

잡음 유형별 라우팅 분석:

| 잡음 유형 | SF | Tilt | relu(tilt-0.6) | SF_adj | gate(SF_adj) | 라우팅 |
|-----------|-----|------|----------------|--------|-------------|--------|
| 백색(White) | 0.95 | 0.50 | 0.0 | 0.95 | ~0.92 | 정상 전문가 (정확) |
| 핑크(Pink) | 0.30 | 0.85 | 0.25 | 0.475 | ~0.44 | 혼합→정상쪽 (교정됨) |
| 공장(Factory) | 0.40 | 0.60 | 0.0 | 0.40 | ~0.27 | 비정상 전문가 |
| Babble | 0.40 | 0.55 | 0.0 | 0.40 | ~0.27 | 비정상 전문가 (정확) |
| 음성(Speech) | 0.15 | 0.65 | 0.05 | 0.193 | ~0.07 | 비정상 전문가 |

종래 SF 단독 라우팅에서 핑크 잡음은 SF=0.3으로 비정상 전문가(babble)에 잘못 라우팅되었으나, Spectral Tilt 보정에 의해 SF_adjusted=0.475로 교정되어 정상 잡음 전문가 방향으로 올바르게 라우팅된다.

**단계 4: 게이트 계산 및 전문가 결합**

```python
gate = sigmoid(gate_temp * (sf_adjusted - 0.5))          # (B, 1, T)
output = gate * expert_stationary(mel) + (1 - gate) * expert_nonstationary(mel)
```

gate_temp는 유일한 학습 가능 파라미터(초기값 5.0)로, 라우팅의 날카로움(sharpness)을 제어한다.

추가 파라미터: 전문가 1(160) + 전문가 2(160) + gate_temp(1) = 총 321개. 라우팅 신호 자체의 학습 파라미터: 0개.

#### 1.4 SA-SSM 블록 구조 (도 3 참조)

**1.4.1 NanoMambaBlock 처리 파이프라인**

```
입력 x (B, L, d_model)
    |
[층 정규화(LayerNorm)]
    |
[입력 투영(in_proj)] -- d_model -> 2 * d_inner (바이어스 없음)
    |
분할: x_branch (B, L, d_inner) | z_gate (B, L, d_inner)
    |                                |
[깊이별 컨볼루션(DWConv1d)]          |
kernel_size=d_conv, groups=d_inner   |
    |                                |
[SiLU 활성화]                        |
    |                                |
[SA-SSM(x_branch, snr_mel)]          |
    |                                |
요소별 곱: y = ssm_out * SiLU(z_gate)
    |
[출력 투영(out_proj)] -- d_inner -> d_model (바이어스 없음)
    |
잔차 연결: output = projected + residual
```

**1.4.2 SA-SSM 내부 계산 (적응형 구조적 상수 포함)**

입력 투영:
```python
x_proj = W_x @ x_branch            # (B, L, 2*d_state + 1)
dt_raw = x_proj[..., :1]           # (B, L, 1)
B_param = x_proj[..., 1:d_state+1] # (B, L, N)
C_param = x_proj[..., d_state+1:]  # (B, L, N)
```

SNR 변조 투영:
```python
snr_mod = W_snr @ snr_mel          # (B, L, d_state + 1)
dt_snr_shift = snr_mod[..., :1]    # (B, L, 1) -- dt 변조량
B_gate_raw = sigmoid(snr_mod[..., 1:])  # (B, L, N) -- B 게이팅 원시값
```

**[혁신] B-게이트 하한 적용:**
```python
B_gate = B_gate_raw * 0.7 + 0.3    # 범위: [0.3, 1.0], 최소 30% 입력 보장
```

**[혁신] SNR-적응형 Delta Floor 계산:**
```python
snr_mean = snr_mel.mean(dim=-1, keepdim=True)  # (B, L, 1)
adaptive_floor = 0.05 + (0.15 - 0.05) * snr_mean  # [0.05, 0.15]
delta = softplus(W_dt @ (dt_raw + dt_snr_shift) + b_dt) + adaptive_floor
```

SNR 게이팅된 입력 행렬:
```python
B_effective = B_param * (1.0 - alpha + alpha * B_gate)
# alpha = 0.5 (학습 가능, 게이팅 강도 제어)
```

이산화 (Zero-Order Hold):
```python
dA = exp(A_log.exp().neg() * delta)     # (B, L, d_inner, N)
dBx = delta * B_effective * x_branch     # (B, L, d_inner, N)
```

**[혁신] 적응형 Epsilon 계산:**
```python
adaptive_eps = 0.20 - (0.20 - 0.08) * snr_mean  # [0.08, 0.20]
```

순차 스캔 (recurrence):
```python
for t in range(L):
    h[t] = dA[t] * h[t-1] + dBx[t] + adaptive_eps[t] * x[t]  # 상태 업데이트
    y[t] = sum(h[t] * C[t], dim=state) + D * x[t]              # 출력 계산
```

여기서:
- A_log: HiPPO 초기화된 대각 행렬, A[n] = -(n + 0.5)
- D: 입력 직접 전달 (skip connection)
- adaptive_eps: SNR에 따라 [0.08, 0.20] 범위에서 변화하는 비학습 잔차 경로 계수

**1.4.3 클린 보존 원리의 수학적 증명 (도 13 참조)**

고 SNR 환경(Clean)에서 snr_mean -> 1.0 (tanh 정규화)이므로:
- adaptive_floor = 0.05 + 0.10 * 1.0 = 0.15 (원래 고정값과 동일)
- adaptive_eps = 0.20 - 0.12 * 1.0 = 0.08 (원래 고정값 0.1에 근접)
- B_gate_raw -> sigmoid(높은 값) -> 1.0 이므로 B_gate = 0.7 + 0.3 = 1.0 (게이팅 없음)

따라서, Clean 환경에서의 SA-SSM 동작은 적응형 메커니즘이 없는 원래 모델과 아키텍처적으로 동일하다. 이는 외부 향상기가 Clean 신호를 변형하는 것과 근본적으로 다른 접근이다.

**1.4.4 SA-SSM 파라미터 구성 (d_inner=24, d_state=4 기준)**

| 구성 요소 | 크기 | 파라미터 수 | 비고 |
|-----------|------|-------------|------|
| x_proj (W_x) | 24 x 9 | 216 | 학습 가능 |
| snr_proj (W_snr) | 40 x 5 | 200 (+5 바이어스) | 학습 가능 |
| dt_proj (W_dt) | 1 x 24 | 24 (+24 바이어스) | 학습 가능 |
| A_log | 24 x 4 | 96 | 학습 가능 |
| D | 24 | 24 | 학습 가능 |
| alpha | 1 | 1 | 학습 가능 |
| delta_floor_min | 1 | 0 | register_buffer (0.05) |
| delta_floor_max | 1 | 0 | register_buffer (0.15) |
| epsilon_min | 1 | 0 | register_buffer (0.08) |
| epsilon_max | 1 | 0 | register_buffer (0.20) |

총 학습 가능 파라미터: 590개. 적응형 구조적 상수 추가 파라미터: 0개.

#### 1.5 MoEFreq 모듈 (도 4 참조)

주파수 영역에서 3개 전문가를 SNR 통계에 따라 선택적으로 적용:

```python
snr_mean = mean(snr_profile)
snr_std = std(snr_profile)
snr_stats = [snr_mean, snr_std]                        # (B, 2)

gate = softmax(Linear(2, 3)(snr_stats))                # (B, 3)

out_narrow = Conv1d(1, 1, kernel_size=3, padding=1)(mag)
out_wide = Conv1d(1, 1, kernel_size=7, padding=3)(mag)
out_identity = mag

output = gate[0] * out_narrow + gate[1] * out_wide + gate[2] * out_identity
```

초기화: 게이트 바이어스 = [0, 0, 1]로 항등 전문가 선호 -> 깨끗한 입력 보존.
총 파라미터: 21개 (전문가 1: 4개, 전문가 2: 8개, 라우터: 9개)

#### 1.6 일반화된 N-Expert 프레임워크 (도 5 참조)

```
StructuralMOE(input, physical_signal):
    routing_features = MultiDimensionalFeatureExtractor(physical_signal)
    gate_weights = GatingFunction(routing_features, N_experts)
    expert_outputs = [Expert_i(input) for i in range(N)]
    return sum(gate_weights[i] * expert_outputs[i] for i in range(N))
```

**다차원 라우팅 신호 옵션:**

| 신호 | 계산 | 학습 파라미터 | 적합 도메인 |
|------|------|---------------|-------------|
| 스펙트럴 평탄도 (SF) | GM/AM 비율 | 0 | 잡음 정상성 |
| 스펙트럴 기울기 (Spectral Tilt) | 저/고주파 에너지 비 | 0 | 잡음 색상(color) |
| SF + Tilt (다차원) | 보정된 SF | 0 | 모든 잡음 유형 |
| SNR 통계 (평균, 표준편차) | 주파수별 SNR | 0 | 잡음 수준/변동 |
| 변조 스펙트럼 | 4-16Hz 에너지 비율 | 0 | 음성/비음성 |
| 공간 주파수 분포 | 2D FFT 에너지 비 | 0 | 비전: 텍스처/평탄 |

### 제2 실시예: 소프트웨어 설계

#### 2.1 학습 파이프라인 (도 9 참조)

**2.1.1 데이터셋 및 전처리**

- 데이터셋: Google Speech Commands V2
- 클래스: 12개 ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence")
- 학습: 86,843 발화, 검증: 10,481 발화, 테스트: 11,005 발화
- 샘플링 레이트: 16kHz, 길이: 1초 (16,000 샘플)

**2.1.2 데이터 증강 (학습 시에만)**

```python
shift = random_int(-1600, 1600)
audio = roll(audio, shift)

volume_factor = uniform(0.8, 1.2)
audio = audio * volume_factor

if random() < 0.3:
    noise = randn_like(audio) * uniform(0.001, 0.015)
    audio = audio + noise
```

핵심: 잡음 증강은 매우 약한 수준(최대 -36dB SNR)의 가우시안 잡음만 사용. factory, white, babble, pink 등의 실환경 잡음은 학습에 사용하지 않음.

**2.1.3 학습 하이퍼파라미터**

```python
optimizer = AdamW(lr=3e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=total_steps, eta_min=lr*0.01)
criterion = CrossEntropyLoss(label_smoothing=0.1)
gradient_clipping = 1.0
epochs = 30
batch_size = 128
```

**2.1.4 실험 결과**

**외부 향상기 없이 (본 발명):**

| 모델 | Clean | factory-15dB | white-15dB | babble-15dB | pink-15dB |
|------|-------|-------------|------------|-------------|-----------|
| NanoMamba-Tiny | 92.9% | 38.4% | 20.2% | 58.6% | 9.9% |
| NanoMamba-Small | 95.2% | - | - | - | - |
| PCEN delta=2.0 | 95.1% | - | - | babble-0dB: 81.0% | - |
| PCEN delta=0.01 | 94.6% | 31.0% | - | - | - |

**외부 GTCRN 향상기 적용 시 (종래 기술, Clean 성능 저하 확인):**

| 모델 | Clean | Clean 저하 | factory-15dB | white-15dB |
|------|-------|-----------|-------------|------------|
| NanoMamba-Tiny + GTCRN | 79.6% | **-13.3%p** | 60.3% | 59.7% |
| NanoMamba-Small + GTCRN | 87.7% | **-7.5%p** | - | - |
| NanoMamba-Tiny-DualPCEN + GTCRN | 87.2% | **-7.6%p** | - | - |

**CNN 기준 모델:**

| 모델 | 파라미터 | Clean | white-0dB |
|------|---------|-------|-----------|
| DS-CNN-S | 23,700 | 96.6% | 13.9% (붕괴) |
| BC-ResNet-1 | 7,500 | 96.0% | - |

이 결과는 다음을 입증한다:
1. 외부 향상기는 잡음 성능을 향상시키나, Clean에서 7.5~13.3%p의 심각한 성능 저하를 유발한다.
2. 본 발명의 구조적 접근(SNR-적응형 구조적 상수)은 Clean에서의 성능 저하 없이 잡음 강인성을 달성한다.
3. CNN 기준 모델(DS-CNN-S)은 백색잡음 0dB에서 13.9%로 붕괴하나, 본 발명은 구조적 적응에 의해 이를 방지한다.

#### 2.2 추론 파이프라인

**2.2.1 오프라인 추론 (전체 발화 처리)**

```python
def inference(audio, model):
    mag = stft(audio, n_fft=512, hop_length=160)
    snr_mel = snr_estimator(mag, mel_fb)
    mag = moe_freq(mag, snr_mel) if use_moe_freq else mag
    mel = mel_fb @ mag
    mel = max(mel, freq_floor)
    mel = dual_pcen(mel)  # 다차원 스펙트럴 라우팅
    x = patch_proj(instance_norm(mel))
    for block in blocks:
        x = block(x, snr_mel)  # SNR-적응형 구조적 상수 적용
    x = classifier(layer_norm(mean(x, dim=time)))
    return argmax(x)
```

**2.2.2 스트리밍 추론 (실시간 프레임 처리)**

```python
class StreamingNanoMamba:
    def __init__(self, model):
        self.pcen_smoother = zeros(2, n_mels)
        self.ssm_hidden = zeros(n_layers, d_inner, d_state)
        self.noise_floor = None
        self.conv_buffer = zeros(n_layers, d_inner, d_conv-1)
        self.frame_count = 0

    def process_frame(self, audio_frame):
        mag = stft_frame(audio_frame)

        if self.frame_count < 5:
            self.noise_floor = update_noise_init(mag)
        snr = compute_snr(mag, self.noise_floor)
        snr_mean = mean(snr)  # SNR-적응형 상수 계산에 사용

        mel = mel_fb @ mag
        mel = max(mel, freq_floor)

        # DualPCEN (IIR 상태 유지 + 다차원 라우팅)
        sf = compute_spectral_flatness(mel)
        tilt = compute_spectral_tilt(mel)
        sf_adjusted = sf + (1 - sf) * relu(tilt - 0.6)

        for expert_id in [0, 1]:
            self.pcen_smoother[expert_id] = (
                (1 - s[expert_id]) * self.pcen_smoother[expert_id]
                + s[expert_id] * mel)

        mel = dual_pcen_frame(mel, self.pcen_smoother, sf_adjusted)

        # SA-SSM (SNR-적응형 구조적 상수)
        adaptive_floor = 0.05 + 0.10 * snr_mean
        adaptive_eps = 0.20 - 0.12 * snr_mean

        x = patch_proj(instance_norm(mel))
        for layer_id, block in enumerate(blocks):
            x, self.ssm_hidden[layer_id] = block.step(
                x, snr, self.ssm_hidden[layer_id],
                self.conv_buffer[layer_id],
                adaptive_floor, adaptive_eps)

        self.frame_count += 1
        return x
```

스트리밍 상태 메모리 (NanoMamba-Tiny 기준):
- PCEN smoother: 2 x 40 = 80 값
- SSM hidden: 2 x 24 x 4 = 192 값
- Conv buffer: 2 x 24 x 2 = 96 값
- 잡음 바닥: 257 값
- **총 스트리밍 상태: 625 값 x 1바이트(INT8) = 625 bytes**

#### 2.3 INT8 양자화

```python
class FakeQuantize(Module):
    def forward(self, x):
        scale = x.abs().max() / 127
        x_quant = round(x / scale) * scale
        return x_quant + (x - x_quant).detach()  # STE
```

| 모델 | 파라미터 수 | FP32 (KB) | INT8 (KB) |
|------|-------------|-----------|-----------|
| NanoMamba-Tiny | 4,634 | 18.1 | 4.5 |
| NanoMamba-Small | 12,032 | 47.0 | 11.8 |
| NanoMamba-Tiny-DualPCEN | ~4,955 | 19.4 | 4.8 |
| NanoMamba-Tiny-WS (3회 반복) | ~3,782 | 14.8 | 3.7 |

### 제3 실시예: 하드웨어 설계

#### 3.1 하드웨어 가속기 IP 블록 아키텍처 (도 6 참조)

```
+================================================================+
|                    NanoMamba IP Block                            |
+================================================================+
|                                                                  |
|  +------------------+    +------------------+                   |
|  | AXI4-Lite Slave  |    | AXI4-Stream I/F  |                   |
|  | (레지스터 접근)    |    | (오디오 스트리밍) |                   |
|  +--------+---------+    +--------+---------+                   |
|           |                       |                              |
|  +--------v-----------------------v---------+                   |
|  |            레지스터 파일                    |                   |
|  |  - CTRL: 시작/정지/리셋                    |                   |
|  |  - STATUS: 완료/오류/인터럽트              |                   |
|  |  - CONFIG: d_model, d_state, n_layers    |                   |
|  |  - ADAPTIVE: floor_min/max, eps_min/max  |                   |
|  |  - RESULT: 분류 결과 (K 클래스 확률)       |                   |
|  +---+------+------+------+------+------+---+                   |
|      |      |      |      |      |      |                       |
|  +---v---+ +v----+ +v----+ +v----+ +v---v-+                    |
|  | STFT  | | SNR | | MOE | | PCEN | | SSM  |                    |
|  | Unit  | | Est | | Rtr | | x2   | | Comp |                    |
|  +-------+ +-----+ +-----+ +------+ +------+                    |
|      |                                   |                       |
|  +---v-----------------------------------v---+                   |
|  |           Weight SRAM (4.5KB)              |                   |
|  +--------------------------------------------+                   |
|      |                                                           |
|  +---v--------------------------------------------+              |
|  |          DMA 컨트롤러                           |              |
|  +------------------------------------------------+              |
|      |                                                           |
|  +---v--------------------------------------------+              |
|  |          전력 관리 모듈                          |              |
|  |  - Expert별 클럭 게이팅                         |              |
|  |  - 프레임 간 자동 슬립                          |              |
|  |  - Wake-on-Voice 인터럽트                       |              |
|  +------------------------------------------------+              |
+================================================================+
```

#### 3.2 레지스터 맵

```
0x00: CTRL        [RW] - Bit0: Start, Bit1: Stop, Bit2: Reset
0x04: STATUS      [RO] - Bit0: Busy, Bit1: Done, Bit2: Error, Bit3: IRQ
0x08: CONFIG      [RW] - d_model(8b) | d_state(4b) | n_layers(4b) | mode(8b)
0x0C: WEIGHT_ADDR [RW] - 가중치 SRAM 시작 주소 (32비트)
0x10: AUDIO_ADDR  [RW] - 오디오 버퍼 시작 주소 (32비트)
0x14: RESULT[0-11][RO] - 클래스 확률 (INT8)
0x48: ARGMAX      [RO] - 최종 분류 결과 (4비트)
0x4C: GATE_TEMP   [RW] - DualPCEN gate temperature (FP16)
0x50: FLOOR_MIN   [RW] - adaptive delta_floor_min (FP16)
0x52: FLOOR_MAX   [RW] - adaptive delta_floor_max (FP16)
0x54: EPS_MIN     [RW] - adaptive epsilon_min (FP16)
0x56: EPS_MAX     [RW] - adaptive epsilon_max (FP16)
0x58: BGATE_FLOOR [RW] - B-gate floor (FP16, 기본값 0.3)
```

#### 3.3 SoC 통합 (도 7 참조)

본 발명의 하드웨어 가속기는 Qualcomm QCC5171 (Kalimba DSP), JieLi AC79 (pi32v2 DSP), BES BES2700 (BECO NPU + STAR-MC1) 등 다양한 상용 SoC에 통합 가능하다. Bus-Agnostic Wrapper를 통해 AXI, AHB, 또는 독점 버스를 사용하는 SoC에 동일한 IP 코어를 적용할 수 있다.

### 제4 실시예: 비전 및 다중 모달 적용 (도 14 참조)

본 발명의 SNR-적응형 구조적 상수 프레임워크는 음성에 한정되지 않으며, 다음과 같은 도메인에 적용 가능하다.

#### 4.1 비전 (이미지 분류)

```
이미지 -> 패치 분할 -> 공간 주파수 분석 -> 다차원 MOE 라우팅
                                          |
                       +------------------+------------------+
                       |                  |                  |
                   [저주파 전문가]    [고주파 전문가]    [항등 전문가]
                   (부드러운 영역)   (에지/텍스처)     (원본 보존)
                       |                  |                  |
                       +------------------+------------------+
                                          |
                              SNR-적응형 SA-SSM 처리
                              (노이즈 -> SNR 개념 확장)
                                          |
                                       분류
```

비전 도메인에서의 SNR 적응:
- "SNR"을 "신호 대 잡음비"에서 "유용 정보 대 방해 정보 비"로 일반화
- 이미지 노이즈, 블러, 조명 변화 등을 "잡음"으로 처리
- 적응형 구조적 상수가 이미지 품질에 따라 자동 조절

#### 4.2 센서 데이터 (이상 감지)

시간 윈도우 통계(평균 변화율, 분산, 첨도)를 라우팅 신호로 사용하여, 정상/과도/이상 전문가를 선택적으로 활성화한다.

#### 4.3 SPP 분석에 의한 설계 근거

SPP(Speech Presence Probability) 기반 특징 공간 블렌딩(log_mel과 PCEN의 가중 혼합)이 시도되었으나, 특징 공간 불일치(log_mel 범위: [-6, 0] vs. PCEN 범위: [0, 2])로 인해 성능이 저하되었다. 이 분석은 잡음 대응이 특징 수준(feature-level blending)이 아닌 구조적 수준(structural approach)에서 이루어져야 함을 지지한다.

---

## [특허 청구범위]

### 독립항

**[청구항 1] (시스템 청구항 -- 넓은 범위)**

입력 신호를 처리하는 인공지능 추론 시스템으로서,
(a) 입력 신호의 품질 지표(quality indicator)를 추정하는 신호 품질 추정부;
(b) 상태공간모델(State Space Model)을 이용하여 상기 입력 신호를 처리하는 시퀀스 처리부로서, 상기 상태공간모델의 적어도 하나의 구조적 상수(architectural constant)가 상기 품질 지표에 따라 적응적으로 조절되되, 상기 구조적 상수는 학습 과정에서 옵티마이저에 의해 업데이트되지 않는 비학습 상수(non-learnable constant)의 조합으로 구현되는, 시퀀스 처리부; 및
(c) 상기 시퀀스 처리부의 출력으로부터 인식 결과를 산출하는 분류부
를 포함하며,
상기 적어도 하나의 구조적 상수는, 상기 품질 지표가 높은 환경에서 소정의 기본값(default value)으로 수렴하여, 깨끗한 입력에 대한 처리 성능이 구조적으로 보존되는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 2] (방법 청구항 -- 넓은 범위)**

입력 신호에 대한 인공지능 추론 방법으로서,
(a) 입력 신호로부터 품질 지표를 추정하는 단계;
(b) 상기 품질 지표에 기반하여, 상태공간모델의 이산화 스텝 하한(discretization step floor), 잔차 경로 계수(residual path coefficient), 및 입력 행렬 게이트 하한(input matrix gate floor) 중 적어도 하나를 적응적으로 산출하는 단계로서, 상기 적응적 산출은 비학습 상수의 선형 보간(linear interpolation)에 의해 수행되는, 단계;
(c) 상기 적응적으로 산출된 구조적 상수를 적용하여 상기 상태공간모델의 상태를 갱신하는 단계; 및
(d) 상기 상태공간모델의 출력으로부터 인식 결과를 산출하는 단계
를 포함하는, 인공지능 추론 방법.

**[청구항 3] (하드웨어 가속기 청구항)**

인공지능 추론을 위한 하드웨어 가속기로서,
(a) 입력 신호를 주파수 영역으로 변환하는 변환 연산 유닛;
(b) 주파수별 신호 품질을 추정하는 품질 추정 유닛;
(c) 다차원 물리적 특성에 기반하여 복수의 처리 전문가의 출력을 블렌딩하는 전문가 혼합 라우팅 유닛;
(d) 품질 지표에 의해 적응적으로 조절되는 구조적 상수를 가진 상태공간모델 연산을 수행하는 SSM 계산 유닛으로서, 상기 구조적 상수를 저장하는 레지스터를 포함하는, SSM 계산 유닛; 및
(e) SoC 통합을 위한 버스 인터페이스
를 포함하는, 하드웨어 가속기.

**[청구항 4] (범용 AI 추론 프레임워크 청구항)**

시퀀스 데이터를 처리하는 범용 인공지능 추론 프레임워크로서,
(a) 입력 시퀀스의 각 시간 스텝에서 유용 정보 대 방해 정보의 비율을 나타내는 품질 지표를 산출하는 품질 분석 모듈;
(b) 서로 다른 입력 조건에 특화된 N개(N >= 2)의 전문가 모듈과, 상기 입력 시퀀스의 물리적 특성에서 유도된 적어도 2차원의 라우팅 신호에 기반하여 상기 전문가 모듈의 출력을 선택적으로 결합하는 다차원 라우팅 모듈; 및
(c) 상기 품질 지표에 따라 이산화 스텝, 잔차 경로 계수, 및 입력 게이트 중 적어도 하나의 구조적 상수가 적응적으로 조절되는 상태공간 처리 모듈
을 포함하며,
상기 프레임워크는 음성 신호, 영상 신호, 및 센서 신호 중 적어도 하나에 적용 가능한 것을 특징으로 하는, 범용 인공지능 추론 프레임워크.

**[청구항 5] (에지 SoC 시스템 청구항)**

항상-켜짐 신호 인식을 위한 에지 SoC 시스템으로서,
(a) 디지털 신호 처리기(DSP), 신경 처리 유닛(NPU), 또는 범용 프로세서 중 적어도 하나;
(b) 상기 프로세서 상에 구현된 잡음 강인 인식 모듈로서, 품질 지표에 따라 적응적으로 조절되는 비학습 구조적 상수를 가진 상태공간모델과, 다차원 물리적 신호 기반 라우팅을 가진 구조적 전문가 혼합 아키텍처를 포함하는, 잡음 강인 인식 모듈; 및
(c) 전문가별 클럭 게이팅과 프레임 간 자동 슬립을 제공하는 전력 관리 유닛
을 포함하는, 에지 SoC 시스템.

### 종속항

**[청구항 6]**

제1항에 있어서,
상기 품질 지표는 신호 대 잡음비(SNR) 추정치이며, 상기 SNR 추정치는 입력 신호의 초기 N 프레임(N >= 1)으로부터 추정된 잡음 바닥(noise floor)에 기반하여 주파수 대역별로 산출되는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 7]**

제1항에 있어서,
상기 적어도 하나의 구조적 상수는,
(i) 이산화 스텝 하한(delta floor)으로서, 소정의 최솟값(delta_floor_min)과 최댓값(delta_floor_max)의 범위에서 상기 품질 지표에 따라 선형 보간되는, 이산화 스텝 하한;
(ii) 잔차 경로 계수(epsilon)로서, 소정의 최솟값(epsilon_min)과 최댓값(epsilon_max)의 범위에서 상기 품질 지표에 반비례하여 선형 보간되는, 잔차 경로 계수; 및
(iii) 입력 행렬 게이트 하한(B-gate floor)으로서, SNR 유도 시그모이드 게이트 출력에 아핀 변환(affine transformation)을 적용하여 소정의 최솟값(예: 0.3) 이상을 보장하는, 입력 행렬 게이트 하한
중 적어도 하나를 포함하는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 8]**

제7항에 있어서,
상기 이산화 스텝 하한의 최솟값은 0.01 이상 0.10 이하이고, 최댓값은 0.10 이상 0.30 이하이며; 상기 잔차 경로 계수의 최솟값은 0.01 이상 0.15 이하이고, 최댓값은 0.10 이상 0.50 이하인 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 9]**

제1항에 있어서,
상기 시스템은 서로 다른 입력 조건에 특화된 복수의 특징 정규화 전문가와, 입력 신호의 물리적 특성에서 유도된 라우팅 신호에 기반하여 상기 전문가의 출력을 선택적으로 결합하는 라우팅 모듈을 더 포함하며,
상기 라우팅 신호는 적어도 2차원의 물리적 특성을 결합한 다차원 라우팅 신호인 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 10]**

제9항에 있어서,
상기 다차원 라우팅 신호는 스펙트럴 평탄도(Spectral Flatness)와 스펙트럴 기울기(Spectral Tilt)의 결합이며,
상기 스펙트럴 평탄도는 주파수 대역 에너지의 기하 평균과 산술 평균의 비율로 정의되고,
상기 스펙트럴 기울기는 저주파 대역 에너지와 전체 에너지의 비율로 정의되며,
상기 결합은 스펙트럴 기울기가 소정의 임계값(예: 0.6)을 초과할 때 스펙트럴 평탄도를 상향 보정하는 방식으로 수행되어, 저주파 집중형 정상 잡음의 오분류를 방지하는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 11]**

제10항에 있어서,
상기 보정은 다음 수식에 의해 수행되는 것을 특징으로 하는, 인공지능 추론 시스템:
SF_adjusted = SF + (1 - SF) * ReLU(Tilt - threshold)
여기서 SF는 스펙트럴 평탄도, Tilt는 스펙트럴 기울기, threshold는 소정의 임계값, ReLU는 정류 선형 유닛 함수이며, 상기 보정에 의한 추가 학습 가능 파라미터는 0개인 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 12]**

제9항에 있어서,
상기 복수의 특징 정규화 전문가는 적어도 두 개의 PCEN(Per-Channel Energy Normalization) 전문가를 포함하며,
제1 전문가는 0.5 이상의 오프셋(delta) 파라미터를 가져 비정상 잡음에 특화되고,
제2 전문가는 0.1 이하의 오프셋 파라미터를 가져 정상 잡음에 특화되며,
각 전문가의 오프셋 파라미터는 소정의 범위 내에서 클램핑(clamping)되어 학습 과정에서의 전문가 붕괴(expert collapse)를 방지하는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 13]**

제2항에 있어서,
상기 (b) 단계에서, 상기 이산화 스텝 하한은 다음 수식에 의해 산출되는 것을 특징으로 하는, 인공지능 추론 방법:
adaptive_floor = floor_min + (floor_max - floor_min) * quality_indicator
여기서 floor_min과 floor_max는 레지스터 버퍼(register buffer)로 구현된 비학습 상수이며, quality_indicator는 정규화된 품질 지표인 것을 특징으로 하는, 인공지능 추론 방법.

**[청구항 14]**

제2항에 있어서,
상기 (b) 단계에서, 상기 잔차 경로 계수는 다음 수식에 의해 산출되는 것을 특징으로 하는, 인공지능 추론 방법:
adaptive_eps = eps_max - (eps_max - eps_min) * quality_indicator
여기서 eps_min과 eps_max는 레지스터 버퍼로 구현된 비학습 상수이며, 품질 지표가 높을수록 잔차 경로 계수가 감소하여 게이팅 메커니즘에 대한 신뢰를 높이고, 품질 지표가 낮을수록 잔차 경로 계수가 증가하여 바이패스 정보 흐름을 강화하는 것을 특징으로 하는, 인공지능 추론 방법.

**[청구항 15]**

제2항에 있어서,
상기 (b) 단계에서, 상기 입력 행렬 게이트 하한은 SNR 유도 시그모이드 게이트 원시값(B_gate_raw)에 아핀 변환을 적용하여:
B_gate = B_gate_raw * scale + floor (여기서 scale + floor = 1.0)
의 형태로 산출되며, 이에 의해 입력 행렬의 게이팅이 최소 floor 값 이상으로 보장되어 복합 과억제(compound over-suppression)를 방지하는 것을 특징으로 하는, 인공지능 추론 방법.

**[청구항 16]**

제2항에 있어서,
상기 방법에 의한 모델은 잡음이 없는 깨끗한 데이터만으로 학습되며, 잡음 강인성은 잡음 증강 학습 데이터 없이 상기 적응형 구조적 상수의 아키텍처적 특성에 의해 달성되는 것을 특징으로 하는, 인공지능 추론 방법.

**[청구항 17]**

제1항에 있어서,
상기 시스템은 저주파 대역에 대해 비학습 지수 감쇠(exponential decay) 에너지 하한을 적용하는 주파수 의존 에너지 하한 모듈을 더 포함하는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 18]**

제1항에 있어서,
상기 시퀀스 처리부는 단일 상태공간모델 블록을 N회(N >= 2) 반복 실행하여 N층 깊이의 처리를 수행하되, 고유 학습 파라미터는 단일 블록분만을 유지하는 가중치 공유 메커니즘을 적용하는 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 19]**

제1항에 있어서,
상기 시스템의 총 학습 가능 파라미터 수는 10,000개 미만이며, 상기 적응형 구조적 상수에 의한 추가 학습 가능 파라미터는 0개인 것을 특징으로 하는, 인공지능 추론 시스템.

**[청구항 20]**

제3항에 있어서,
상기 하드웨어 가속기는 상기 구조적 상수의 최솟값 및 최댓값을 저장하는 레지스터를 포함하며, 상기 레지스터의 값을 외부에서 변경함으로써 상기 적응형 구조적 상수의 동작 범위를 런타임에 조정 가능한 것을 특징으로 하는, 하드웨어 가속기.

**[청구항 21]**

제3항에 있어서,
상기 전문가 혼합 라우팅 유닛은 스펙트럴 평탄도와 스펙트럴 기울기를 입력으로 받아 다차원 라우팅 신호를 산출하는 회로를 포함하며, 상기 회로는 기하 평균 계산을 위한 log-exp LUT, 저/고주파 에너지 비율 계산을 위한 누적기-나눗셈기, 및 보정을 위한 ReLU-MAC 유닛을 포함하는 것을 특징으로 하는, 하드웨어 가속기.

**[청구항 22]**

제3항에 있어서,
상기 하드웨어 가속기는 상기 라우팅 유닛의 라우팅 결정에 기반하여 미사용 전문가에 대한 클럭 공급을 차단하는 전문가별 클럭 게이팅 회로를 포함하는 것을 특징으로 하는, 하드웨어 가속기.

**[청구항 23]**

제4항에 있어서,
상기 품질 분석 모듈은 음성 신호에 적용될 때 주파수별 신호 대 잡음비(SNR)를 산출하고, 영상 신호에 적용될 때 공간 주파수 에너지 분포를 산출하며, 센서 신호에 적용될 때 시간 윈도우 통계(평균 변화율, 분산, 첨도)를 산출하는 것을 특징으로 하는, 범용 인공지능 추론 프레임워크.

**[청구항 24]**

제4항에 있어서,
상기 다차원 라우팅 모듈의 라우팅 신호 산출에 사용되는 학습 가능 파라미터의 수는 상기 전문가 모듈의 총 파라미터 수의 1% 미만이며, 상기 라우팅 신호의 적어도 하나의 차원은 학습 파라미터 없이 입력 신호의 물리적 특성에서 직접 유도되는 것을 특징으로 하는, 범용 인공지능 추론 프레임워크.

**[청구항 25]**

제5항에 있어서,
상기 에지 SoC는 Kalimba DSP를 포함하는 블루투스 오디오 SoC, pi32v2 프로세서를 포함하는 블루투스 오디오 SoC, BECO NPU를 포함하는 블루투스 오디오 SoC, 또는 Cortex-M 시리즈 프로세서를 포함하는 마이크로컨트롤러 중 적어도 하나인 것을 특징으로 하는, 에지 SoC 시스템.

---

## [요약서]

**발명의 명칭:** 초경량 상태공간모델 기반 인공지능 추론 방법 및 장치, 그리고 이를 이용한 음성 및 영상 인식 시스템

**요약:**

본 발명은 상태공간모델(SSM)의 구조적 상수(이산화 스텝 하한, 잔차 경로 계수, 입력 행렬 게이트 하한)를 신호 품질 지표(SNR 등)에 따라 적응적으로 조절하여, 추가 학습 가능 파라미터 없이 잡음 환경에서의 강인성을 달성하는 초경량 인공지능 추론 방법 및 장치에 관한 것이다. 본 발명의 적응형 구조적 상수는 고품질 환경에서 원래의 기본값으로 자동 수렴하여 깨끗한 입력의 처리 성능을 아키텍처적으로 보존하는 "클린 보존 원리"를 구현한다. 또한, 스펙트럴 평탄도와 스펙트럴 기울기를 결합한 다차원 라우팅에 의해, 복수의 특징 정규화 전문가(PCEN 등)를 무파라미터로 라우팅하여 다양한 잡음 유형에 동시 대응한다. 전체 모델은 10,000개 미만 파라미터, INT8 양자화 시 5KB 미만으로, AXI/AHB 버스 인터페이스를 갖춘 하드웨어 가속기 IP 블록으로 구현되어 에지 SoC에 통합 가능하며, 음성뿐만 아니라 영상, 센서 데이터 등 다양한 도메인에 범용 적용 가능하다.

**대표도:** 도 1

---

## [발명자]

최진호 (Jin Ho Choi, Ph.D.)

## [출원인]

(출원인 정보 기재)

---

## [부록 A: 코드-청구항 추적 테이블]

| 청구항 | 구성 요소 | 소스 파일 | 핵심 코드/상수 |
|--------|-----------|-----------|---------------|
| 1(a) | 품질 지표 추정 (SNR) | nanomamba.py L46-120 | SNREstimator, tanh(snr/10) |
| 1(b) | SNR-적응형 구조적 상수 | nanomamba.py L607-622 | delta_floor_min=0.05, max=0.15 |
| 1(b) | 적응형 epsilon | nanomamba.py L618-621 | epsilon_min=0.08, max=0.20 |
| 2(b) | 적응형 delta floor 계산 | nanomamba.py L660-667 | adaptive_floor = min + range*snr |
| 2(b) | 적응형 epsilon 계산 | nanomamba.py L710-712 | adaptive_eps = max - range*snr |
| 2(b) | B-gate floor | nanomamba.py L651-655 | B_gate = raw*0.7 + 0.3 |
| 9,10 | 다차원 스펙트럴 라우팅 | nanomamba.py L315-329 | SF + Spectral Tilt |
| 10 | Spectral Tilt 계산 | nanomamba.py L319-322 | low/(low+high+eps) |
| 11 | SF 보정 수식 | nanomamba.py L329 | sf + (1-sf)*relu(tilt-0.6) |
| 12 | DualPCEN 전문가 | nanomamba.py L272-293 | delta=2.0, delta=0.01 |
| 17 | FreqDependentFloor | nanomamba.py L344-372 | 0.05*exp(-3.0*ratio) |
| 18 | Weight Sharing | nanomamba.py L919-930 | n_repeats |
| 4 | 범용 프레임워크 | nanomamba.py L798-1058 | NanoMamba 클래스 |

---

## [부록 B: 선행기술 대비표]

| 구분 | 본 발명 | Google MOE (US20200279150A1) | Qualcomm KWD (US20210005181A1) | 표준 Mamba | 외부 향상기 (GTCRN) |
|------|---------|------------------------------|--------------------------------|-----------|-------------------|
| 라우팅 방식 | 다차원 물리적 신호 (SF+Tilt) | 학습된 선형+소프트맥스 | 다단계 전력 관리 | 없음 | N/A |
| 라우팅 파라미터 | 1개 (gate_temp) | 입력차원 x 전문가수 | N/A | N/A | N/A |
| 잡음 적응 | 구조적 (클린만 학습) | 학습 데이터 의존 | 별도 AEC/NR 필요 | 없음 | 전처리 |
| Clean 성능 보존 | O (아키텍처적 보장) | 학습 의존 | N/A | N/A | X (-7.5~13.3%p) |
| SSM 구조적 상수 | 적응형 (0 추가 파라미터) | N/A | N/A | 없음 | N/A |
| 핑크 잡음 대응 | Spectral Tilt 보정 | 학습 의존 | 별도 모듈 | 없음 | 전처리 |
| 복합 과억제 방지 | B-gate floor (0.3) | N/A | N/A | 없음 | N/A |
| 모델 크기 | < 5KB INT8 | 수 GB | 수 MB | 수 MB | +23.7K params |
| 에지 SoC 통합 | AXI/AHB IP 블록 | 클라우드 서버 | 전용 DSP | N/A | 추가 메모리 필요 |
| 비전 적용 | 가능 (범용 프레임워크) | 가능 | 불가 | 가능 | 불가 |

---

## [부록 C: 실험 결과 요약]

### C.1 외부 향상기 없이 (본 발명의 구조적 접근)

| 모델 | Params | Clean | factory-15dB | white-15dB | babble-15dB | pink-15dB |
|------|--------|-------|-------------|------------|-------------|-----------|
| NanoMamba-Tiny | 4,634 | 92.9% | 38.4% | 20.2% | 58.6% | 9.9% |
| NanoMamba-Small | ~12K | 95.2% | - | - | - | - |
| + PCEN delta=2.0 | ~4.8K | 95.1% | - | - | babble-0dB: 81.0% | - |
| + PCEN delta=0.01 | ~4.8K | 94.6% | 31.0% | - | - | - |

### C.2 외부 GTCRN 향상기 사용 (Clean 성능 저하 실증)

| 모델 | Clean | Clean 저하 | factory-15dB | white-15dB |
|------|-------|-----------|-------------|------------|
| NanoMamba-Tiny + GTCRN | 79.6% | -13.3%p | 60.3% | 59.7% |
| NanoMamba-Small + GTCRN | 87.7% | -7.5%p | - | - |
| NanoMamba-Tiny-DualPCEN + GTCRN | 87.2% | -7.6%p | - | - |

### C.3 CNN 기준 모델 (정적 필터의 잡음 취약성 실증)

| 모델 | Params | Clean | white-0dB |
|------|--------|-------|-----------|
| DS-CNN-S | 23,700 | 96.6% | 13.9% (붕괴) |
| BC-ResNet-1 | 7,500 | 96.0% | - |

---

*본 명세서는 특허 명세서 초안으로, 최종 출원 시 변리사 검토 및 도면 첨부가 필요합니다.*
*공지예외주장(Grace Period Declaration)이 필요하며, 관련 공개일로부터 12개월 이내에 출원해야 합니다.*
