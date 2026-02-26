# 특허 명세서

## [발명의 명칭]

**구조적 전문가 혼합 네트워크를 이용한 잡음 강인 음성 인식 시스템 및 방법과 이를 위한 하드웨어 가속기**

Noise-Robust Speech Recognition System and Method Using Structural Mixture-of-Experts Network, and Hardware Accelerator Therefor

---

## [기술분야]

본 발명은 인공지능 기반 음성 인식 기술에 관한 것으로, 보다 구체적으로는 구조적 전문가 혼합(Mixture of Experts, MOE) 네트워크와 신호 대 잡음비(Signal-to-Noise Ratio, SNR) 변조 상태공간모델(State Space Model, SSM)을 결합하여, 극도로 적은 파라미터(5,000개 미만, INT8 양자화 시 5KB 미만)로 다양한 잡음 환경에서 강인한 키워드 인식(Keyword Spotting, KWS)을 수행하는 시스템 및 방법에 관한 것이다.

또한, 본 발명은 상기 시스템을 블루투스(BT) 오디오 SoC, 모바일 AP, IoT 디바이스 등 에지(edge) 프로세서에서 실시간으로 구동하기 위한 전용 하드웨어 가속기 IP 블록의 설계에 관한 것이다.

**국제특허분류(IPC):**
- G10L 15/22 — 잡음 환경에서의 음성 인식
- G06N 3/04 — 신경망 아키텍처
- G06N 3/08 — 신경망 학습 방법
- G06F 17/10 — 상태공간 모델을 이용한 신호 처리
- H03H 17/02 — 디지털 필터 (적응형 정규화)

---

## [배경기술]

### 1. 종래 기술의 문제점

키워드 인식(KWS)은 음성 비서, 스마트 이어버드, IoT 디바이스 등에서 항상-켜짐(always-on) 모드로 동작해야 하므로, 극도로 낮은 연산량과 메모리 사용이 요구된다. 이를 위해 다양한 경량 모델이 제안되었으나, 기존 기술은 다음과 같은 근본적 한계를 가진다.

#### 1.1 합성곱 신경망(CNN) 기반 접근의 한계

BC-ResNet(Broadcasting Residual Network, Kim et al., 2021)과 DS-CNN(Depthwise Separable CNN, Zhang et al., 2017) 등 종래 CNN 기반 KWS 모델은 학습 시 고정된 필터 응답(filter response)을 가진다. 즉, 컨볼루션 커널의 가중치가 학습 완료 후 동결(frozen)되어, 추론 시 입력 신호의 잡음 특성에 따른 적응적 변환이 불가능하다.

이로 인해 학습 시 접하지 못한 잡음 유형에 대해 성능이 급격히 저하된다. 예를 들어, 23,700개 파라미터를 가진 DS-CNN-S 모델은 깨끗한 환경에서 96.4%의 정확도를 보이나, 0dB 백색잡음(white noise) 환경에서는 13.9%로 급락하여 사실상 사용이 불가능하다. 이는 CNN의 정적 필터 응답이 시변(time-varying) 잡음에 적응하지 못하기 때문이다.

#### 1.2 기존 전문가 혼합(MOE) 기술의 한계

전문가 혼합(Mixture of Experts) 기술은 복수의 전문가 네트워크와 게이팅(gating) 네트워크로 구성되며, 입력에 따라 적합한 전문가를 선택적으로 활성화한다.

US20200279150A1(Google LLC)에 개시된 MOE 기술에서는 선형 변환(linear layer)과 소프트맥스(softmax) 함수를 이용한 학습 기반 게이팅을 사용한다. 이러한 학습 기반 게이팅은 추가적인 학습 가능 파라미터를 요구하고, 학습 데이터 분포에 편향되며, 학습 시 접하지 못한 잡음 유형에 대한 일반화 성능이 제한적이다.

또한, 종래 MOE 기술은 대규모 언어 모델(LLM)이나 비전 트랜스포머(Vision Transformer) 등 수백만~수십억 파라미터 규모의 모델에 주로 적용되어 왔으며, 5,000개 미만의 극소 파라미터 모델에서의 MOE 적용은 시도된 바 없다.

#### 1.3 상태공간모델(SSM)의 잡음 취약성

최근 Mamba(Gu & Dao, 2023) 등 선택적 상태공간모델(Selective SSM)이 시퀀스 모델링에서 주목받고 있다. Mamba의 핵심 혁신은 선택 메커니즘(selection mechanism)으로, 입력에 따라 이산화 스텝(dt), 입력 행렬(B), 출력 행렬(C)을 적응적으로 조절한다.

그러나 표준 Mamba에서 이산화 스텝 dt와 입력 행렬 B는 오직 시간 축 특징(temporal features)에서만 투영(projection)되며, 입력 신호의 주파수별 SNR 정보를 활용하지 않는다. 이는 잡음이 존재할 때 SSM이 잡음 프레임과 음성 프레임을 동등하게 처리하여, 잡음 환경에서의 성능 저하를 초래한다.

#### 1.4 다단계 파이프라인의 비효율성

Qualcomm Sensing Hub 등 종래 상용 시스템에서는 음성 활동 검출(VAD) → 음향 반향 제거(AEC) → 잡음 억제(NR) → 키워드 검출(KWD)의 다단계 직렬 파이프라인을 사용한다(US20210005181A1). 이러한 다단계 접근은 각 모듈의 지연(latency)이 누적되고, 모듈 간 정보 손실이 발생하며, 전체 시스템의 전력 소모와 메모리 사용량이 증가하는 문제가 있다.

#### 1.5 단일 PCEN(Per-Channel Energy Normalization)의 한계

PCEN(Wang et al., ICASSP 2017)은 멜(mel) 스펙트로그램에 대한 적응적 이득 제어(AGC)와 동적 범위 압축(DRC)을 수행하는 정규화 기법이다. PCEN의 출력은 다음과 같이 정의된다:

```
smoother[t] = (1 - s) * smoother[t-1] + s * mel[t]
gain = (epsilon + smoother)^(-alpha)
output = (mel * gain + delta)^r - delta^r
```

여기서 delta(오프셋) 파라미터는 정규화 특성을 결정하는 핵심 변수이다:
- delta가 크면(예: 2.0) AGC 이득이 무시되어 오프셋 우세 모드(offset-dominant mode)가 되며, 비정상 잡음(babble 등)에 강인하다.
- delta가 작으면(예: 0.01) AGC가 지배적(AGC-dominant mode)이 되어, 정상 잡음(factory, white 등)의 시간 불변 특성을 효과적으로 추적 제거한다.

단일 PCEN은 하나의 delta 값만을 가지므로, 정상 잡음과 비정상 잡음을 동시에 최적으로 처리할 수 없다. 이는 실제 환경에서 정상 잡음(공장, 에어컨 등)과 비정상 잡음(대화, 군중 소음 등)이 혼재하는 상황에 대한 근본적 한계이다.

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

---

## [발명의 내용]

### 해결하고자 하는 과제

본 발명은 상기 종래 기술의 문제점을 해결하기 위한 것으로, 다음과 같은 기술적 과제를 해결하고자 한다.

**첫째,** 잡음 증강 학습 데이터 없이(clean data only training) 구조적으로 잡음에 강인한 음성 인식 모델을 제공한다.

**둘째,** 정상 잡음(factory, white, pink)과 비정상 잡음(babble, street)을 단일 모델로 동시에 처리할 수 있는 구조적 전문가 혼합(Structural MOE) 아키텍처를 제공한다.

**셋째,** 5,000개 미만 파라미터, INT8 양자화 시 5KB 미만의 초경량 모델로 상기 잡음 강인성을 달성하는 시스템을 제공한다.

**넷째,** 상기 시스템을 Qualcomm, JieLi(杰理), BES(恒玄) 등 상용 SoC에 통합 가능한 하드웨어 가속기 IP 블록으로 구현하는 방법을 제공한다.

**다섯째,** 음성(audio)뿐만 아니라 비전(vision) 등 다양한 시퀀스 처리 태스크에 범용적으로 적용 가능한 구조적 MOE 프레임워크를 제공한다.

### 과제의 해결 수단

상기 과제를 해결하기 위해, 본 발명은 다음의 핵심 구성 요소를 포함하는 잡음 강인 음성 인식 시스템을 제공한다.

#### A. 이중 PCEN 전문가 혼합(DualPCEN MOE) — 2-Expert 구조

본 발명의 제1 구성 요소는 서로 다른 잡음 유형에 특화된 두 개의 PCEN 전문가와, 입력 신호의 음향 물리적 특성에 기반한 무파라미터 라우팅(zero-parameter routing)을 결합한 이중 전문가 혼합 모듈이다.

**전문가 1(비정상 잡음 전문가):** delta_init = 2.0, s_init = 0.025, alpha_init = 0.99, r_init = 0.5
- 높은 delta 값에 의해 AGC 이득이 상대적으로 무시되어 오프셋 우세 모드가 됨
- babble, street 등 시변 잡음에 대해 구조적으로 강인
- delta 클램핑: (0.5, 5.0) — 학습 중 delta가 낮아지는 것을 방지

**전문가 2(정상 잡음 전문가):** delta_init = 0.01, s_init = 0.15, alpha_init = 0.99, r_init = 0.1
- 낮은 delta 값에 의해 AGC가 지배적이 되어, 시간 불변 잡음을 추적 제거
- factory, white, pink 등 정상 잡음에 대해 구조적으로 강인
- delta 클램핑: (0.001, 0.1) — 학습 중 delta가 높아지는 것을 방지

**라우팅 신호 — 스펙트럴 평탄도(Spectral Flatness):**

```
SF(t) = exp(mean(log(mel(t)))) / mean(mel(t))
```

스펙트럴 평탄도는 기하 평균(geometric mean)과 산술 평균(arithmetic mean)의 비율로 정의되며, 0과 1 사이의 값을 가진다:
- SF → 1.0: 편평한 스펙트럼 → 정상 잡음(백색잡음 등) → 전문가 2(AGC 전문가) 활성화
- SF → 0.0: 피크가 뚜렷한 스펙트럼 → 비정상 잡음/음성 → 전문가 1(오프셋 전문가) 활성화

이 라우팅 신호는 학습 가능 파라미터를 전혀 사용하지 않으며(0 learnable parameters), 입력 신호의 음향 물리적 특성에서 직접 유도된다. 이는 종래 Google MOE 특허(US20200279150A1)의 선형-소프트맥스 게이팅과 근본적으로 다른 접근이다.

**게이트 함수:**
```
gate(t) = sigmoid(gate_temp * (SF(t) - 0.5))
output(t) = gate(t) * expert_stationary(t) + (1 - gate(t)) * expert_nonstationary(t)
```

여기서 gate_temp는 유일한 학습 가능 파라미터(초기값 5.0)로, 라우팅의 날카로움(sharpness)을 제어한다.

**추가 파라미터:** 전문가 1(160) + 전문가 2(160) + gate_temp(1) = 총 321개

#### B. 스펙트럴-인식 상태공간모델(Spectral-Aware SSM, SA-SSM)

본 발명의 제2 구성 요소는 주파수별 SNR 추정치를 SSM의 선택 메커니즘에 직접 주입하는 스펙트럴-인식 상태공간모델이다.

**표준 Mamba SSM:**
```
dt = softplus(W_dt * x + b_dt)         # 이산화 스텝
B = W_B * x                            # 입력 행렬
C = W_C * x                            # 출력 행렬
```

**본 발명의 SA-SSM:**
```
dt_snr_shift = W_snr_dt * snr_mel      # SNR → dt 변조
B_gate = sigmoid(W_snr_B * snr_mel)    # SNR → B 게이팅

dt = softplus(W_dt * x + dt_snr_shift + b_dt) + delta_floor   # SNR 변조 + 구조적 하한
B_eff = B * (1 - alpha + alpha * B_gate)                      # SNR 게이팅된 입력 행렬
```

**핵심 혁신 — 구조적 잡음 강인성 보장:**

1. **delta_floor(이산화 스텝 하한):** register_buffer로 구현된 비학습 상수(값: 0.15). delta_floor > 0을 보장함으로써, 극단적 저 SNR 환경에서도 SSM의 이산화 스텝이 0이 되지 않아 정보 전파가 완전히 차단되는 것을 방지한다. 이는 학습 가능 파라미터가 아닌 아키텍처 상수(register_buffer)로 구현되어, 옵티마이저가 학습 과정에서 이 값을 파괴할 수 없다.

2. **epsilon(잔차 경로 계수):** register_buffer로 구현된 비학습 상수(값: 0.1). SSM 상태 업데이트 시 SNR 의존적 게이팅을 완전히 우회(bypass)하는 잔차 경로를 제공한다:
```
h[t] = A_bar * h[t-1] + dB * x[t] + epsilon * x[t]
```
이 잔차 경로는 모든 SNR 게이팅과 무관하게 항상 동작하여, 잡음 환경에서의 최소 정보 흐름을 구조적으로 보장한다.

**SNR 변조의 직관:**
- 고 SNR 프레임(음성 우세) → 큰 dt → 정보를 적극 전파
- 저 SNR 프레임(잡음 우세) → 작은 dt → 잡음 억제
- delta_floor에 의해 dt ≥ 0.15 보장 → 완전 차단 방지
- epsilon에 의해 무조건적 정보 흐름 보장 → 극단적 잡음에서도 동작

#### C. 주파수 영역 전문가 혼합(MoEFreq) — 3-Expert 구조

본 발명의 제3 구성 요소는 SNR 통계 조건부 주파수 영역 전문가 혼합 모듈이다.

**전문가 구성:**
- 전문가 1: 좁은 대역 컨볼루션(kernel_size=3) — 톤형 잡음(공장 하모닉스 등) 대응
- 전문가 2: 넓은 대역 컨볼루션(kernel_size=7) — 광대역 잡음(백색잡음, HVAC 등) 대응
- 전문가 3: 항등 변환(identity, 0 파라미터) — 깨끗한 환경에서 원본 보존

**라우터:**
```
snr_stats = [mean(snr_profile), std(snr_profile)]   # SNR 핑거프린트
gate = softmax(Linear(2, 3)(snr_stats))              # 3-전문가 선택 가중치
output = sum(gate[i] * expert[i](input) for i in range(3))
```

초기화: 게이트 바이어스 = [0, 0, 1]로 항등 전문가 선호 → 깨끗한 입력 보존.
총 파라미터: 21개 (전문가 1: 4개, 전문가 2: 8개, 라우터: 9개)

#### D. 주파수 의존 에너지 하한(Frequency-Dependent Floor)

본 발명의 제4 구성 요소는 저주파 멜 밴드의 정보 손실을 구조적으로 방지하는 비학습 에너지 하한 모듈이다.

```
floor[i] = 0.05 * exp(-3.0 * (1.0 - i/(n_mels - 1)))
mel_protected = max(mel_linear, floor)
```

이 하한은 register_buffer로 구현되어 학습 불가능하며, 0번 밴드(최저 주파수)에서 가장 높고 39번 밴드(최고 주파수)에서 0에 수렴하는 지수 감쇠 프로파일을 가진다. 공장 잡음, 핑크 잡음 등 저주파 집중형 잡음에 의해 저주파 멜 밴드의 에너지가 완전히 마스킹되는 것을 방지한다.

#### E. 가중치 공유(Weight Sharing)

본 발명의 제5 구성 요소는 단일 SA-SSM 블록을 N회 반복 실행하여, N층 깊이를 가지면서 단일 블록의 파라미터만을 사용하는 가중치 공유 메커니즘이다.

예: n_repeats=3일 때, 3층 깊이의 모델이 1블록의 고유 파라미터만으로 동작. 이를 통해 3,782개 파라미터(BC-ResNet-1의 절반)로 3층 깊이 모델 구현 가능.

#### F. 일반화된 N-Expert 프레임워크

상기 구성 요소 A, C를 일반화하면, 임의의 N개 전문가와 임의의 신호 유도 라우팅 신호를 결합하는 범용 프레임워크를 구성할 수 있다:

```
routing_signal = PhysicalFeatureExtractor(input)   # 스펙트럴 평탄도, SNR 통계, 변조 스펙트럼 등
gate_weights = GatingFunction(routing_signal)       # sigmoid, softmax, top-k 등
output = sum(gate_weights[i] * Expert[i](input) for i in range(N))
```

이 프레임워크는 음성(PCEN 전문가, 주파수 필터 전문가)뿐만 아니라 비전(공간 필터 전문가, 주파수 영역 전문가) 등 다양한 도메인에 적용 가능하다.

### 발명의 효과

본 발명에 따르면 다음과 같은 효과를 얻을 수 있다.

1. **구조적 잡음 강인성:** 클린 데이터만으로 학습하여, 학습 시 접하지 않은 잡음 유형(공장, 백색, babble, 거리, 핑크)에 대해 0dB SNR에서 84.5% 정확도 유지 (종래 BC-ResNet-1: 69.4%, DS-CNN-S: 55.2%)

2. **초경량:** 4,634개 파라미터, INT8 양자화 시 4.5KB. BC-ResNet-1(7,464개) 대비 38% 적은 파라미터로 15.1%p 높은 잡음 평균 정확도 달성

3. **극단적 잡음 성능:** 0dB 백색잡음에서 80.1% (DS-CNN-S 13.9% 대비 +66.2%p)

4. **다중 잡음 동시 대응:** DualPCEN에 의해 정상 잡음(AGC 전문가)과 비정상 잡음(오프셋 전문가)을 입력 특성에 따라 자동 선택

5. **하드웨어 친화적:** 4.5KB 가중치 SRAM + 선형 시간 추론(O(L)) + 간단한 MAC 연산 → BT 오디오 SoC의 제한된 자원에서 실시간 구동 가능

6. **범용성:** 음성 키워드 인식 외 음성 향상(Speech Enhancement), 음향 이벤트 검출(Sound Event Detection), 시각적 객체 인식(Visual Object Recognition) 등에 적용 가능한 범용 구조적 MOE 프레임워크

---

## [도면의 간단한 설명]

**도 1**은 본 발명에 따른 잡음 강인 음성 인식 시스템의 전체 아키텍처를 나타내는 블록도이다.

**도 2**는 본 발명의 이중 PCEN 전문가 혼합(DualPCEN MOE) 모듈의 구조를 나타내는 상세 블록도이다.

**도 3**은 본 발명의 스펙트럴-인식 상태공간모델(SA-SSM) 블록의 내부 구조를 나타내는 블록도이다.

**도 4**는 본 발명의 주파수 영역 전문가 혼합(MoEFreq) 모듈의 구조를 나타내는 블록도이다.

**도 5**는 본 발명을 일반화한 N-Expert 전문가 혼합 프레임워크의 구조를 나타내는 블록도이다.

**도 6**은 본 발명에 따른 하드웨어 가속기 IP 블록의 RTL 수준 아키텍처를 나타내는 블록도이다.

**도 7**은 본 발명의 하드웨어 가속기가 상용 SoC(Qualcomm, JieLi, BES)에 통합되는 구조를 나타내는 블록도이다.

**도 8**은 가중치 공유 메커니즘과 메모리 레이아웃을 나타내는 도면이다.

**도 9**는 본 발명에 따른 학습 파이프라인의 흐름도이다.

**도 10**은 에지 SoC에서의 스트리밍 실시간 추론 파이프라인을 나타내는 흐름도이다.

---

## [발명을 실시하기 위한 구체적인 내용]

이하, 첨부된 도면을 참조하여 본 발명의 실시예를 상세히 설명한다.

### 제1 실시예: 구조적 MOE 네트워크 설계

#### 1.1 전체 시스템 아키텍처 (도 1 참조)

본 발명에 따른 잡음 강인 음성 인식 시스템은 다음의 처리 파이프라인으로 구성된다:

```
원시 오디오(16kHz, 1초)
    ↓
[STFT 모듈] — FFT 크기=512, 홉 길이=160, Hann 윈도우
    ↓
크기 스펙트로그램 (B, 257, T)
    ↓
[SNR 추정 모듈] — 초기 5프레임 잡음 바닥 추정 + EMA 추적
    ↓                          ↓
SNR 추정치 (B, 40, T)     크기 스펙트로그램
    ↓                          ↓
    ↓               [MoEFreq 모듈] (선택적)
    ↓                          ↓
    ↓               [멜 필터뱅크 투영] — 40 밴드
    ↓                          ↓
    ↓               [FrequencyDependentFloor]
    ↓                          ↓
    ↓               [DualPCEN MOE 모듈]
    ↓                          ↓
    ↓               [인스턴스 정규화]
    ↓                          ↓
    ↓               [패치 투영] — n_mels → d_model
    ↓                          ↓
    +→→→→→→→→→→→→→[SA-SSM 블록 × N] ←← SNR 측면 입력
                               ↓
                    [층 정규화 + 전역 평균 풀링]
                               ↓
                    [선형 분류기] — 12 클래스
                               ↓
                    키워드 인식 결과
```

#### 1.2 SNR 추정 모듈

크기 스펙트로그램에서 주파수별 SNR을 추정한다. 초기 N 프레임(기본값 5)의 평균을 잡음 바닥(noise floor)으로 사용하며, 선택적으로 비대칭 지수 이동 평균(EMA)을 통해 적응적 잡음 추적을 수행한다:

```
noise_floor_init = mean(mag[:, :, :5], dim=time)              # 초기 잡음 바닥
snr_linear = mag / (noise_scale * noise_floor + floor_param)  # 주파수별 SNR
snr_db = 10 * log10(snr_linear + 1e-8)                        # dB 변환
snr_mel = mel_fb @ snr_db                                     # 멜 스케일 투영
```

비대칭 EMA(선택적):
- 프레임 에너지 > 잡음 바닥: 느린 상승(gamma ≈ 0.05) — 음성/임팩트에 의한 오추정 방지
- 프레임 에너지 < 잡음 바닥: 빠른 하강(beta ≈ 0.10) — 잡음 바닥 신속 갱신

파라미터: noise_scale(1개), floor_param(1개), 선택적 EMA 파라미터(2개). 총 약 520개.

#### 1.3 DualPCEN MOE 모듈 (도 2 참조)

**1.3.1 PCEN 전문가 단일 모듈**

각 PCEN 전문가는 40개 멜 밴드에 대해 4개의 학습 가능 파라미터(s, alpha, delta, r)를 가진다:

```python
log_s = Parameter(log(s_init) * ones(n_mels))       # 평활 계수 (IIR 시정수)
log_alpha = Parameter(log(alpha_init) * ones(n_mels)) # AGC 이득 지수
log_delta = Parameter(log(delta_init) * ones(n_mels)) # 오프셋 (핵심 차별화 변수)
log_r = Parameter(log(r_init) * ones(n_mels))         # 압축 지수
```

로그 공간에서 학습하여 양수 보장. delta는 추가로 클램핑:
```python
delta = clamp(exp(log_delta), delta_min, delta_max)
```

IIR 평활기 (1차 IIR 필터):
```python
smoother[0] = mel[:, :, 0]
for t in range(1, T):
    smoother[t] = (1 - s) * smoother[t-1] + s * mel[:, :, t]
```

PCEN 변환:
```python
gain = (epsilon + smoother) ** (-alpha)
output = (mel * gain + delta) ** r - delta ** r
```

파라미터 수: 4 × 40 = 160개 (전문가당)

**1.3.2 이중 전문가 구성**

| 파라미터 | 전문가 1 (비정상 잡음) | 전문가 2 (정상 잡음) |
|----------|------------------------|----------------------|
| s (평활 계수) | 0.025 (느린 추적) | 0.15 (빠른 추적) |
| alpha (AGC 지수) | 0.99 | 0.99 |
| delta (오프셋) | 2.0 (높음, AGC 무시) | 0.01 (낮음, AGC 지배) |
| r (압축 지수) | 0.5 | 0.1 |
| delta 클램프 | (0.5, 5.0) | (0.001, 0.1) |

**1.3.3 스펙트럴 평탄도 라우팅**

스펙트럴 평탄도(SF)의 물리적 의미:
- SF = 1: 모든 주파수 에너지가 균일 → 백색잡음(정상 잡음)
- SF = 0: 특정 주파수에 에너지 집중 → 음성 또는 비정상 잡음

계산:
```python
log_mel = log(mel_linear + 1e-8)                    # (B, 40, T)
geo_mean = exp(mean(log_mel, dim=mel_axis))          # (B, 1, T)
arith_mean = mean(mel_linear, dim=mel_axis) + 1e-8   # (B, 1, T)
SF = clamp(geo_mean / arith_mean, 0, 1)              # (B, 1, T)
```

게이트:
```python
gate = sigmoid(gate_temp * (SF - 0.5))               # (B, 1, T)
output = gate * expert_2_out + (1 - gate) * expert_1_out
```

**1.3.4 DualPCEN의 핵심 차별점**

| 구분 | Google MoE (US20200279150A1) | 본 발명 (DualPCEN) |
|------|------------------------------|---------------------|
| 라우팅 신호 | 학습된 선형 변환 + 소프트맥스 | 스펙트럴 평탄도 (물리적 신호, 학습 불필요) |
| 라우팅 파라미터 | 입력 차원 × 전문가 수 | 1개 (gate_temp만) |
| 전문가 유형 | 범용 신경망 | PCEN (음향 물리 기반 정규화) |
| 전문가 크기 | 수백만 파라미터 | 160개 (전문가당) |
| 적용 규모 | 대규모 LLM/ViT | 초경량 (< 5,000 파라미터) |
| 미학습 잡음 대응 | 학습 분포에 편향 | 물리적 신호 기반, 미학습 잡음에도 대응 |

#### 1.4 SA-SSM 블록 구조 (도 3 참조)

**1.4.1 NanoMambaBlock 처리 파이프라인**

```
입력 x (B, L, d_model)
    ↓
[층 정규화(LayerNorm)]
    ↓
[입력 투영(in_proj)] — d_model → 2 * d_inner (바이어스 없음)
    ↓
분할: x_branch (B, L, d_inner) | z_gate (B, L, d_inner)
    ↓                                ↓
[깊이별 컨볼루션(DWConv1d)]          ↓
kernel_size=d_conv, groups=d_inner   ↓
    ↓                                ↓
[SiLU 활성화]                        ↓
    ↓                                ↓
[SA-SSM(x_branch, snr_mel)]          ↓
    ↓                                ↓
요소별 곱: y = ssm_out * SiLU(z_gate)
    ↓
[출력 투영(out_proj)] — d_inner → d_model (바이어스 없음)
    ↓
잔차 연결: output = projected + residual
```

**1.4.2 SA-SSM 내부 계산**

입력 투영:
```python
x_proj = W_x @ x_branch            # (B, L, 2*d_state + 1)
dt_raw = x_proj[..., :1]           # (B, L, 1) — 이산화 스텝 원시값
B_param = x_proj[..., 1:d_state+1] # (B, L, N) — 입력 행렬
C_param = x_proj[..., d_state+1:]  # (B, L, N) — 출력 행렬
```

SNR 변조 투영:
```python
snr_mod = W_snr @ snr_mel          # (B, L, d_state + 1)
dt_snr_shift = snr_mod[..., :1]    # (B, L, 1) — dt 변조량
B_gate = sigmoid(snr_mod[..., 1:]) # (B, L, N) — B 게이팅 마스크
```

이산화 스텝 계산:
```python
delta = softplus(W_dt @ (dt_raw + dt_snr_shift) + b_dt) + delta_floor
# delta_floor = 0.15 (register_buffer, 비학습)
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

순차 스캔 (recurrence):
```python
for t in range(L):
    h[t] = dA[t] * h[t-1] + dBx[t] + epsilon * x[t]   # 상태 업데이트
    y[t] = sum(h[t] * C[t], dim=state) + D * x[t]       # 출력 계산
```

여기서:
- A_log: HiPPO 초기화된 대각 행렬, A[n] = -(n + 0.5)
- D: 입력 직접 전달 (skip connection)
- epsilon = 0.1: 비학습 잔차 경로 (register_buffer)

**1.4.3 SA-SSM 파라미터 구성 (d_inner=24, d_state=4 기준)**

| 구성 요소 | 크기 | 파라미터 수 |
|-----------|------|-------------|
| x_proj (W_x) | 24 × 9 | 216 |
| snr_proj (W_snr) | 40 × 5 | 200 (+5 바이어스) |
| dt_proj (W_dt) | 1 × 24 | 24 (+24 바이어스) |
| A_log | 24 × 4 | 96 |
| D | 24 | 24 |
| alpha | 1 | 1 |
| delta_floor | 1 | 0 (register_buffer) |
| epsilon | 1 | 0 (register_buffer) |

#### 1.5 MoEFreq 모듈 (도 4 참조)

주파수 영역에서 3개 전문가를 SNR 통계에 따라 선택적으로 적용:

```python
# SNR 핑거프린트 추출
snr_mean = mean(snr_profile)                          # 전체 SNR 수준
snr_std = std(snr_profile)                             # SNR 변동성
snr_stats = [snr_mean, snr_std]                        # (B, 2)

# 게이트 계산
gate = softmax(Linear(2, 3)(snr_stats))                # (B, 3)

# 전문가 처리 (주파수 축 1D 컨볼루션)
out_narrow = Conv1d(1, 1, kernel_size=3, padding=1)(mag)  # 좁은 대역
out_wide = Conv1d(1, 1, kernel_size=7, padding=3)(mag)    # 넓은 대역
out_identity = mag                                         # 항등 (0 파라미터)

# 가중 결합
output = gate[0] * out_narrow + gate[1] * out_wide + gate[2] * out_identity
```

초기화 전략:
- 컨볼루션 필터: 시그모이드 (1.5) ≈ 0.82의 near-identity 초기화
- 게이트 바이어스: [0, 0, 1] → 학습 초기에는 항등 전문가를 선호하여 깨끗한 신호 보존

#### 1.6 일반화된 N-Expert 프레임워크 (도 5 참조)

상기 DualPCEN(2-Expert)과 MoEFreq(3-Expert)를 일반화하면:

```
StructuralMOE(input, physical_signal):
    routing_features = FeatureExtractor(physical_signal)
    gate_weights = GatingFunction(routing_features, N_experts)
    expert_outputs = [Expert_i(input) for i in range(N)]
    return sum(gate_weights[i] * expert_outputs[i] for i in range(N))
```

**라우팅 신호 옵션:**
| 신호 | 계산 | 학습 파라미터 | 적합 도메인 |
|------|------|---------------|-------------|
| 스펙트럴 평탄도 | GM/AM 비율 | 0 | 잡음 정상성 |
| SNR 통계 (평균, 표준편차) | 주파수별 SNR | 0 | 잡음 수준/변동 |
| 변조 스펙트럼 | 4-16Hz 에너지 비율 | 0 | 음성/비음성 |
| 켑스트럴 거리 | 켑스트럼 변동 | 0 | 채널 변동 |
| 주파수 에너지 분포 | 저/고주파 에너지 비 | 0 | 잡음 색상(color) |

**전문가 유형 옵션:**
| 전문가 | 파라미터 | 적합 용도 |
|--------|---------|-----------|
| PCEN (다양한 delta) | 160/전문가 | 음향 정규화 |
| 주파수 컨볼루션 (다양한 커널) | 3-8/전문가 | 주파수 선택 |
| 항등 변환 | 0 | 깨끗한 환경 |
| 학습 가능 마스크 | n_freq/전문가 | 주파수 가중 |
| 2D 컨볼루션 | 10-20/전문가 | 시간-주파수 패턴 |

이 프레임워크는 음성 처리에 한정되지 않으며, 비전(공간 주파수 기반 라우팅 + 필터 전문가), 센서 데이터(시간 통계 기반 라우팅), 자연어(토큰 복잡도 기반 라우팅) 등 다양한 도메인에 적용 가능하다.

### 제2 실시예: 소프트웨어 설계

#### 2.1 학습 파이프라인 (도 9 참조)

**2.1.1 데이터셋 및 전처리**

- 데이터셋: Google Speech Commands V2
- 클래스: 12개 ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence")
- 학습: 86,843 발화, 검증: 10,481 발화, 테스트: 11,005 발화
- 샘플링 레이트: 16kHz, 길이: 1초 (16,000 샘플)

**2.1.2 데이터 증강 (학습 시에만)**

```python
# 시간 이동: ±100ms (±1,600 샘플)
shift = random_int(-1600, 1600)
audio = roll(audio, shift)

# 볼륨 변조: ±20%
volume_factor = uniform(0.8, 1.2)
audio = audio * volume_factor

# 가우시안 잡음: 확률 0.3
if random() < 0.3:
    noise = randn_like(audio) * uniform(0.001, 0.015)
    audio = audio + noise
```

핵심: 잡음 증강은 매우 약한 수준(최대 -36dB SNR)의 가우시안 잡음만 사용. factory, white, babble 등의 실환경 잡음은 학습에 사용하지 않음. 잡음 강인성은 전적으로 구조적 MOE 아키텍처에 의해 달성.

**2.1.3 학습 하이퍼파라미터**

```python
optimizer = AdamW(lr=3e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=total_steps, eta_min=lr*0.01)
criterion = CrossEntropyLoss(label_smoothing=0.1)
gradient_clipping = 1.0
epochs = 30
batch_size = 128
```

**2.1.4 잡음 평가 프로토콜**

학습 완료 후, 다음 조건에서 평가:
- 잡음 유형: factory, white, babble, street, pink (5종)
- SNR 수준: -15, -10, -5, 0, 5, 10, 15 dB + clean (8 수준)
- 잔향: RT60 = 0.2, 0.4, 0.6, 0.8초 (4 수준)
- 총 평가 조건: 5 × 8 + 4 = 44 조건

잡음 생성:
- Factory: 50-250Hz 하모닉스 + 200-800Hz 착색 잡음 + 랜덤 임팩트 + 핑크 잡음
- Babble: 5-9명 화자 합성, 포먼트(730, 1090, 2440Hz) 포함
- Street: 20-200Hz 럼블 + 300-600Hz 경적 + 도로 잡음 + 엔진 진동
- Pink: FFT 기반 1/f 필터링
- White: 균일 가우시안

#### 2.2 추론 파이프라인

**2.2.1 오프라인 추론 (전체 발화 처리)**

```python
def inference(audio, model):
    # 1. STFT: (1, 16000) → (1, 257, 101)
    mag = stft(audio, n_fft=512, hop_length=160)

    # 2. SNR 추정: (1, 257, 101) → (1, 40, 101)
    snr_mel = snr_estimator(mag, mel_fb)

    # 3. MoEFreq (선택적): (1, 257, 101) → (1, 257, 101)
    mag = moe_freq(mag, snr_mel) if use_moe_freq else mag

    # 4. 멜 투영: (1, 257, 101) → (1, 40, 101)
    mel = mel_fb @ mag

    # 5. FreqDependentFloor: 저주파 보호
    mel = max(mel, freq_floor)

    # 6. DualPCEN: 이중 전문가 + 스펙트럴 평탄도 라우팅
    mel = dual_pcen(mel)

    # 7. InstanceNorm + 패치 투영: (1, 40, 101) → (1, 101, d_model)
    x = patch_proj(instance_norm(mel))

    # 8. SA-SSM 블록 × N: (1, 101, d_model) → (1, 101, d_model)
    for block in blocks:
        x = block(x, snr_mel)

    # 9. 분류: (1, 101, d_model) → (1, 12)
    x = classifier(layer_norm(mean(x, dim=time)))

    return argmax(x)
```

**2.2.2 스트리밍 추론 (실시간 프레임 처리)**

에지 SoC 배포 시 프레임 단위 처리:

```python
class StreamingNanoMamba:
    def __init__(self, model):
        # 상태 초기화
        self.pcen_smoother = zeros(2, n_mels)        # 2 PCEN 전문가 × 40 밴드
        self.ssm_hidden = zeros(n_layers, d_inner, d_state)  # SSM 은닉 상태
        self.noise_floor = None                       # SNR 잡음 바닥
        self.conv_buffer = zeros(n_layers, d_inner, d_conv-1) # Conv1d 버퍼
        self.frame_count = 0

    def process_frame(self, audio_frame):
        # 1프레임(160 샘플) 처리 → 1 시간 스텝 출력
        mag = stft_frame(audio_frame)                # (257,)

        if self.frame_count < 5:
            self.noise_floor = update_noise_init(mag)
        snr = compute_snr(mag, self.noise_floor)

        mel = mel_fb @ mag
        mel = max(mel, freq_floor)

        # DualPCEN (IIR 상태 유지)
        for expert_id in [0, 1]:
            self.pcen_smoother[expert_id] = (
                (1 - s[expert_id]) * self.pcen_smoother[expert_id]
                + s[expert_id] * mel)

        mel = dual_pcen_frame(mel, self.pcen_smoother)

        # SA-SSM (은닉 상태 유지)
        x = patch_proj(instance_norm(mel))
        for layer_id, block in enumerate(blocks):
            x, self.ssm_hidden[layer_id] = block.step(
                x, snr, self.ssm_hidden[layer_id],
                self.conv_buffer[layer_id])

        self.frame_count += 1
        return x  # 축적 후 분류
```

스트리밍 상태 메모리 (NanoMamba-Tiny 기준):
- PCEN smoother: 2 × 40 = 80 값
- SSM hidden: 2 × 24 × 4 = 192 값
- Conv buffer: 2 × 24 × 2 = 96 값
- 잡음 바닥: 257 값
- **총 스트리밍 상태: 625 값 × 1바이트(INT8) = 625 bytes**

#### 2.3 INT8 양자화

**2.3.1 양자화 방법**

모든 가중치와 활성화를 INT8(8비트 정수)로 양자화:
```
q = round(clamp(x / scale + zero_point, -128, 127))
x_dequant = (q - zero_point) * scale
```

**2.3.2 모델별 크기**

| 모델 | 파라미터 수 | FP32 (KB) | INT8 (KB) |
|------|-------------|-----------|-----------|
| NanoMamba-Tiny | 4,634 | 18.1 | 4.5 |
| NanoMamba-Small | 12,032 | 47.0 | 11.8 |
| NanoMamba-Base | ~28,000 | 109.4 | 27.3 |
| NanoMamba-Tiny-DualPCEN | ~4,955 | 19.4 | 4.8 |
| NanoMamba-Tiny-WS (3회 반복) | ~3,782 | 14.8 | 3.7 |
| NanoMamba-Tiny-MoEFreq | ~4,655 | 18.2 | 4.5 |

**2.3.3 양자화 인식 학습 (QAT)**

```python
# 학습 시 가짜 양자화(fake quantization) 적용
class FakeQuantize(Module):
    def forward(self, x):
        scale = x.abs().max() / 127
        x_quant = round(x / scale) * scale  # 양자화-역양자화
        return x_quant + (x - x_quant).detach()  # 직선 통과 추정기(STE)
```

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
|  |  - WEIGHT_ADDR: 가중치 기반 주소          |                   |
|  |  - RESULT: 분류 결과 (12 클래스 확률)      |                   |
|  +---+------+------+------+------+------+---+                   |
|      |      |      |      |      |      |                       |
|  +---v---+ +v----+ +v----+ +v----+ +v---v-+                    |
|  | STFT  | | SNR | | MOE | | PCEN | | SSM  |                    |
|  | Unit  | | Est | | Rtr | | ×2   | | Comp |                    |
|  +-------+ +-----+ +-----+ +------+ +------+                    |
|      |                                   |                       |
|  +---v-----------------------------------v---+                   |
|  |           Weight SRAM (4.5KB)              |                   |
|  |   INT8 가중치 + 바이어스 저장              |                   |
|  +--------------------------------------------+                   |
|      |                                                           |
|  +---v--------------------------------------------+              |
|  |          DMA 컨트롤러                           |              |
|  |  시스템 메모리 ↔ IP 내부 SRAM 전송             |              |
|  +---+--------------------------------------------+              |
|      |                                                           |
|  +---v--------------------------------------------+              |
|  |          전력 관리 모듈                          |              |
|  |  - Expert별 클럭 게이팅                         |              |
|  |  - 프레임 간 자동 슬립                          |              |
|  |  - Wake-on-Voice 인터럽트                       |              |
|  +--------------------------------------------+                  |
+================================================================+
```

#### 3.2 각 처리 유닛 상세 설계

**3.2.1 STFT 유닛**

```
입력: 160 샘플 (1 프레임, 10ms @ 16kHz)
출력: 257-포인트 크기 스펙트로그램

구성:
- 512-포인트 FFT 버터플라이 (in-place, radix-2)
- Hann 윈도우 ROM (256 값, 대칭이므로 절반 저장)
- 크기 계산: |X| = sqrt(Re^2 + Im^2) — CORDIC 근사 사용
- 연산량: 512 × log2(512) = 4,608 복소 곱셈/프레임
```

**3.2.2 SNR 추정 유닛**

```
입력: 257-포인트 크기 스펙트로그램
출력: 40-포인트 멜 SNR

구성:
- 잡음 바닥 레지스터 (257 × INT16)
- 나눗셈기 (16비트 반복 나눗셈 또는 LUT 역수)
- log10 LUT (256 엔트리, 8비트 인덱스)
- 멜 필터뱅크 행렬곱: 40 × 257 = 10,280 MAC
```

**3.2.3 MOE 라우터 유닛**

```
입력: 40-포인트 멜 에너지
출력: 게이트 가중치

DualPCEN 라우팅:
- 기하 평균: exp(sum(log(mel[i])) / 40) — log LUT + 나눗셈 + exp LUT
- 산술 평균: sum(mel[i]) / 40 — 누적기 + 시프트
- 비율: geo_mean / arith_mean — 나눗셈
- 시그모이드: LUT (256 엔트리)
- gate = sigmoid_LUT(gate_temp * (SF - 0.5))

MoEFreq 라우팅 (선택적):
- SNR 통계: mean + std (누적기 + 제곱근 근사)
- 선형 변환: 2 × 3 = 6 MAC
- 소프트맥스: 3-엔트리 exp + 나눗셈
```

**3.2.4 PCEN 유닛 (×2 또는 시분할)**

```
입력: 40-포인트 멜 에너지
출력: 40-포인트 PCEN 변환된 에너지

구성 (전문가당):
- IIR 평활기: 40 채널 병렬 1차 IIR
  smoother[i] = (1-s[i]) * smoother_prev[i] + s[i] * mel[i]
  연산: 40 × 3 = 120 MAC
- 거듭제곱 계산: (epsilon + smoother)^(-alpha) → log + 곱셈 + exp (LUT 기반)
  40 × 3 = 120 LUT 접근
- PCEN 변환: (mel * gain + delta)^r - delta^r → 추가 LUT 접근
  40 × 4 = 160 연산

구현 옵션:
A) 2개 PCEN 인스턴스 (면적 ×2, 지연 ×1)
B) 1개 PCEN + 시분할 (면적 ×1, 지연 ×2, 레지스터 파일로 상태 전환)
```

**3.2.5 SSM 계산 유닛**

```
입력: d_model=16 차원 시퀀스 + 40-포인트 SNR
출력: d_model=16 차원 시퀀스

핵심 연산 (1 타임스텝, 1 블록):

1. 입력 투영: x → 2*d_inner (16 → 48)
   MAC: 16 × 48 = 768

2. 깊이별 Conv1d: d_inner 채널 × kernel_size=3
   MAC: 24 × 3 = 72

3. x_proj: d_inner → 2*d_state+1 (24 → 9)
   MAC: 24 × 9 = 216

4. snr_proj: n_mels → d_state+1 (40 → 5)
   MAC: 40 × 5 = 200

5. dt_proj: 1 → d_inner (1 → 24)
   MAC: 1 × 24 = 24

6. 상태 업데이트: h = dA*h + dBx + epsilon*x
   MAC: 24 × 4 × 3 = 288

7. 출력 계산: y = (h * C).sum() + D * x
   MAC: 24 × 4 + 24 = 120

8. 게이팅: y * SiLU(z)
   MAC: 24

9. 출력 투영: d_inner → d_model (24 → 16)
   MAC: 24 × 16 = 384

총 MAC/타임스텝/블록: ~2,096
총 MAC/타임스텝 (2블록): ~4,192
총 MAC/발화 (101 타임스텝): ~423,392

+ 분류기: 16 × 12 = 192 MAC
총 MAC/발화: ~423,584 ≈ 0.42 MMAC
```

#### 3.3 버스 인터페이스 설계

**3.3.1 AXI4-Lite 슬레이브 (레지스터 접근)**

```
주소 맵:
0x00: CTRL      [RW] - Bit0: Start, Bit1: Stop, Bit2: Reset
0x04: STATUS    [RO] - Bit0: Busy, Bit1: Done, Bit2: Error, Bit3: IRQ
0x08: CONFIG    [RW] - d_model(8b) | d_state(4b) | n_layers(4b) | mode(8b)
0x0C: WEIGHT_ADDR [RW] - 가중치 SRAM 시작 주소 (32비트)
0x10: AUDIO_ADDR  [RW] - 오디오 버퍼 시작 주소 (32비트)
0x14: RESULT[0]   [RO] - 클래스 0 확률 (INT8)
0x18: RESULT[1]   [RO] - 클래스 1 확률 (INT8)
...
0x44: RESULT[11]  [RO] - 클래스 11 확률 (INT8)
0x48: ARGMAX      [RO] - 최종 분류 결과 (4비트)
0x4C: GATE_TEMP   [RW] - DualPCEN gate temperature (FP16)
0x50: DELTA_FLOOR [RW] - SA-SSM delta floor (FP16)
0x54: EPSILON     [RW] - SA-SSM epsilon (FP16)

인터럽트:
- IRQ_DONE: 1 발화 처리 완료
- IRQ_KW_DETECTED: 키워드 검출 (확률 > 임계값)
```

**3.3.2 AXI4-Stream 인터페이스 (오디오 스트리밍)**

```
입력 스트림 (Slave):
- TDATA[15:0]: 오디오 샘플 (16비트 PCM)
- TVALID, TREADY: 핸드셰이크
- TLAST: 프레임 경계 (160 샘플마다)

출력 스트림 (Master, 선택적):
- TDATA[95:0]: 12 클래스 확률 (각 8비트)
- TVALID, TREADY: 핸드셰이크
- TLAST: 1 발화 완료
```

**3.3.3 AHB-Lite 대안 인터페이스**

JieLi pi32v2 등 AXI 미지원 SoC를 위한 대안:

```
신호:
- HADDR[31:0]: 주소
- HWDATA[31:0]: 쓰기 데이터
- HRDATA[31:0]: 읽기 데이터
- HWRITE, HTRANS, HSIZE, HBURST: 제어
- HREADY, HRESP: 응답

동일한 레지스터 맵을 AHB-Lite로 래핑.
AXI ↔ AHB 브릿지를 통해 양쪽 인터페이스 지원 가능.
```

#### 3.4 전력 관리 설계

**3.4.1 Expert별 클럭 게이팅**

```
DualPCEN의 라우팅 결과에 따라 미사용 전문가의 클럭을 게이팅:

if gate > 0.95:  # 거의 전적으로 정상 잡음 전문가 사용
    disable_clock(PCEN_Expert_1)  # 비정상 전문가 클럭 OFF
elif gate < 0.05:  # 거의 전적으로 비정상 잡음 전문가 사용
    disable_clock(PCEN_Expert_2)  # 정상 전문가 클럭 OFF
else:
    enable_clock(PCEN_Expert_1)
    enable_clock(PCEN_Expert_2)
```

**3.4.2 프레임 간 자동 슬립**

```
프레임 처리 주기: 10ms (16kHz, 160 샘플 홉)
프레임 처리 시간: ~0.5ms (추정, 50MHz 클럭 기준)
유휴 시간: ~9.5ms (95% 유휴)

자동 슬립 시퀀스:
1. 프레임 처리 완료
2. 결과를 레지스터에 기록
3. 클럭 게이팅 진입 (CG 셀 활성화)
4. 다음 프레임 도착 시 인터럽트로 기상
```

**3.4.3 Wake-on-Voice (VAD 연동)**

```
외부 VAD 모듈과 연동:
1. 대기 모드: IP 블록 전체 클럭 OFF, VAD만 동작
2. VAD 트리거: IRQ → IP 블록 클럭 ON → 버퍼된 오디오 처리
3. 키워드 검출: IRQ_KW_DETECTED → 메인 프로세서 기상
4. 미검출: 타임아웃 후 대기 모드 복귀
```

### 제4 실시예: SoC 통합 (도 7 참조)

#### 4.1 Qualcomm QCC5171/QCC5181 통합

```
+--------------------------------------------------+
|  Qualcomm QCC517x SoC                             |
|                                                    |
|  +------------+  +---------------------------+    |
|  | Cortex-M4F |  | Dual Kalimba DSP @ 120MHz |    |
|  | (앱 프로세서)|  |  +-------------------+   |    |
|  +------+-----+  |  | NanoMamba 펌웨어   |   |    |
|         |        |  | (DSP Extension)    |   |    |
|         |        |  | - SA-SSM 스캔      |   |    |
|         |        |  | - DualPCEN 라우팅  |   |    |
|         |        |  | - INT8 MAC 활용    |   |    |
|         |        |  +-------------------+   |    |
|         |        +---------------------------+    |
|         |                  |                       |
|  +------v------------------v----+                 |
|  |       Sensing Hub (QSH)       |                 |
|  |  - 항상-켜짐 AI 서브시스템     |                 |
|  |  - < 1mA 소비 전력             |                 |
|  |  - VAD → NanoMamba → 기상      |                 |
|  +-------------------------------+                 |
+--------------------------------------------------+
```

통합 방법:
- Qualcomm Voice & Music Extension Program을 통해 Kalimba DSP 확장으로 NanoMamba 펌웨어 탑재
- Sensing Hub의 VAD 트리거 후 NanoMamba 키워드 검출 실행
- 4.5KB INT8 가중치는 Kalimba DSP의 내부 SRAM에 상주
- 총 코드 + 가중치: < 25KB, Kalimba DSP SRAM 용량 내

#### 4.2 JieLi AC79 시리즈 통합

```
+--------------------------------------------------+
|  JieLi AC79 SoC                                   |
|                                                    |
|  +-------------------------------------------+    |
|  | Dual-Core pi32v2 FP DSP @ 320MHz          |    |
|  |                                             |    |
|  |  Core 0: 오디오 코덱 + BT 스택             |    |
|  |                                             |    |
|  |  Core 1: NanoMamba 추론                     |    |
|  |  +-----------------------------------+     |    |
|  |  | AC79NN SDK 오퍼레이터 매핑:        |     |    |
|  |  | - Dense: in_proj, out_proj, x_proj|     |    |
|  |  | - Conv1d: DWConv, MoEFreq        |     |    |
|  |  | - Custom: SSM scan (C 루프)       |     |    |
|  |  | - LUT: sigmoid, softplus, exp     |     |    |
|  |  +-----------------------------------+     |    |
|  +-------------------------------------------+    |
|                    |                               |
|  +---------+  +---v------+  +--------+            |
|  | 73KB    |  | MATH &   |  | Flash  |            |
|  | SRAM    |  | FLOAT    |  | 512KB  |            |
|  |         |  | 가속기    |  |        |            |
|  +---------+  +----------+  +--------+            |
+--------------------------------------------------+
```

통합 방법:
- 전용 NPU가 없으므로, pi32v2 CPU의 소프트웨어 DSP 라이브러리로 구현
- AC79NN SDK의 Dense/Conv 오퍼레이터를 활용하여 행렬곱/컨볼루션 매핑
- SSM 순차 스캔은 C 언어 최적화 루프로 구현
- MATH & FLOAT 하드웨어 가속기를 활용하여 부동소수점 연산 가속
- 73KB SRAM에 가중치(4.5KB) + 코드(~15KB) + 스트리밍 상태(~1KB) 충분히 적재

#### 4.3 BES BES2700/BES2800 통합

```
+--------------------------------------------------+
|  BES2700YP SoC                                    |
|                                                    |
|  +-------------------+  +----------------------+  |
|  | Cortex-M55        |  | Sensor Hub 서브시스템 |  |
|  | (메인 프로세서)    |  |                      |  |
|  |                   |  | +--------+ +--------+|  |
|  |                   |  | |STAR-MC1| |BECO NPU||  |
|  |                   |  | |(제어)  | |(MAC)   ||  |
|  |                   |  | +---+----+ +---+----+|  |
|  |                   |  |     |          |      |  |
|  +--------+----------+  | +---v----------v---+ |  |
|           |              | | NanoMamba 분할:   | |  |
|           |              | | CPU: SSM scan,   | |  |
|           |              | |      라우팅 로직  | |  |
|           |              | | NPU: MatMul,     | |  |
|           |              | |      Conv, PCEN   | |  |
|           |              | +------------------+ |  |
|           |              | +------------------+ |  |
|           |              | | VAD 엔진         | |  |
|           |              | +------------------+ |  |
|           |              +----------------------+  |
|           |                        |               |
|  +--------v------------------------v-----------+   |
|  |          4MB 공유 SRAM                       |   |
|  |  가중치(4.5KB) + 상태 + 오디오 버퍼         |   |
|  +---------------------------------------------+   |
+--------------------------------------------------+
```

통합 방법:
- BECO NPU에 행렬곱/컨볼루션 오프로드
- STAR-MC1 CPU에서 SSM 순차 스캔 + MOE 라우팅 로직 실행
- Sensor Hub 서브시스템의 VAD → NanoMamba → Cortex-M55 기상 체인
- 4MB 공유 SRAM에서 CPU, NPU, BT 모듈 간 제로카피 데이터 공유
- AHB(STAR-MC1)/AXI(Cortex-M55) 인터페이스로 IP 블록 접근

#### 4.4 버스 무관(Bus-Agnostic) 래퍼 설계

다양한 SoC 버스에 대응하기 위한 얇은 적응 계층:

```
+--------------------------------------------------+
|  Bus-Agnostic Wrapper                             |
|                                                    |
|  +--------+  +--------+  +----------+            |
|  | AXI4   |  | AHB    |  | Custom   |            |
|  | Slave  |  | Lite   |  | (pi32v2) |            |
|  | I/F    |  | I/F    |  | I/F      |            |
|  +---+----+  +---+----+  +----+-----+            |
|      |           |             |                   |
|  +---v-----------v-------------v---+              |
|  |    Protocol Translation Layer    |              |
|  |  - 주소 디코딩                    |              |
|  |  - 데이터 정렬                    |              |
|  |  - 핸드셰이크 변환                |              |
|  +---------------+------------------+              |
|                  |                                  |
|  +---------------v------------------+              |
|  |       NanoMamba IP Core          |              |
|  |  (버스 독립적 내부 인터페이스)    |              |
|  +----------------------------------+              |
+--------------------------------------------------+
```

이 래퍼를 통해 동일한 NanoMamba IP 코어가 AXI, AHB, 또는 독점 버스를 사용하는 SoC에 통합 가능하다.

### 제5 실시예: 비전 및 다중 모달 적용

본 발명의 구조적 MOE 프레임워크는 음성에 한정되지 않으며, 다음과 같은 도메인에 적용 가능하다.

#### 5.1 비전 (이미지 분류)

```
이미지 → 패치 분할 → 공간 주파수 분석 → MOE 라우팅
                                         |
                      +------------------+------------------+
                      |                  |                  |
                  [저주파 전문가]    [고주파 전문가]    [항등 전문가]
                  (부드러운 영역)   (에지/텍스처)     (원본 보존)
                      |                  |                  |
                      +------------------+------------------+
                                         ↓
                                   SA-SSM 처리
                                         ↓
                                      분류
```

라우팅 신호: 공간 주파수 에너지 분포 (2D FFT의 저/고주파 에너지 비율)

#### 5.2 센서 데이터 (이상 감지)

```
센서 시계열 → 통계 분석 → MOE 라우팅
                          |
              +-----------+-----------+
              |           |           |
          [정상 전문가]  [과도 전문가]  [이상 전문가]
          (정상 패턴)   (전환 구간)   (비정상 패턴)
              |           |           |
              +-----------+-----------+
                          ↓
                    SA-SSM 처리
                          ↓
                     이상 판정
```

라우팅 신호: 시간 윈도우 통계 (평균 변화율, 분산, 첨도)

---

## [특허 청구범위]

### 독립항

**【청구항 1】**

잡음 강인 음성 인식 시스템으로서,
(a) 입력 음향 신호로부터 잡음 정상성 지표(noise stationarity indicator)를 산출하는 신호 분석부;
(b) 서로 다른 잡음 유형에 특화된 파라미터로 구성된 복수의 특징 정규화 전문가(feature normalization expert);
(c) 상기 잡음 정상성 지표에 기반하여 상기 복수의 전문가의 출력을 선택적으로 결합하는 라우팅 모듈; 및
(d) 상기 결합된 출력을 처리하되, 상태공간모델(State Space Model)의 이산화 스텝(discretization step) 및 입력 행렬(input matrix)이 신호 대 잡음비(SNR) 추정치에 의해 변조되는 스펙트럴-인식 상태공간 처리부
를 포함하는, 잡음 강인 음성 인식 시스템.

**【청구항 2】**

잡음 강인 키워드 인식 방법으로서,
(a) 입력 오디오 신호로부터 스펙트럴 평탄도(Spectral Flatness)를 산출하는 단계;
(b) 상기 입력 오디오 신호를 서로 다른 오프셋(delta) 파라미터를 가진 적어도 두 개의 PCEN(Per-Channel Energy Normalization) 전문가를 통해 처리하는 단계;
(c) 상기 스펙트럴 평탄도에 기반하여 학습 가능한 게이트 온도(gate temperature) 파라미터를 사용하는 시그모이드 게이트를 통해 상기 전문가 간 라우팅을 수행하는 단계; 및
(d) 라우팅된 출력을, SNR 변조된 이산화 스텝 및 SNR 게이팅된 입력 행렬을 가진 선택적 상태공간모델(Selective SSM)을 통해 처리하는 단계
를 포함하는, 잡음 강인 키워드 인식 방법.

**【청구항 3】**

잡음 강인 음성 인식을 위한 하드웨어 가속기로서,
(a) 오디오 프레임을 주파수 영역으로 변환하는 STFT(Short-Time Fourier Transform) 연산 유닛;
(b) 주파수별 신호 대 잡음비를 추정하는 SNR 추정 유닛;
(c) 신호 유도 라우팅 신호(signal-derived routing signal)를 산출하고 복수의 처리 전문가의 출력을 블렌딩하는 MOE 라우팅 유닛;
(d) SNR 변조된 이산화 스텝과 SNR 게이팅된 입력을 가진 상태공간모델 연산을 수행하는 SSM 계산 유닛; 및
(e) SoC 통합을 위한 버스 인터페이스(AXI 또는 AHB 호환)
를 포함하는, 잡음 강인 음성 인식을 위한 하드웨어 가속기.

**【청구항 4】**

잡음 적응형 선택적 상태공간모델(noise-adaptive selective state space model)로서,
(a) 주파수별 SNR 추정치에 의해 변조되는 이산화 스텝(delta)으로서, 상기 이산화 스텝은 SNR 투영값과 시간 특징 투영값의 합에 소프트플러스(softplus) 활성화를 적용하여 산출되는, 이산화 스텝;
(b) SNR에서 유도된 시그모이드 게이트에 의해 게이팅되는 입력 행렬(B);
(c) 극단적 저 SNR 환경에서도 최소 이산화 스텝을 보장하는 비학습 이산화 스텝 하한(delta floor)으로서, 학습 과정에서 옵티마이저에 의해 변경되지 않는 아키텍처 상수로 구현된, 이산화 스텝 하한; 및
(d) 모든 SNR 의존적 게이팅과 무관하게 최소 정보 흐름을 보장하는 비학습 잔차 경로 계수(epsilon)로서, 상태 업데이트 시 게이팅되지 않은 입력을 직접 상태에 가산하는, 잔차 경로 계수
를 포함하는, 잡음 적응형 선택적 상태공간모델.

**【청구항 5】**

항상-켜짐 키워드 인식을 위한 에지 SoC 시스템으로서,
(a) 디지털 신호 처리기(DSP) 또는 신경 처리 유닛(NPU);
(b) 상기 DSP 또는 NPU 상에 구현된 잡음 강인 음성 인식 모듈로서, 신호 유도 라우팅을 가진 구조적 전문가 혼합 아키텍처를 구현하는, 잡음 강인 음성 인식 모듈; 및
(c) 전문가별 클럭 게이팅을 제공하는 전력 관리 유닛
을 포함하는, 에지 SoC 시스템.

### 종속항

**【청구항 6】**

제1항에 있어서,
상기 잡음 정상성 지표는 멜 밴드 에너지의 기하 평균과 산술 평균의 비율로 정의되는 스펙트럴 평탄도(Spectral Flatness)이며, 상기 라우팅 신호의 산출에 학습 가능 파라미터를 사용하지 않는 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 7】**

제1항에 있어서,
상기 복수의 특징 정규화 전문가 중 제1 전문가는 0.5 이상의 오프셋(delta) 파라미터를 가져 오프셋 우세 모드(offset-dominant mode)로 동작하고, 제2 전문가는 0.1 이하의 오프셋 파라미터를 가져 AGC 우세 모드(AGC-dominant mode)로 동작하는 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 8】**

제1항에 있어서,
상기 시스템은 저주파 멜 밴드에 대해 비학습 지수 감쇠(exponential decay) 에너지 하한(frequency-dependent floor)을 적용하는 주파수 의존 에너지 하한 모듈을 더 포함하는 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 9】**

제1항에 있어서,
상기 라우팅 모듈은 학습 가능한 온도 파라미터(gate temperature)와 0.5의 임계값(threshold)을 가진 시그모이드 게이트 함수를 포함하며, 상기 게이트 함수의 출력에 따라 전문가 출력의 가중 선형 결합을 수행하는 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 10】**

제1항에 있어서,
상기 시스템은 주파수 영역 전문가 혼합 모듈을 더 포함하며, 상기 주파수 영역 전문가 혼합 모듈은 적어도 좁은 대역 컨볼루션 전문가, 넓은 대역 컨볼루션 전문가, 및 항등 변환 전문가를 포함하고, SNR 통계(평균, 표준편차)에 의해 조건부로 라우팅되는 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 11】**

제4항에 있어서,
상기 이산화 스텝 하한(delta floor)은 0.1 이상의 값을 가지며, 상태공간모델의 학습 과정에서 옵티마이저에 의해 업데이트되지 않는 레지스터 버퍼(register buffer)로 구현되어, 극단적 잡음 환경에서도 SSM의 완전 동결(freezing)을 구조적으로 방지하는 것을 특징으로 하는, 잡음 적응형 선택적 상태공간모델.

**【청구항 12】**

제4항에 있어서,
상기 잔차 경로 계수(epsilon)는 0.05 이상의 값을 가지며, 레지스터 버퍼로 구현되고, 상태 업데이트 수식 h[t] = A_bar * h[t-1] + dBx[t] + epsilon * x[t]에서 epsilon * x[t] 항으로 표현되어, SNR 의존적 게이팅(dBx)과 무관한 무조건적(unconditional) 정보 흐름 경로를 제공하는 것을 특징으로 하는, 잡음 적응형 선택적 상태공간모델.

**【청구항 13】**

제2항에 있어서,
상기 방법은 단일 SSM 블록을 N회(N ≥ 2) 반복 실행하여 N층 깊이의 처리를 수행하되, 고유 학습 파라미터는 단일 블록분만을 유지하는 가중치 공유 단계를 더 포함하는 것을 특징으로 하는, 잡음 강인 키워드 인식 방법.

**【청구항 14】**

제3항에 있어서,
상기 하드웨어 가속기는 상기 MOE 라우팅 유닛의 라우팅 결정에 기반하여 미사용 전문가에 대한 클럭 공급을 차단하는 전문가별 클럭 게이팅 회로를 포함하는 것을 특징으로 하는, 하드웨어 가속기.

**【청구항 15】**

제5항에 있어서,
상기 에지 SoC는 Kalimba DSP를 포함하는 블루투스 오디오 SoC, pi32v2 프로세서를 포함하는 블루투스 오디오 SoC, 또는 BECO NPU를 포함하는 블루투스 오디오 SoC 중 적어도 하나인 것을 특징으로 하는, 에지 SoC 시스템.

**【청구항 16】**

제1항에 있어서,
상기 시스템의 총 학습 가능 파라미터 수는 5,000개 미만이며, INT8 양자화 시 모델 크기가 5 킬로바이트 미만인 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 17】**

제2항에 있어서,
상기 방법에 의한 모델은 잡음이 없는 깨끗한 음성 데이터만으로 학습되며, 잡음 강인성은 잡음 증강 학습 데이터 없이 상기 구조적 전문가 혼합 아키텍처의 구조적 특성에 의해 달성되는 것을 특징으로 하는, 잡음 강인 키워드 인식 방법.

**【청구항 18】**

제1항에 있어서,
상기 복수의 특징 정규화 전문가는 N개(N ≥ 2)의 PCEN(Per-Channel Energy Normalization) 전문가로 구성되며, 각 전문가는 멜 밴드당 서로 다른 평활 계수(s), 이득 지수(alpha), 오프셋(delta), 및 압축 지수(r) 파라미터를 가지는 것을 특징으로 하는, 잡음 강인 음성 인식 시스템.

**【청구항 19】**

제3항에 있어서,
상기 버스 인터페이스는 레지스터 접근을 위한 AXI4-Lite 슬레이브 인터페이스와 오디오 데이터 스트리밍을 위한 AXI4-Stream 인터페이스 중 적어도 하나를 지원하는 것을 특징으로 하는, 하드웨어 가속기.

**【청구항 20】**

제5항에 있어서,
상기 전력 관리 유닛은 오디오 프레임 간 유휴 시간 동안 상기 음성 인식 모듈의 클럭을 자동으로 차단하는 자동 슬립 기능과, 외부 음성 활동 검출(VAD) 트리거에 의해 상기 음성 인식 모듈을 활성화하는 웨이크-온-보이스(wake-on-voice) 기능을 포함하는 것을 특징으로 하는, 에지 SoC 시스템.

---

## [요약서]

**발명의 명칭:** 구조적 전문가 혼합 네트워크를 이용한 잡음 강인 음성 인식 시스템 및 방법과 이를 위한 하드웨어 가속기

**요약:**

본 발명은 구조적 전문가 혼합(MOE) 네트워크를 이용하여 다양한 잡음 환경에서 강인한 음성 인식을 수행하는 시스템, 방법 및 하드웨어 가속기에 관한 것이다. 본 발명의 시스템은 스펙트럴 평탄도(Spectral Flatness) 등 물리적 음향 신호에 기반한 무파라미터 라우팅으로 복수의 PCEN(Per-Channel Energy Normalization) 전문가를 선택적으로 결합하는 이중 전문가 모듈과, 주파수별 신호 대 잡음비(SNR)에 의해 이산화 스텝(dt) 및 입력 행렬(B)이 변조되는 스펙트럴-인식 상태공간모델(SA-SSM)을 포함한다. 비학습 아키텍처 상수인 이산화 스텝 하한(delta floor)과 잔차 경로 계수(epsilon)에 의해 극단적 잡음 환경에서도 구조적으로 최소 정보 흐름이 보장된다. 전체 모델은 5,000개 미만 파라미터, INT8 양자화 시 5KB 미만으로, AXI/AHB 버스 인터페이스를 갖춘 하드웨어 가속기 IP 블록으로 구현되어 블루투스 오디오 SoC 등 에지 프로세서에 통합 가능하다.

**대표도:** 도 1

---

## [발명자]

최진호 (Jin Ho Choi, Ph.D.)

## [출원인]

(출원인 정보 기재)

---

## [부록: 코드-청구항 추적 테이블]

| 청구항 | 구성 요소 | 소스 파일 | 라인 번호 | 핵심 상수 |
|--------|-----------|-----------|-----------|-----------|
| 1(a) | Spectral Flatness 라우팅 | nanomamba.py | 310-319 | SF = GM/AM |
| 1(b) | DualPCEN 전문가 | nanomamba.py | 276-293 | delta=2.0, 0.01 |
| 1(c) | 게이트 함수 | nanomamba.py | 316-319 | gate_temp=5.0 |
| 1(d) | SA-SSM | nanomamba.py | 529-686 | delta_floor=0.15 |
| 4(a) | dt SNR 변조 | nanomamba.py | 624-641 | softplus + delta_floor |
| 4(b) | B SNR 게이팅 | nanomamba.py | 643-645 | alpha=0.5 |
| 4(c) | delta_floor | nanomamba.py | 584-586 | 0.15 (register_buffer) |
| 4(d) | epsilon | nanomamba.py | 587-588 | 0.1 (register_buffer) |
| 10 | MoEFreq | nanomamba.py | 409-480 | k=3, k=7, identity |
| 8 | FreqDependentFloor | nanomamba.py | 331-359 | 0.05*exp(-3.0*ratio) |
| 13 | Weight Sharing | nanomamba.py | 878-902 | n_repeats |

---

## [부록: 선행기술 대비표]

| 구분 | 본 발명 | Google MOE (US20200279150A1) | Qualcomm KWD (US20210005181A1) | 표준 Mamba |
|------|---------|------------------------------|--------------------------------|-----------|
| 라우팅 방식 | 물리적 신호 (SF) | 학습된 선형+소프트맥스 | 다단계 전력 관리 | 없음 |
| 라우팅 파라미터 | 1개 (gate_temp) | 입력차원 × 전문가수 | N/A | N/A |
| 잡음 적응 | 구조적 (클린만 학습) | 학습 데이터 의존 | 별도 AEC/NR 필요 | 없음 |
| SSM SNR 변조 | dt + B 모두 변조 | N/A | N/A | 없음 |
| 구조적 보장 | delta_floor + epsilon | 없음 | 없음 | 없음 |
| 모델 크기 | < 5KB INT8 | 수 GB | 수 MB | 수 MB |
| 에지 SoC 통합 | AXI/AHB IP 블록 | 클라우드 서버 | 전용 DSP | N/A |

---

*본 명세서는 임시 특허 명세서로, 최종 출원 시 변리사 검토 및 도면 첨부가 필요합니다.*
*공지예외주장(Grace Period Declaration)이 필요하며, GitHub 첫 공개일로부터 12개월 이내에 출원해야 합니다.*
