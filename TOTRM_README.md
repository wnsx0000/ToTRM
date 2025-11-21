# ToTRM (Tree-of-Thought Recursive Model)

ToTRM은 TRM에 Tree-of-Thought 개념을 아키텍처 수준에서 적용한 모델입니다.

## 핵심 아이디어

TRM이 Chain-of-Thought를 layer 수준에서 구현했다면, ToTRM은 **Tree-of-Thought**를 구현합니다.

### 기존 TRM의 동작

```
각 supervision step에서:
  H_cycle마다:
    L_cycle 1: z_L = f(z_L, z_H + input)
    L_cycle 2: z_L = f(z_L, z_H + input)
    ...
    L_cycle n: z_L = f(z_L, z_H + input)
    z_H = f(z_H, z_L)  # 최종 업데이트
```

### ToTRM의 동작

```
각 supervision step에서:
  H_cycle마다:
    L_cycle 1: z_L = f(z_L, z_H + input) → 2개로 분기
    L_cycle 2: z_L = f(z_L, z_H + input) → 4개로 분기
    L_cycle 3: z_L = f(z_L, z_H + input) → 8개로 분기
    ...
    L_cycle n-1: z_L = f(z_L, z_H + input) → 2^(n-1)개로 분기
    L_cycle n: 2^(n-1)개의 z_L을 하나로 병합
    z_H = f(z_H, z_L_merged)  # 병합된 상태로 업데이트
```

## 아키텍처 세부사항

### 트리 분기 (Branching)

각 L_cycle에서 상태를 복제하여 이진 트리를 형성합니다:
- Step 1: 1개 → 2개
- Step 2: 2개 → 4개
- Step 3: 4개 → 8개
- Step n-1: 2^(n-2) → 2^(n-1)개

구현은 batch dimension을 활용하여 효율적으로 처리:
```python
def _branch_state(self, z: torch.Tensor) -> torch.Tensor:
    # [batch_size * current_width, ...] → [batch_size * current_width * 2, ...]
    return z.repeat_interleave(2, dim=0)
```

### 트리 병합 (Merging)

마지막 L_cycle에서 모든 분기를 하나로 합칩니다. 세 가지 방법 지원:

1. **Mean Pooling** (기본값)
   ```python
   z_merged = z.mean(dim=tree_dimension)
   ```

2. **Max Pooling**
   ```python
   z_merged = z.max(dim=tree_dimension)[0]
   ```

3. **Learned Weighted Sum**
   ```python
   weights = softmax(learned_weights)
   z_merged = (z * weights).sum(dim=tree_dimension)
   ```

## 하이퍼파라미터

### ToTRM 전용 파라미터

- `tree_branching_steps`: 분기를 수행할 L_cycle 수 (n-1)
  - 예: `L_cycles=6`, `tree_branching_steps=5`이면 처음 5번 분기, 마지막 1번 병합
  - 최종 트리 너비: 2^`tree_branching_steps`

- `tree_merge_method`: 병합 방법
  - `mean`: 평균 (기본값, 안정적)
  - `max`: 최댓값 (가장 좋은 경로 선택)
  - `learned_weighted`: 학습 가능한 가중치

### 기존 TRM 파라미터

- `H_cycles`: High-level 순환 횟수
- `L_cycles`: Low-level 순환 횟수 (분기 + 병합 포함)
- `hidden_size`, `num_heads`, `expansion`: Transformer 설정

## 사용 방법

### 1. 데이터 준비

```bash
# Sudoku 4x4 (빠른 테스트용)
uv run python dataset/build_sudoku_4x4_dataset.py

# 또는 다른 데이터셋
uv run python dataset/build_rubik2x2_dataset.py
```

### 2. 모델 학습

```bash
# Sudoku 4x4로 빠른 실험
./train_totrm_sudoku4x4.sh

# 또는 직접 설정
uv run python pretrain.py \
    arch=totrm \
    data_paths="[data/sudoku4x4]" \
    arch.tree_branching_steps=3 \
    arch.tree_merge_method=mean \
    arch.L_cycles=4 \
    global_batch_size=256 \
    epochs=1500
```

### 3. 모델 평가

```bash
uv run python evaluate.py \
    --data-path data/sudoku4x4/ \
    --config checkpoints/totrm/<run-name>/all_config.yaml \
    --checkpoint checkpoints/totrm/<run-name>/final_step_X/model.pt
```

## 메모리 및 계산량

### 계산 복잡도

TRM 대비 ToTRM의 추가 비용:
- **메모리**: 최대 2^`tree_branching_steps` 배 (트리 너비만큼)
- **계산**: 거의 2^`tree_branching_steps` 배 (분기된 상태 모두 처리)

예시:
- `tree_branching_steps=3`: 8배 메모리/계산
- `tree_branching_steps=5`: 32배 메모리/계산
- `tree_branching_steps=6`: 64배 메모리/계산

### 최적화 팁

1. **작은 tree_branching_steps로 시작**
   - 2-3으로 시작하여 효과 확인
   - 성능 향상이 있으면 점진적으로 증가

2. **batch_size 조정**
   - ToTRM은 내부적으로 batch를 확장하므로
   - `global_batch_size`를 TRM 대비 줄여야 함
   - 예: TRM batch_size=512 → ToTRM batch_size=256 또는 128

3. **더 작은 모델 사용**
   - `hidden_size`를 줄이거나 `L_layers`를 줄여서 보상

## TRM과의 차이점 요약

| 특징 | TRM | ToTRM |
|------|-----|-------|
| 추론 경로 | 단일 경로 (Chain) | 다중 경로 (Tree) |
| L_cycle당 상태 수 | 1개 | 1 → 2^n개 |
| 병합 단계 | 없음 | 매 H_cycle마다 |
| 메모리 사용량 | 기준 | 2^n배 |
| 탐색 능력 | 순차적 | 병렬적 |

## 코드 구조

```
models/recursive_reasoning/totrm.py
├── ToTRM (메인 모델)
├── ToTRM_Inner (핵심 로직)
│   ├── _branch_state()      # 트리 분기
│   ├── _merge_tree()        # 트리 병합
│   └── forward()            # Tree-of-Thought 순전파
├── ToTRM_ReasoningModule
└── ToTRM_Block
```

## 실험 아이디어

1. **병합 방법 비교**
   - mean vs max vs learned_weighted
   - 어떤 방법이 가장 효과적인지?

2. **분기 깊이 조절**
   - 얕은 트리(2-3 분기) vs 깊은 트리(5-6 분기)
   - 성능 vs 계산량 trade-off

3. **비대칭 트리**
   - 모든 L_cycle에서 동일하게 분기하지 않고
   - 특정 단계에서만 더 많이 분기

4. **분기 다양성 증가**
   - 단순 복제 대신 노이즈 추가
   - 서로 다른 초기화로 분기

## 참고

- 기반 모델: TRM (Tiny Recursive Model)
- 영감: Tree-of-Thought prompting
- 구현: TRM 코드를 최소한으로 수정하여 ToT 개념 적용
