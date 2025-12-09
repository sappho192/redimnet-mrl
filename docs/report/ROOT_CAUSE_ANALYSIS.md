# Root Cause Analysis: Validation Loss Problem

**Date**: 2025-12-03
**Issue**: Validation loss remains high (~18) and increases during Stage 2 fine-tuning
**Status**: ⚠️ **Critical Design Flaw Identified**

---

## TL;DR

**The validation loss is fundamentally broken** because we're using an AAMSoftmax classifier trained on VoxCeleb2 speakers (5,994 classes) to evaluate VoxCeleb1 speakers (1,251 completely different people).

This is like training a face classifier on Dataset A and evaluating it by forcing Dataset B faces into Dataset A categories - it doesn't make sense.

---

## The Fundamental Problem

### Current Setup (Broken)

```python
# Training
train_speakers = VoxCeleb2  # Speaker IDs: id01000 - id09272 (5,994 speakers)
classifier = AAMSoftmax(num_classes=5994, embed_dim=256)

# Validation
val_speakers = VoxCeleb1  # Speaker IDs: id10001 - id11251 (1,251 DIFFERENT speakers)
val_loss = classifier(embedding, label)  # ← BROKEN!
```

### What Happens During Validation

**Step 1**: Load VoxCeleb1 speaker (e.g., id10270)
```python
audio, label = val_dataset[0]  # label = 0 (첫 번째 VoxCeleb1 speaker)
```

**Step 2**: Extract embedding
```python
embedding = model(audio)  # [1, 256]
```

**Step 3**: Compute loss with AAMSoftmax
```python
# AAMSoftmax weight matrix: [5994, 256]
# - Row 0: VoxCeleb2 speaker id01000
# - Row 1: VoxCeleb2 speaker id01001
# ...
# - Row 5993: VoxCeleb2 speaker id09272

# Validation label = 0 (VoxCeleb1 speaker id10001)
# → Tries to match VoxCeleb1 id10001 to VoxCeleb2 id01000
# → Completely wrong mapping!

loss = classifier(embedding, label=0)  # Meaningless!
```

**Why it's broken:**
- VoxCeleb1 speaker id10001 ≠ VoxCeleb2 speaker id01000
- They are completely different people
- Forcing one person's embedding to match another person's class
- Loss is meaningless for validation

---

## Why Validation Loss is High (~18)

### Training Loss Analysis

**Epoch 2 (Best validation):**
- Train Loss: 11.89
- Val Loss: 18.00

**Why train loss is lower:**
- Classifier has learned to separate 5,994 VoxCeleb2 speakers
- AAMSoftmax weight matrix optimized for these specific people
- Model can correctly assign speaker IDs

**Why val loss is high:**
- Trying to classify 1,251 NEW speakers using 5,994 old speaker weights
- Random chance with 5,994 classes ≈ log(5994) ≈ 8.7
- Getting ~18 because embeddings are forced into wrong categories
- Loss is not measuring "speaker recognition quality" but "misclassification penalty"

---

## Why Val Loss Increases When Backbone Unfreezes

### Stage 1 (Frozen Backbone, Epochs 1-15)

```
Train Loss: 12.93 → 11.54 (decreasing)
Val Loss: 18.02 → 18.02 (stable)
```

**What's happening:**
- Projection head learns to output embeddings that work with pretrained backbone
- Training speakers get correctly classified (train loss decreases)
- Validation speakers still random (val loss stable but high)
- **Validation loss is not actually "improving" or "getting worse" - it's just noise**

### Stage 2 (Unfrozen Backbone, Epoch 16+)

```
Epoch 16: Train 8.16, Val 18.18
Epoch 20: Train 3.05, Val 18.33
```

**What's happening:**
- Backbone + projection optimize HARD for training speakers
- Training loss drops dramatically (model memorizing training speakers)
- Validation loss increases slightly because:
  - Embeddings now extremely specialized for VoxCeleb2 speakers
  - Even more incompatible with VoxCeleb1 speakers
  - But this doesn't mean "overfitting" in the traditional sense
  - It means **we're measuring the wrong thing**

---

## The Real Issue: Wrong Validation Metric

### Current Validation (Classification Loss) ❌

```python
# This measures: "Can you classify VoxCeleb1 speakers using VoxCeleb2 classes?"
# Answer: No, and it shouldn't!
val_loss = AAMSoftmax(embedding, voxceleb1_label)
```

**Problems:**
1. Speaker IDs don't overlap
2. Classification is not the end goal
3. Loss is meaningless for speaker verification

### Correct Validation (Verification Pairs) ✅

```python
# This measures: "Can you distinguish same vs different speakers?"
# This is what we actually care about!

# Load verification pairs
pairs = [
    (audio1, audio2, is_same_speaker),  # (tensor, tensor, 0 or 1)
    ...
]

# Compute embeddings
emb1 = model(audio1)
emb2 = model(audio2)

# Compute similarity
similarity = cosine_similarity(emb1, emb2)

# Compute EER (Equal Error Rate)
eer = compute_eer(similarities, labels)
```

**This is the standard evaluation for speaker recognition!**

---

## Why We Didn't Catch This Earlier

### 1. Common Misunderstanding
- Assumed classification loss on validation set is valid
- Didn't realize speaker IDs are completely disjoint
- Standard practice in classification (ImageNet) doesn't apply here

### 2. Speaker Recognition is Different
- **Classification**: Assign to one of K known classes
- **Verification**: Determine if two utterances are from same speaker
- VoxCeleb is designed for verification, not classification

### 3. Training Still Works
- Training loss is valid (same speakers in train set)
- Model IS learning useful embeddings
- Just can't evaluate them with classification loss on new speakers

---

## Evidence This is the Root Cause

### 1. Validation Loss Magnitude

Random chance with 5,994 classes:
```
Expected random loss ≈ -log(1/5994) ≈ 8.7
Observed validation loss ≈ 18.0
```

**Why is it 18, not 8.7?**
- Embeddings are being pushed toward WRONG speaker classes
- Angular margin penalty (AAMSoftmax margin=0.2) adds penalty
- Model is confident but wrong → high loss

### 2. Validation Loss Never Improves

```
Epochs 1-15 (Stage 1): Val loss stuck at 18.00-18.04
```

If validation was measuring real quality:
- Should improve as embeddings get better
- Should correlate with embedding quality

Instead:
- Stays constant because it's measuring classification of wrong speakers
- Slight variations are just noise

### 3. Training Loss Pattern is Normal

```
Stage 1: 12.93 → 11.54 (gradual, expected)
Stage 2: 11.54 → 3.05 (rapid, expected for unfreezing)
```

This is actually **normal behavior** for fine-tuning!
- The issue is NOT overfitting
- The issue is **invalid validation metric**

---

## What We Thought vs Reality

### What We Thought (Wrong)

❌ "Model is overfitting because val loss increases while train loss decreases"
- Assumed val loss is a valid metric
- Applied regularization (dropout, weight decay)
- Extended Stage 1 duration
- None of this addresses the real problem

### Reality (Correct)

✅ "Validation loss is measuring the wrong thing"
- Train loss: Valid (classification of VoxCeleb2 speakers)
- Val loss: Invalid (classification of VoxCeleb1 speakers using VoxCeleb2 classes)
- Model is probably learning fine
- We just can't tell because our metric is broken

---

## Correct Solutions

### Option 1: Use Same Speakers for Train/Val (Quick Fix)

**Approach**: Split VoxCeleb2 utterances, not speakers
```yaml
data:
  # Use 80% of VoxCeleb2 utterances for training
  train_dataset: VoxCeleb2 (utterances 0-873K)

  # Use 20% of VoxCeleb2 utterances for validation
  val_dataset: VoxCeleb2 (utterances 873K-1092K)

  # Same 5,994 speakers, different utterances
```

**Pros:**
- Classification loss becomes valid
- Easy to implement
- Standard for classification tasks

**Cons:**
- Not testing generalization to new speakers
- Less realistic evaluation
- Misses point of speaker verification

### Option 2: Implement EER Validation (Recommended)

**Approach**: Use verification pairs for validation
```python
# Load VoxCeleb1 verification pairs
pairs = load_voxceleb1_pairs()  # (audio1, audio2, is_same)

# Compute EER
eer = compute_eer(model, pairs)
```

**Pros:**
- Measures actual speaker verification performance
- Tests generalization to new speakers
- Industry standard metric

**Cons:**
- Requires implementation
- Need verification pair files
- More complex

### Option 3: Projection-Only Training (Practical)

**Approach**: Never unfreeze backbone
```yaml
advanced:
  freeze_backbone_epochs: 9999  # Never unfreeze
training:
  num_epochs: 30  # Shorter training
```

**Rationale:**
- Pretrained backbone already excellent (EER 0.57% on VoxCeleb1)
- Only need to learn MRL projection
- Much less prone to "overfitting" (smaller capacity)
- Validation loss still broken, but less dramatic divergence

**Pros:**
- Faster training (30 epochs vs 100)
- More stable
- Less capacity = less memorization risk
- Pretrained features are already good

**Cons:**
- Might not achieve best possible performance
- Not fully utilizing pretrained model potential

---

## Recommendation

**Immediate**: Use Option 3 (Projection-only)
- Stop current training
- Set `freeze_backbone_epochs: 9999`
- Train for 30 epochs
- Save best checkpoint based on... train loss? (since val is broken)

**Short-term**: Implement Option 2 (EER validation)
- Add proper verification pair evaluation
- Use VoxCeleb1 test pairs
- Measure real EER metrics

**Long-term**: Consider Option 1 for development
- Quick iteration with valid metrics
- Switch to EER for final evaluation

---

## How to Implement EER Validation

### 1. Load Verification Pairs

```python
# veri_test2.txt format:
# 1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
# 0 id10309/0cYFdtyWVds/00001.wav id10296/q-8fGPszYYI/00001.wav

pairs = []
with open('voxceleb1_test_pairs.txt') as f:
    for line in f:
        label, path1, path2 = line.strip().split()
        pairs.append((path1, path2, int(label)))
```

### 2. Compute Similarities

```python
similarities = []
labels = []

for path1, path2, is_same in pairs:
    emb1 = model(load_audio(path1))
    emb2 = model(load_audio(path2))

    sim = cosine_similarity(emb1, emb2)
    similarities.append(sim)
    labels.append(is_same)
```

### 3. Compute EER

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(labels, similarities)
fnr = 1 - tpr
eer_threshold = thresholds[np.argmin(np.abs(fpr - fnr))]
eer = fpr[np.argmin(np.abs(fpr - fnr))]

print(f"EER: {eer*100:.2f}%")
```

---

## Lessons Learned

### 1. Domain-Specific Validation Matters

**Classification** (ImageNet):
- Train/val can use same classes
- Classification loss is valid metric
- Dropout/regularization prevents overfitting

**Verification** (Speaker Recognition):
- Train/val MUST use different speakers (for generalization test)
- Classification loss is INVALID for new speakers
- Need similarity-based metrics (EER, cosine similarity)

### 2. Validation Metric Must Match End Goal

**Our goal**: Speaker verification (same/different)
**Current metric**: Speaker classification (which of 5,994?)
**Mismatch**: Metric doesn't measure what we care about

### 3. Pre-trained Models Need Careful Handling

**Pretrained ReDimNet:**
- Already trained on VoxCeleb2 with classification
- Embeddings are already good for verification
- Fine-tuning risks breaking what already works
- Safer to just train projection head (MRL adapter)

---

## Action Items

### Immediate (Before Next Training Run)

- [ ] Decide on validation strategy (Options 1, 2, or 3)
- [ ] Update config accordingly
- [ ] Stop current training (still overfitting)
- [ ] Document decision in report

### Short-term

- [ ] Implement EER validation function
- [ ] Add VoxCeleb1 test pairs to repo
- [ ] Create proper evaluation script
- [ ] Benchmark against baseline ReDimNet

### Long-term

- [ ] Add early stopping based on EER
- [ ] Compare MRL dimensions (64D vs 256D EER)
- [ ] Document best practices for speaker recognition MRL
- [ ] Consider different loss functions (e.g., Triplet Loss, Contrastive)

---

## Comparison: Classification vs Verification Training

### Classification Approach (Current)

```python
# Loss
loss = AAMSoftmax(embedding, speaker_id)

# Problem
- Requires same speakers in train/val
- Doesn't match verification task
- Over-optimizes for seen speakers
```

### Verification Approach (Better)

```python
# Loss
loss = TripletLoss(anchor, positive, negative)
# or
loss = ContrastiveLoss(emb1, emb2, is_same)

# Benefits
- Works with new speakers
- Directly optimizes similarity metric
- Better for speaker verification
```

**Note**: AAMSoftmax is still widely used and effective, but requires same speakers in train/val OR proper EER evaluation.

---

## Technical Details

### Current Training Architecture

```
Input: Audio waveform
   ↓
ReDimNet Backbone (pretrained on VoxCeleb2)
   ↓
MRL Projection [64, 128, 192, 256]
   ↓
MatryoshkaLoss
   ├→ AAMSoftmax_64D  (5994 classes)
   ├→ AAMSoftmax_128D (5994 classes)
   ├→ AAMSoftmax_192D (5994 classes)
   └→ AAMSoftmax_256D (5994 classes)
```

**Training**: ✅ Works (speakers in classifier)
**Validation**: ❌ Broken (speakers NOT in classifier)

### What Validation Loss Actually Measures

**Not**: "How well does the model recognize speakers?"
**Actually**: "How badly do VoxCeleb1 embeddings fail to match VoxCeleb2 classes?"

**Analogy**:
- Training: Learning to recognize faces of people in Group A
- Validation: Trying to classify faces of people in Group B as someone from Group A
- Result: Everyone in Group B gets misclassified (high loss)
- This doesn't mean the face recognition is bad - it means the metric is wrong!

---

## Why This Explains Everything

### 1. High Initial Validation Loss (18.0)

**Not because**: Model is bad at speaker recognition
**Actually**: VoxCeleb1 speakers forced into VoxCeleb2 classes
**Expected**: High loss for wrong classification task

### 2. Validation Loss Doesn't Improve (Stage 1)

**Not because**: Projection head isn't learning
**Actually**: Better embeddings don't help when speakers aren't in classifier
**Evidence**: Train loss improves, showing model IS learning

### 3. Validation Loss Increases (Stage 2)

**Not because**: Model is overfitting
**Actually**:
- Embeddings becoming MORE discriminative for VoxCeleb2 speakers
- This makes them LESS likely to accidentally match VoxCeleb1 speakers
- Higher loss, but this might mean BETTER embeddings!

**Paradox**: Improving the model for its actual task (speaker verification) makes the broken validation metric worse!

---

## Statistical Analysis

### Expected Validation Loss Under Random Embeddings

If embeddings were random:
```python
# 5,994 classes, uniform distribution
baseline_loss = -log(1/5994) = 8.7

# With angular margin (AAMSoftmax margin=0.2, scale=30)
# Margin makes loss higher when confident but wrong
expected_random_loss ≈ 12-15
```

### Observed Validation Loss

```
Observed: 18.0
Random baseline: 12-15
```

**Interpretation**: Model is "confidently wrong"
- Embeddings are strong and discriminative
- But discriminating for wrong set of speakers
- Confident predictions on wrong classes = high loss

### If Model Was Actually Bad

```
Bad embeddings → Low confidence → Loss closer to random baseline (12-15)
Good embeddings for wrong task → High confidence → Loss higher than random (18)
```

**Conclusion**: Val loss of 18 might actually indicate the model is learning good embeddings!

---

## Experiments to Confirm

### Test 1: Compute Cosine Similarities

```python
# On validation set
emb1, emb2 = model(same_speaker_pair)
sim_same = cosine_similarity(emb1, emb2)

emb3, emb4 = model(different_speaker_pair)
sim_diff = cosine_similarity(emb3, emb4)

print(f"Same speaker: {sim_same}")  # Should be high (>0.7)
print(f"Different speaker: {sim_diff}")  # Should be low (<0.3)
```

**If this works**: Model IS good, validation metric is just broken

### Test 2: Compute EER on Validation Pairs

```python
eer = evaluate_eer(model, voxceleb1_pairs)
print(f"EER: {eer*100:.2f}%")  # Should be <2% if model is good
```

**If EER is good**: Confirms model quality, invalidates classification loss

---

## Corrected Understanding

### What's Actually Happening

**Stage 1 (Frozen Backbone):**
- ✅ Projection head learns to adapt pretrained features for MRL
- ✅ Training loss decreases (expected)
- ⚠️ Validation loss stays high (but this is OK - metric is broken)
- **No overfitting** - just invalid metric

**Stage 2 (Unfrozen Backbone):**
- ✅ Full model optimizes for speaker discrimination
- ✅ Training loss decreases rapidly (expected for fine-tuning)
- ⚠️ Validation loss increases slightly (embeddings more specialized)
- **Still not overfitting** - metric is still broken, but embeddings might be better!

### True Overfitting Would Look Like

If model was truly overfitting:
- Same speaker pairs: Similarity decreases
- Different speaker pairs: Similarity increases
- Verification accuracy: Drops
- We haven't measured these - so we don't actually know!

---

## Conclusion

### The Real Problem

**Not**: Overfitting
**Actually**: Invalid validation metric

We've been optimizing the wrong thing:
- Added regularization → Doesn't help (not the problem)
- Extended Stage 1 → Doesn't help (not the problem)
- Need to fix the metric → This will help

### The Real Solution

**Option A (Quick)**: Don't unfreeze backbone
```yaml
freeze_backbone_epochs: 9999
num_epochs: 30
```
- Safest approach
- Pretrained features already good
- Just learn MRL adapter

**Option B (Proper)**: Implement EER validation
```python
def validate():
    eer = compute_eer(model, voxceleb1_pairs)
    return eer
```
- Measures actual performance
- Standard practice
- More complex to implement

**Option C (Development)**: Same speakers for train/val
```yaml
val_dataset: VoxCeleb2 (different utterances)
```
- Valid classification loss
- Quick iteration
- Less realistic

---

## Next Steps

1. **Stop current training** (still using broken metric)
2. **Choose validation strategy** (A, B, or C above)
3. **Implement chosen approach**
4. **Restart training with valid metrics**
5. **Re-evaluate what "good performance" looks like**

---

**Report Status**: ✅ Root cause identified
**Severity**: Critical - invalidates all validation metrics
**Impact**: All previous training runs cannot be properly evaluated
**Recommendation**: Implement proper EER validation or use projection-only training
