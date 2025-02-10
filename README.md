# Abusing Dimensionality Collapse for Adversarial Training

We define:

- **Generator** \( G(\text{noise}) \) → **fake images**
- **RankNet** \( D(x) \) → **embedding for each image in a batch**, measure **SmoothRank** across the batch.

## Objectives

- **\(D\) wants**:
  - Real batch → SmoothRank \(\approx 1\)
  - Fake batch → SmoothRank \(\approx 0\)
- **\(G\) wants**:
  - Fake batch → SmoothRank \(\approx 1\) (to fool \(D\))

### Intuitive Explanation

- Real data is expected to be **dimension-preserving**.
- Fake data is encouraged to undergo **dimension-collapse** such that the SmoothRank of the generated batch is indicative of real images.

---
```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2025}
}
```
