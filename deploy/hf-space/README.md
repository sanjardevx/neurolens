---
title: NeuroLens API
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Brain tumor MRI analysis API (research, not a medical device)
---

# NeuroLens API

The backend of [NeuroLens](https://github.com/sanjardevx/neurolens): FastAPI + ONNX Runtime on CPU. Classifies
brain MRI, segments the tumor in 3D and measures it; a doctor reviews every result in the dashboard.

Research and portfolio project, **not a medical device**. Deployed automatically from the `master` branch after CI
passes; do not edit files here.
