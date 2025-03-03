---
title: "[2016 CVPR] Deep residual learning for image recognition"
date: 2025-03-03
categories: [Perception]
tags: [ResNet, PaperReview]
---

본 논문은 딥러닝에서 매우 깊은 신경망을 더 효과적으로 학습할 수 있도록 **residual learning framework를 제안**하는 논문임

# Introduction
Deep learning에서 network가 깊어질수록 강력한 표현력을 제공하지만, network가 깊어질 수록 학습이 어려워지는 문제가 발생함  
Network가 깊어질수록(deep) 강력한 표현력을 제공하는 이유는 레이어가 깊을수록 더 심도깊은 추상화로 전체적인 해석이 가능한 feature들을 추출하기 때문
- Low-level feature: edge, line, color 등의 낮은 추상화로 추출할 수 있는 정보
- Mid-level feature: texture, more complex feature과 같이 보다 깊은 추상화로 추출할 수 있는 정보
- High-level feature: 구체적인 개체나 장면의 의미론적 구조와 같이 매우 깊은 추상화로 추출할 수 있는 정보

즉, 차선과 같은 간단한 task들은 MLP(Multi-layer perceptron)으로 충분히 검출가능하지만, 이미지 내의 전체적인 구조나 의미를 찾기위해선 매우 깊은 신경망을 통해 심도깊은 feature들을 추출해야함 
Network가 깊어질수록 더 높은 수준의 feature를 추출할 수 있는 이유는 layer를 거치면 정보가 점점 더 추상화되기 때문  
1. **계측적 특징 학습**: 딥러닝 신경망에서는 각 layer가 이전 layer 출력을 입력으로 받아, 점차 더 복잡한 feature들을 학습함
    - 초반 layer들은 이미지의 기본적인 패턴, 예를 들면 가장자리(edge)나 단순한 모양 같은 기본적인 시각 패턴을 학습하고, 이 정보를 중간 layer로 전달함
2. **추상화의 진행**: 네트워크가 깊어지면서 중간 layer들은 이전 레이어에서 학습한 단순한 패턴을 조합하여 더 복잡한 패턴들을 학습함
    - 여러 가장자리들을 조합하여 패턴이나 모양같은 중간수준의 feature들을 학습함
3. **고수준의 특징 학습**: 가장 깊은 layer에서는 이전의 layer들에서 학습한 복잡한 패턴들을 다시 결합하여 구체적인 객체나 개념들에 대한 고수준의 feature들을 학습함
    - 고수준의 feature들은 특정 동물의 얼굴이나 자동차같은 전체적인 개체의 구조를 인식함
4. **기계적 추상화의 표현력**: Layer가 많아질수록 network는 점차적인 추상화를 통해 단순한 픽셀정보를 넘어서 의미론적이고 인식가능한 객체로 변환할 수 있는 능력이 향상됨
    - Layer가 얕을때는 low-level feature까지만 학습가능하지만, layer가 많아지면서 정보가 더 복잡하게 결합되고, 이를 통해 더 높은 수준의 표현이 가능해짐

#### But!
Network가 깊어질수록 성능이 좋아질 것 같지만, 여기서 발생하는 문제는 *network의 깊이가 깊어질수록 학습 정확도가 saturation에 도달한 후 점차 감소하는 degradation이 발생*하는데 이는 overfitting이 아닌 최적화(optimization)의 어려움에서 기인함

