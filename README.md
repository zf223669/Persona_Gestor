# Persona_Gestor (Source Code is coming soon!)
## Speech-driven Personalized Gesture Synthetics: Harnessing Automatic Fuzzy Feature Inference
# Abstract
Speech-driven gesture generation is an emerging field within virtual human creation. However, a significant challenge lies in accurately determining and processing the multitude of input features (such as acoustic, semantic, emotional, personality, and even subtle unknown features). Traditional approaches, reliant on various explicit feature inputs and complex multimodal processing, constrain the expressiveness of resulting gestures and limit their applicability.


To address these challenges, we present **Persona-Gestor**, a novel end-to-end generative model designed to generate highly personalized 3D full-body gestures solely relying on raw speech audio. The model combines a fuzzy feature extractor and a non-autoregressive Adaptive Layer Normalization (AdaLN) transformer diffusion architecture. The fuzzy feature extractor harnesses a fuzzy inference strategy that automatically infers implicit, continuous fuzzy features. These fuzzy features, represented as a unified latent feature, are fed into the AdaLN transformer. The AdaLN transformer introduces a conditional mechanism that applies a uniform function across all tokens, thereby effectively modeling the correlation between the fuzzy features and the gesture sequence. This module ensures a high level of gesture-speech synchronization while preserving naturalness. Finally, we employ the diffusion model to train and infer various gestures. Extensive subjective and objective evaluations on the Trinity, ZEGGS, and BEAT datasets confirm our model's superior performance to the current state-of-the-art approaches.


**Persona-Gestor** improves the system's usability and generalization capabilities, setting a new benchmark in speech-driven gesture synthesis and broadening the horizon for virtual human technology.

![image](./Image/Summary.png)
Each pose depicted is personalized gestures generated solely relying on raw speech audio. Persona-Gestor offers a versatile solution, bypassing complex multimodal processing and thereby enhancing user-friendliness.


For clarity, our contributions are summarized as follows:
1) **We pioneering introduce the fuzzy feature inference strategy that enables driving a wider range of personalized gesture synthesis from speech audio alone, removing the need for style labels or extra inputs.**
This fuzzy feature extractor improves the usability and the generalization capabilities of the system. To the best of our knowledge, it is the first approach that uses fuzzy features to generate co-speech personalized gestures.
2) **We combined AdaLN transformer architecture within the diffusion model to enhance the Modeling of the gesture-speech interplay.** We demonstrate that this architecture can generate gestures that achieve an optimal balance of natural and speech synchronization.
3) **Extensive subjective and objective evaluations reveal our model superior outperform to the current state-of-the-art approaches.** These results show the remarkable capability of our method in generating credible, speech-appropriateness, and personalized gestures.

Presentation page: https://zf223669.github.io/Diffmotion-v2-website/

Showcases(Goodele Drive): https://drive.google.com/drive/folders/1YWQaS4BZZsYvPAKychusLJOhT--dfv2G

Showcases(Baidu Cloud Disk): https://pan.baidu.com/s/1Mt6K-mVUnAcLk6Ol8NALQA?pwd=1234
