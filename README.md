# **NeuroVision: Translating EEG Signals into Text and Images**
NeuroVision is a research project focused on decoding visual stimuli from 64-channel EEG signals. This repository contains the complete PyTorch implementation of a novel hybrid architecture that translates brain activity into descriptive text captions, which are then used to reconstruct the original visual scene using a pre-trained text-to-image model.
The project's core contribution is a robust, direct-pathway architecture and a decoupled two-stage training curriculum that successfully learns to generate semantically relevant captions from complex EEG data.

## **Architecture & Pipeline**
The final, successful architecture avoids the "pseudo-image" bottleneck by creating a direct pathway from a domain-specific EEG encoder to the BLIP-2 Q-Former.
![alt text](<reports/NeuroVision Pipeline.jpg>)

![alt text](<reports/NeuroVision-Final Year Project Research Paper.jpg>)

## **Key Features**
* Novel Direct-Pathway Architecture: A custom hybrid EEGTransformerEncoder connects directly to the BLIP-2 Q-Former, bypassing the vision encoder to avoid domain mismatch issues.

* Decoupled Two-Stage Training: A robust curriculum that first trains the from-scratch EEG encoder (Stage 1) and then freezes it to gently fine-tune the pre-trained Q-Former (Stage 2), ensuring stability and preventing catastrophic forgetting.

* Combined Loss Function: Utilizes a powerful, weighted combination of Cross-Entropy, InfoNCE (alignment), and Semantic Similarity losses to guide the model.

* End-to-End Inference: The final script takes raw spectrograms and produces final reconstructed images for qualitative evaluation.

## **The Project Journey & Key Findings**
This project followed a rigorous experimental process that is as valuable as the final result.

1. **Initial Goal & Failure:** The project began with the goal of direct EEG-to-Image reconstruction using models like GANs and Stable Diffusion. While these models could generate high-quality images, the core failure was a semantic accuracy bottleneck—the generated images were not conceptually related to the EEG stimulus.

2. **Pivotal Discovery:** An intermediate experiment using contrastive learning and a t-SNE visualization revealed that distinct semantic clusters do exist within the high-channel EEG data. This proved that the signal was present but the method of extraction was flawed.

3. **Architectural Evolution:** This insight led to a pivot towards EEG-to-Text translation using BLIP-2.
* Initial attempts to treat spectrograms as "pseudo-images" failed, as the model learned to describe the visual artifacts of the spectrograms themselves (e.g., "green and purple lines").
* This led to the final, successful architecture: a powerful, from-scratch EEGTransformerEncoder that feeds its features directly to the BLIP-2 Q-Former.
4. **Training Strategy Refinement:** A simple end-to-end training approach proved unstable and led to "catastrophic forgetting," where the language model's weights were damaged. This was solved by developing the decoupled two-stage curriculum with differential learning rates, which finally enabled stable and effective learning.

## **Results**
The final model, trained with the two-stage strategy, demonstrates a clear and successful learning trajectory.
* **Quantitative:** The model achieves a final best validation loss of 16.42, a decrease of over 45% from the initial loss of 29.37, proving that the model is effectively learning.
* **Qualitative:** The model successfully moves beyond generating random phrases to producing coherent sentences that show emergent semantic relevance to the ground truth.

## **Sample Visual Reconstruction**
![alt text](<reports/Final Recdonstructions.png>)

## **How to Use**
1. **Data Preparation**
    * Download the "AllJoined" dataset and preprocess it into spectrograms. Place the data in the /content/final_lightweight_17k directory (or update the path in the config).
    * Ensure the metadata.csv file is present.
    * Run the data statistics script (calculate_stats.py, from our notebook) to generate spec_mean.pt and spec_std.pt.
    * Run the caption generation script (generate_captions.py) once to create the train_gt_captions.json and val_gt_captions.json files.
    
2. **Training**
    * Modify the TRAIN_CONFIG class in the main training script to set your desired hyperparameters. Then,
    * Run the eeg-text-image.ipynb notebook to execute the training process as we have or with your own custom changes.
  
3. **Inference**
    * Open the inference.ipynb notebook.
    * Update the BEST_MODEL_PATH in the INFERENCE_CONFIG to point to your best saved checkpoint.
    * Set your desired BATCH_SIZE.
    * Run the script to generate and save a grid of reconstructed images.

4. **Future Work**

This project serves as a strong foundation for several future research directions:
    * Training for a larger number of epochs to allow the Q-Former to fully converge.
    * Extensive hyperparameter tuning of learning rates and loss weights.
    * A potential "Stage 3" of training to fine-tune the final layers of the large language model.
    * Using the generated high-quality captions as prompts for text-to-image models to close the loop on the original EEG-to-Image reconstruction goal.


## MindBigData Dataset download link:
https://www.mindbigdata.com/opendb/imagenet.html

## Alljoined Dataset download link:
https://huggingface.co/datasets/Alljoined/05_125

https://linktr.ee/alljoined1

## Google Drive link for the Two-stage BLIP2 fine-tuned model with custom EEG feature extractor:
https://drive.google.com/file/d/1bz0AXqew60FmHut-GaEh_OuS7wPR0BAo/view?usp=sharing


